"""协调 Skill 热读、动态命令、专属工具、主运行时和 fork 执行。"""

from __future__ import annotations

from collections.abc import Callable

from mycode.agent.cancellation import CancellationToken
from mycode.commands.models import (
    ForkSkillSubmission,
    InlineSkillSubmission,
    SkillSubmission,
)
from mycode.commands.registry import CommandRegistry
from mycode.errors import MyCodeError
from mycode.models.messages import ChatMessage
from mycode.models.tools import ToolSource
from mycode.skills.catalog import SkillCatalog
from mycode.skills.commands import build_skill_commands
from mycode.skills.fork import SkillForkRunner
from mycode.skills.loader import SkillLoader
from mycode.models.skills import (
    ActiveSkill,
    SkillCatalogSnapshot,
    SkillDefinition,
    SkillInvocation,
    SkillInvocationResult,
    SkillMode,
    SkillReloadIssue,
    SkillReloadReport,
)
from mycode.skills.runtime import SkillRuntime
from mycode.skills.subprocess_tool import SkillSubprocessTool
from mycode.tools.registry import SkillToolRegistration, ToolRegistry


class SkillService:
    """提供应用和 LoadSkill 共用的唯一 Skill 调用入口。

    CLI 创建本类后，斜杠命令、自然语言 LoadSkill 和 /skill 管理命令都通过
    它工作。它负责在执行前重读文件，并让 Catalog、动态命令、专属工具和
    当前活动 SOP 使用同一份有效定义。
    """

    def __init__(
        self,
        loader: SkillLoader,
        catalog: SkillCatalog,
        runtime: SkillRuntime,
        fork_runner: SkillForkRunner,
        command_registry: CommandRegistry,
        tool_registry: ToolRegistry,
        main_history: Callable[[], tuple[ChatMessage, ...]],
    ) -> None:
        """连接 Skill 系统需要同步更新的现有组件。

        Args:
            loader: 扫描三层目录并在执行前热读入口文件的加载器。
            catalog: 保存当前有效版本和候选诊断的共享目录。
            runtime: 当前主会话的 inline Skill 运行状态。
            fork_runner: 独立模式使用的临时 Agent 执行器。
            command_registry: 保存静态命令和动态 Skill 命令的双层注册表。
            tool_registry: 保存基础工具和 Skill 专属工具的共享注册表。
            main_history: fork 开始前读取主会话消息快照的函数。

        Returns:
            None。调用 install_initial 后才会把扫描结果开放给用户和 Agent。
        """

        # 每次显式或自动调用都从这里重读当前入口文件。
        self._loader = loader
        # Runtime、命令和管理界面查询同一个可替换 Catalog 对象。
        self._catalog = catalog
        # inline 激活、资源范围、白名单和会话旁路元数据由它维护。
        self._runtime = runtime
        # fork 调用只通过此对象运行，不复用主 SessionManager。
        self._fork_runner = fork_runner
        # reload 会整批替换这里的 Skill 命令层。
        self._command_registry = command_registry
        # reload 会整批替换这里的 Skill 专属工具层。
        self._tool_registry = tool_registry
        # fork 仅在真正开始时读取主历史，避免保存过期快照。
        self._main_history = main_history

    def install_initial(self, snapshot: SkillCatalogSnapshot) -> None:
        """校验并安装启动扫描得到的全部 Skill。

        Args:
            snapshot: Loader 完成三级覆盖选择后的候选和有效定义。

        Returns:
            None。安装成功后命令、工具和轻量目录会同时可用。

        Raises:
            MyCodeError: 名字冲突或 allowedTools 引用了不存在的工具。
        """

        self._install_snapshot(snapshot)

    def scan_and_install(self) -> None:
        """在 MCP 工具发现完成后扫描并安装三层 Skill。

        Returns:
            None。调用成功后轻量目录、动态命令和专属工具同时生效。

        Raises:
            MyCodeError: 最终白名单、命令名或工具名校验失败。
        """

        self.install_initial(self._loader.scan())

    def submission_for(
        self,
        name: str,
        arguments: str,
        display_text: str,
    ) -> SkillSubmission | None:
        """把动态斜杠命令转换成 Application 可以识别的提交类型。

        Args:
            name: CommandParser 解析出的规范 Skill 名。
            arguments: 命令名之后保留大小写的原始参数。
            display_text: UI 和主会话需要保存的简短斜杠输入。

        Returns:
            inline 或 fork 提交；命令注册后 Skill 已被删除时返回 None。
        """

        skill = self._catalog.get(name)
        if skill is None:
            return None
        submission_type = (
            InlineSkillSubmission
            if skill.mode is SkillMode.INLINE
            else ForkSkillSubmission
        )
        return submission_type(skill.name, arguments, display_text)

    async def invoke(
        self,
        invocation: SkillInvocation,
        cancellation: CancellationToken,
    ) -> SkillInvocationResult:
        """热读目标 Skill，并按 inline 或 fork 模式执行。

        Args:
            invocation: 目标名字、参数和主会话显示文字。
            cancellation: 用户取消当前前台操作时触发的令牌。

        Returns:
            inline 激活结果，或已经完成的 fork 最终报告。

        Raises:
            MyCodeError: Skill 未知、入口已删除或新定义无法安全安装。
        """

        skill, warning = self._latest_definition(invocation.name)
        if skill.mode is SkillMode.INLINE:
            self._runtime.activate_inline(skill, invocation.arguments)
            return SkillInvocationResult(
                skill=skill,
                display_text=invocation.display_text,
                warning=warning,
            )
        final_text = await self._fork_runner.run(
            skill,
            invocation.arguments,
            self._main_history(),
            cancellation,
        )
        return SkillInvocationResult(
            skill=skill,
            display_text=invocation.display_text,
            final_text=final_text,
            warning=warning,
        )

    async def load_for_agent(
        self,
        name: str,
        arguments: str,
        cancellation: CancellationToken,
    ) -> SkillInvocationResult:
        """处理 Agent 发起的 LoadSkill 调用。

        Args:
            name: Agent 从轻量目录中选择的 Skill 名。
            arguments: 需要替换 $ARGUMENTS 的用户意图补充。
            cancellation: 工具执行被取消时使用的令牌。

        Returns:
            inline 激活确认所需结果，或 fork 最终报告；不包含 SOP 正文。
        """

        cleaned_name = name.strip()
        cleaned_arguments = arguments.strip()
        display = f"/{cleaned_name}"
        if cleaned_arguments:
            display += f" {cleaned_arguments}"
        return await self.invoke(
            SkillInvocation(cleaned_name, cleaned_arguments, display),
            cancellation,
        )

    def deactivate(self, name: str) -> bool:
        """停用当前主会话中的一个 inline Skill。

        Args:
            name: 用户传入的 Skill 名，比较时忽略大小写。

        Returns:
            原来处于活动状态时返回 True，否则返回 False。
        """

        return self._runtime.deactivate(name)

    def format_list(self) -> str:
        """整理 /skill list 需要展示的名称、来源、模式和活动状态。

        Returns:
            可直接交给终端 UI 的多行文字。
        """

        skills = self._catalog.list()
        if not skills:
            return "当前没有可用 Skill"
        active = self._runtime.active_names
        lines = ["可用 Skill："]
        lines.extend(
            f"/{skill.name} [{skill.source.value}/{skill.mode.value}]"
            f"{' [已激活]' if skill.name.casefold() in active else ''}"
            f" — {skill.description}"
            for skill in skills
        )
        return "\n".join(lines)

    def format_info(self, name: str) -> str:
        """整理一个 Skill 的元信息、入口路径和专属工具名。

        Args:
            name: 要查询的 Skill 名。

        Returns:
            不包含 SOP 正文和脚本内容的多行说明。

        Raises:
            MyCodeError: Catalog 中没有该名字。
        """

        skill = self._catalog.get(name)
        if skill is None:
            raise MyCodeError(f"未知 Skill：{name}")
        allowlist = (
            "未限制"
            if skill.allowed_tools is None
            else ", ".join(sorted(skill.allowed_tools)) or "空"
        )
        tools = ", ".join(tool.name for tool in skill.tools) or "无"
        return "\n".join(
            (
                f"名称：{skill.name}",
                f"说明：{skill.description}",
                f"来源：{skill.source.value}",
                f"模式：{skill.mode.value}",
                f"历史：{skill.context.value}",
                f"模型：{skill.model or '默认'}",
                f"可见工具：{allowlist}",
                f"专属工具：{tools}",
                f"入口：{skill.entry_path}",
            )
        )

    def reload(self) -> SkillReloadReport:
        """重新扫描三层目录，并逐 Skill 应用有效变化。

        Returns:
            新增、更新、删除、缓存回退、跳过和停用项目的逐项报告。
        """

        previous = self._catalog.snapshot()
        scanned = self._loader.reload(previous)
        old_skills = dict(previous.skills)
        final_skills = dict(old_skills)
        added: list[str] = []
        updated: list[str] = []
        removed: list[str] = []
        retained: list[SkillReloadIssue] = []
        skipped: list[SkillReloadIssue] = []

        all_names = sorted(
            set(old_skills) | set(scanned.skills) | set(scanned.candidates)
        )
        for name in all_names:
            old = old_skills.get(name)
            new = scanned.skills.get(name)
            candidates = scanned.candidates.get(name, ())
            failure = next(
                (
                    item.diagnostic.message
                    for item in candidates
                    if item.diagnostic is not None
                ),
                "当前没有有效版本",
            )
            if new is None:
                if candidates and old is not None:
                    retained.append(SkillReloadIssue(name, failure))
                elif candidates:
                    skipped.append(SkillReloadIssue(name, failure))
                    final_skills.pop(name, None)
                elif old is not None:
                    final_skills.pop(name, None)
                    removed.append(old.name)
                continue
            if old is None:
                trial = {**final_skills, name: new}
                try:
                    self._validate_skills(tuple(trial.values()))
                except MyCodeError as exc:
                    skipped.append(SkillReloadIssue(name, str(exc)))
                else:
                    final_skills[name] = new
                    added.append(new.name)
                continue
            if old.revision != new.revision:
                trial = {**final_skills, name: new}
                try:
                    self._validate_skills(tuple(trial.values()))
                except MyCodeError as exc:
                    retained.append(SkillReloadIssue(name, str(exc)))
                else:
                    final_skills[name] = new
                    updated.append(new.name)

        final_snapshot = SkillCatalogSnapshot(
            skills=final_skills,
            candidates=scanned.candidates,
            diagnostics=scanned.diagnostics,
        )
        active_before = {
            item.name.casefold(): item for item in self._runtime.active_skills
        }
        self._install_snapshot(final_snapshot)
        deactivated: list[str] = []
        for normalized, active in active_before.items():
            current = final_skills.get(normalized)
            if current is None or current.mode is SkillMode.FORK:
                if self._runtime.deactivate(active.name):
                    deactivated.append(active.name)
                continue
            if current.revision != active.revision:
                self._runtime.activate_inline(current, active.arguments)
        return SkillReloadReport(
            added=tuple(added),
            updated=tuple(updated),
            removed=tuple(removed),
            retained=tuple(retained),
            skipped=tuple(skipped),
            deactivated=tuple(deactivated),
            diagnostics=scanned.diagnostics,
        )

    def format_reload(self, report: SkillReloadReport) -> str:
        """把逐项 reload 报告整理成终端可读文字。

        Args:
            report: reload 实际应用和回退的分类结果。

        Returns:
            至少包含一行总结的多行文字。
        """

        lines = ["Skill reload 完成："]
        for label, names in (
            ("新增", report.added),
            ("更新", report.updated),
            ("删除", report.removed),
            ("停用", report.deactivated),
        ):
            if names:
                lines.append(f"{label}：{', '.join(names)}")
        lines.extend(
            f"保留 {issue.name}：{issue.reason}" for issue in report.retained
        )
        lines.extend(
            f"跳过 {issue.name}：{issue.reason}" for issue in report.skipped
        )
        if len(lines) == 1:
            lines.append("没有变化")
        return "\n".join(lines)

    def _latest_definition(
        self,
        name: str,
    ) -> tuple[SkillDefinition, str | None]:
        """执行前热读一个 Skill，并在新内容损坏时使用缓存。

        Args:
            name: Catalog 中要查找和刷新的 Skill 名。

        Returns:
            本次实际使用的定义，以及可选的缓存回退警告。

        Raises:
            MyCodeError: 名字未知、入口已删除或新版本安装失败。
        """

        current = self._catalog.get(name)
        if current is None:
            names = ", ".join(skill.name for skill in self._catalog.list())
            raise MyCodeError(
                f"未知 Skill：{name}。当前可用：{names or '无'}"
            )
        refresh = self._loader.read_latest_body(current)
        if refresh.missing:
            raise MyCodeError(f"Skill {current.name} 的入口文件已经删除")
        if refresh.definition is None:
            warning = (
                refresh.diagnostic.message
                if refresh.diagnostic is not None
                else "新内容无效，继续使用上一次有效版本"
            )
            return current, warning
        latest = refresh.definition
        if latest.name.casefold() != current.name.casefold():
            return (
                current,
                "Skill 名称已经改变；本次继续使用上一次有效版本，"
                "请执行 /skill reload 重新建立命令索引",
            )
        if latest.revision == current.revision:
            return current, None
        snapshot = self._catalog.snapshot()
        next_skills = dict(snapshot.skills)
        next_skills[current.name.casefold()] = latest
        self._install_snapshot(
            SkillCatalogSnapshot(
                skills=next_skills,
                candidates=snapshot.candidates,
                diagnostics=snapshot.diagnostics,
            )
        )
        active = self._active_by_name(current.name)
        if active is not None:
            if latest.mode is SkillMode.INLINE:
                self._runtime.activate_inline(latest, active.arguments)
            else:
                self._runtime.deactivate(current.name)
        return latest, None

    def _active_by_name(self, name: str) -> ActiveSkill | None:
        """查找主 Runtime 中一个活动 Skill 的参数和顺序。

        Args:
            name: Catalog 中的 Skill 名。

        Returns:
            活动记录；当前未激活时返回 None。
        """

        normalized = name.casefold()
        return next(
            (
                item
                for item in self._runtime.active_skills
                if item.name.casefold() == normalized
            ),
            None,
        )

    def _install_snapshot(self, snapshot: SkillCatalogSnapshot) -> None:
        """校验一份完整快照，再同步替换命令、工具和 Catalog。

        Args:
            snapshot: 准备成为当前有效状态的完整 Skill 快照。

        Returns:
            None。任何校验或批量替换失败时抛出异常。
        """

        skills = tuple(snapshot.skills.values())
        self._validate_skills(skills)
        commands = build_skill_commands(sorted(skills, key=lambda item: item.name))
        registrations = self._tool_registrations(skills)
        old_commands = build_skill_commands(self._catalog.list())
        self._command_registry.replace_skill_commands(commands)
        try:
            self._tool_registry.replace_skill_tools(registrations)
        except Exception:
            self._command_registry.replace_skill_commands(old_commands)
            raise
        self._catalog.replace(snapshot)
        self._runtime.refresh_catalog_instruction()

    def _validate_skills(
        self,
        skills: tuple[SkillDefinition, ...],
    ) -> None:
        """检查命令冲突、专属工具重名和 allowedTools 最终引用。

        Args:
            skills: 准备同时安装的全部最终有效 Skill。

        Returns:
            None。全部名字可以在当前注册表中使用时正常返回。

        Raises:
            MyCodeError: Skill 命令撞静态名称、工具重名或白名单引用未知名字。
        """

        static_names = self._command_registry.static_names
        for skill in skills:
            if skill.name.casefold() in static_names:
                raise MyCodeError(
                    f"Skill {skill.name} 与内置命令名称冲突"
                )
        non_skill_tools = {
            name
            for name in self._tool_registry.registered_names
            if self._tool_registry.source_for(name) is not ToolSource.SKILL
        }
        skill_tool_names: set[str] = set()
        for skill in skills:
            for spec in skill.tools:
                if spec.name in non_skill_tools or spec.name in skill_tool_names:
                    raise MyCodeError(
                        f"Skill {skill.name} 的工具名称冲突：{spec.name}"
                    )
                skill_tool_names.add(spec.name)
        available_tools = non_skill_tools | skill_tool_names
        for skill in skills:
            if skill.allowed_tools is None:
                continue
            unknown = sorted(skill.allowed_tools - available_tools)
            if unknown:
                raise MyCodeError(
                    f"Skill {skill.name} 的 allowedTools 包含未知工具："
                    + ", ".join(unknown)
                )

    def _tool_registrations(
        self,
        skills: tuple[SkillDefinition, ...],
    ) -> tuple[SkillToolRegistration, ...]:
        """把全部目录型 Skill 的工具定义变成注册表安装记录。

        Args:
            skills: 当前最终有效的 Skill 定义。

        Returns:
            包含真实子进程工具、来源、命令和限制的注册记录元组。
        """

        return tuple(
            SkillToolRegistration(
                tool=SkillSubprocessTool(spec),
                skill_name=skill.name,
                skill_origin=skill.source.value,
                source_path=skill.entry_path,
                command=spec.command,
                timeout_seconds=spec.timeout_seconds,
                max_output_bytes=spec.max_output_bytes,
            )
            for skill in skills
            for spec in skill.tools
        )
