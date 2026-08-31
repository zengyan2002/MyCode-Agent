"""根据团队身份、审批状态和合并阶段限制模型可见及可执行的工具。"""

from __future__ import annotations

import shlex
from collections.abc import Callable
from pathlib import Path

from mycode.models.teams import (
    MemberPlanApproval,
    PlanDecision,
    TeamActorContext,
    TeamIntegrationState,
)
from mycode.models.tools import ToolErrorCode, ToolExecutionResult, ToolView
from mycode.teams.integration import TeamIntegrationError, TeamIntegrationService
from mycode.teams.store import TeamStateStore, TeamStoreError
from mycode.tools.interceptors import (
    InterceptionDecision,
    ToolRunContext,
)


TEAM_MANAGEMENT_TOOLS = frozenset(
    {"TeamCreate", "TeamGet", "TeamDelete", "TeamTakeover", "TeamMemberStop"}
)
TEAM_TASK_TOOLS = frozenset(
    {"TeamTaskCreate", "TeamTaskList", "TeamTaskGet", "TeamTaskClaim", "TeamTaskUpdate"}
)
TEAM_MEMBER_TOOLS = TEAM_TASK_TOOLS | {"SendMessage"}
ORDINARY_TEAM_TOOLS = frozenset({"TeamCreate", "TeamTakeover"})
ACTIVE_LEAD_TOOLS = frozenset(
    {
        "Agent",
        "TeamGet",
        "TeamDelete",
        "TeamMemberStop",
        "TeamTaskCreate",
        "TeamTaskList",
        "TeamTaskGet",
        "TeamTaskUpdate",
        "SendMessage",
    }
)
DIRECT_WRITE_TOOLS = frozenset(
    {"write_file", "edit_file", "apply_patch", "WriteFile", "EditFile", "ApplyPatch"}
)
READ_TOOLS = frozenset(
    {"read_file", "glob", "grep", "ReadFile", "Glob", "Grep", "execute_command", "Bash"}
)


def build_team_tool_view(
    actor: TeamActorContext | None,
    *,
    base: ToolView | None = None,
    plan_approved: bool = True,
    conflicted_files: tuple[str, ...] = (),
) -> ToolView:
    """把当前团队身份转换成这一轮模型能看到的工具限制。

    Args:
        actor: 本地运行时确认的 Lead、成员身份；普通会话传 ``None``。
        base: Skill 和 MCP 层已经提供的工具视图；未提供时从空视图开始。
        plan_approved: 成员当前任务是否已经通过计划审批。
        conflicted_files: Lead 正在处理的 Git 冲突文件。非空时临时允许编辑。

    Returns:
        保留原 Skill/MCP 状态，并追加团队白名单或禁止集合的新 ``ToolView``。
    """

    current = base or ToolView()
    if actor is None:
        denied = current.denied_tool_names | (
            (TEAM_MANAGEMENT_TOOLS | TEAM_MEMBER_TOOLS) - ORDINARY_TEAM_TOOLS
        )
        return _copy_view(current, denied=denied)
    if actor.actor_kind == "member":
        denied = (
            current.denied_tool_names
            | TEAM_MANAGEMENT_TOOLS
            | {"Agent", "TeamTaskCreate"}
        )
        if not plan_approved:
            denied |= DIRECT_WRITE_TOOLS | {"execute_command", "Bash"}
        return _copy_view(current, denied=denied)

    # Lead 在团队存续期间只负责协调、读取、验证和 Git 合并。只有 Git
    # 确认处于冲突状态时，才临时开放文件编辑，并由拦截器限制具体路径。
    allowed = ACTIVE_LEAD_TOOLS | READ_TOOLS
    if conflicted_files:
        allowed |= DIRECT_WRITE_TOOLS
    return _copy_view(current, final_allowlist=allowed)


def _copy_view(
    view: ToolView,
    *,
    denied: frozenset[str] | set[str] | None = None,
    final_allowlist: frozenset[str] | set[str] | None = None,
) -> ToolView:
    """复制工具视图并替换团队策略字段。

    Args:
        view: 当前 SkillRuntime 生成的工具视图。
        denied: 需要追加的最终禁止工具名。
        final_allowlist: Coordinator 模式允许的最终工具名。

    Returns:
        不修改输入对象的新 ``ToolView``。
    """

    return ToolView(
        active_skill_names=view.active_skill_names,
        business_allowlist=view.business_allowlist,
        active_mcp_names=view.active_mcp_names,
        final_allowlist=(
            view.final_allowlist
            if final_allowlist is None
            else frozenset(final_allowlist)
        ),
        denied_tool_names=(
            view.denied_tool_names if denied is None else frozenset(denied)
        ),
        visible_tool_names=view.visible_tool_names,
    )


class TeamActorInterceptor:
    """在每次团队工具执行前重新确认 Actor 和 generation。

    Attributes:
        store: 读取团队和成员当前 generation 的持久化入口。
        actor_getter: 返回当前本地 ToolContext 身份的函数。
    """

    def __init__(
        self,
        store: TeamStateStore,
        actor_getter: Callable[[], TeamActorContext | None],
    ) -> None:
        """保存执行前需要读取的团队 Store 和本地身份函数。

        Args:
            store: 团队身份和生命周期的磁盘 Store。
            actor_getter: 无参数调用后返回本轮可信 Actor；普通会话返回 None。

        Returns:
            不返回数据；实例随后加入 ``ToolScheduler`` 拦截链。
        """

        self.store = store
        self.actor_getter = actor_getter

    async def before_tool(self, context: ToolRunContext) -> InterceptionDecision:
        """拒绝没有团队身份或 generation 已失效的团队工具。

        Args:
            context: Scheduler 生成的真实工具名、参数和运行选项。

        Returns:
            非团队工具直接放行；团队工具身份有效时放行，否则返回 BLOCKED。
        """

        name = context.invocation.call.name
        if name not in TEAM_MANAGEMENT_TOOLS | TEAM_MEMBER_TOOLS | {"Agent"}:
            return InterceptionDecision.allow()
        arguments = context.invocation.call.arguments
        if name == "Agent" and not arguments.get("team_name"):
            return InterceptionDecision.allow()
        actor = self.actor_getter()
        if actor is None and name in ORDINARY_TEAM_TOOLS:
            return InterceptionDecision.allow()
        if actor is None:
            return InterceptionDecision.deny(
                ToolErrorCode.BLOCKED, "当前会话不是 Agent Team 的 Lead 或成员"
            )
        try:
            if name in {"TeamGet", "TeamDelete"}:
                self.store.require_cleanup_actor(actor)
            else:
                self.store.require_actor(actor)
        except TeamStoreError as exc:
            return InterceptionDecision.deny(ToolErrorCode.BLOCKED, str(exc))
        return InterceptionDecision.allow()


class CoordinatorCommandInterceptor:
    """阻止 Team Lead 用 Shell 绕过 Coordinator 的代码写入限制。

    Attributes:
        actor_getter: 返回当前本地团队身份的函数。
        integration_getter: 返回团队当前合并与冲突状态的函数。
    """

    _READ_COMMANDS = frozenset(
        {"python", "python3", "pytest", "rg", "grep", "find", "ls", "dir", "type", "cat"}
    )

    def __init__(
        self,
        actor_getter: Callable[[], TeamActorContext | None],
        integration_getter: Callable[[str], TeamIntegrationState],
        integration_service: TeamIntegrationService | None = None,
    ) -> None:
        """保存 Coordinator 判断所需的身份和集成状态读取函数。

        Args:
            actor_getter: 返回本轮可信 Actor 的函数。
            integration_getter: 按 team ID 返回当前合并状态的函数。
            integration_service: 执行 merge 前置检查的服务。普通策略测试或
                不装配团队集成时可以为空。

        Returns:
            不返回数据。
        """

        self.actor_getter = actor_getter
        self.integration_getter = integration_getter
        self.integration_service = integration_service

    async def before_tool(self, context: ToolRunContext) -> InterceptionDecision:
        """只让 Lead 执行读取、测试和受控 Git 命令。

        Args:
            context: 包含命令工具名和模型原始参数的真实调用。

        Returns:
            非 Lead 或非命令工具放行；可确认的读取、测试、Git 合并命令放行；
            可能写功能文件的命令返回 BLOCKED。
        """

        actor = self.actor_getter()
        name = context.invocation.call.name
        if actor is None or actor.actor_kind != "lead" or name not in {"execute_command", "Bash"}:
            return InterceptionDecision.allow()
        raw = context.invocation.call.arguments.get("command")
        if raw is None:
            raw = context.invocation.call.arguments.get("cmd")
        if not isinstance(raw, str) or not raw.strip():
            return InterceptionDecision.deny(ToolErrorCode.BLOCKED, "Coordinator 命令不能为空")
        if any(token in raw for token in ("\n", "\r", "|", ";", "&&", "||", ">", "<")):
            return InterceptionDecision.deny(
                ToolErrorCode.BLOCKED,
                "Coordinator 不允许命令串联、重定向或多行 Shell",
            )
        try:
            parts = shlex.split(raw, posix=False)
        except ValueError:
            return InterceptionDecision.deny(ToolErrorCode.BLOCKED, "无法确认该命令不会修改功能文件")
        executable = parts[0].strip('"').casefold() if parts else ""
        if executable not in self._READ_COMMANDS | {"git"}:
            return InterceptionDecision.deny(
                ToolErrorCode.BLOCKED,
                "Coordinator 只能运行读取、测试和受控 Git 命令，不能用 Shell 修改功能文件",
            )
        if executable == "git":
            action = parts[1].casefold() if len(parts) > 1 else ""
            allowed = {"status", "diff", "log", "show", "rev-parse", "merge", "commit", "add"}
            if action not in allowed:
                return InterceptionDecision.deny(ToolErrorCode.BLOCKED, "该 Git 操作不属于 Coordinator 合并流程")
            integration = self.integration_getter(actor.team_id)
            if action in {"add", "commit"} and not integration.conflicted_files:
                return InterceptionDecision.deny(
                    ToolErrorCode.BLOCKED,
                    "只有已登记的合并冲突流程才能执行 git add/commit",
                )
            if action == "add":
                allowed_paths = {
                    Path(path).as_posix().casefold()
                    for path in integration.conflicted_files
                }
                requested = {
                    Path(item.strip('"')).as_posix().casefold()
                    for item in parts[2:]
                    if not item.startswith("-")
                }
                if not requested or not requested.issubset(allowed_paths):
                    return InterceptionDecision.deny(
                        ToolErrorCode.BLOCKED,
                        "git add 只能提交当前 integration 中登记的冲突文件",
                    )
            if action == "merge" and "--abort" not in parts[2:]:
                branch = _merge_source(parts)
                if branch is None:
                    return InterceptionDecision.deny(
                        ToolErrorCode.BLOCKED,
                        "git merge 必须明确指定一个团队成员分支",
                    )
                if self.integration_service is None:
                    return InterceptionDecision.deny(
                        ToolErrorCode.BLOCKED,
                        "团队合并服务尚未装配，不能开始 merge",
                    )
                try:
                    await self.integration_service.begin_merge(actor, branch)
                except TeamIntegrationError as exc:
                    return InterceptionDecision.deny(ToolErrorCode.BLOCKED, str(exc))
        elif executable in {"python", "python3"}:
            if len(parts) < 3 or parts[1:3] not in (
                ["-m", "pytest"],
                ["-m", "compileall"],
            ):
                return InterceptionDecision.deny(
                    ToolErrorCode.BLOCKED,
                    "Coordinator 的 Python 命令只允许运行 pytest 或 compileall",
                )
        return InterceptionDecision.allow()


class CoordinatorCommandObserver:
    """把 Lead 已执行的 merge 和验证结果写入团队 integration 状态。

    Attributes:
        actor_getter: 返回当前本地团队身份的函数。
        integration: 读取 Git 事实并保存合并、冲突和验证记录的服务。
    """

    def __init__(
        self,
        actor_getter: Callable[[], TeamActorContext | None],
        integration: TeamIntegrationService,
    ) -> None:
        """保存命令执行结束后需要使用的身份和集成服务。

        Args:
            actor_getter: 无参数调用后返回当前可信 Actor。
            integration: 与 Coordinator 拦截器共享的团队集成服务。

        Returns:
            不返回数据；实例随后加入 Scheduler 的 observer 列表。
        """

        self.actor_getter = actor_getter
        self.integration = integration

    async def after_tool(
        self,
        context: ToolRunContext,
        result: ToolExecutionResult,
    ) -> None:
        """根据真实命令和退出结果更新合并或验证状态。

        Args:
            context: 包含已执行命令原文的工具运行上下文。
            result: Executor 返回的成功状态、退出码和输出摘要。

        Returns:
            非 Lead、非命令工具和不属于集成流程的命令不写状态。
        """

        actor = self.actor_getter()
        name = context.invocation.call.name
        if actor is None or actor.actor_kind != "lead" or name not in {"execute_command", "Bash"}:
            return
        raw = context.invocation.call.arguments.get("command")
        if raw is None:
            raw = context.invocation.call.arguments.get("cmd")
        if not isinstance(raw, str):
            return
        try:
            parts = shlex.split(raw, posix=False)
        except ValueError:
            return
        if not parts:
            return
        executable = parts[0].strip('"').casefold()
        if executable == "git" and len(parts) > 1:
            action = parts[1].casefold()
            if action == "merge" and "--abort" in parts[2:]:
                if result.success:
                    self.integration.observe_abort(actor)
                return
            if action == "merge":
                self.integration.observe_merge(actor, command_succeeded=result.success)
                return
            state = self.integration.store.load_team(actor.team_id).integration
            if action == "commit" and state.current_source_branch is not None:
                self.integration.observe_merge(actor, command_succeeded=result.success)
                return
        scope = _validation_scope(executable, parts)
        if scope is None:
            return
        raw_exit_code = result.metadata.get("exit_code")
        exit_code = raw_exit_code if isinstance(raw_exit_code, int) else (0 if result.success else 1)
        self.integration.record_validation(
            actor,
            command=raw,
            scope=scope,
            exit_code=exit_code,
        )


class CoordinatorFileInterceptor:
    """只允许 Team Lead 编辑当前 Git 合并登记的冲突文件。

    Attributes:
        workspace_root: Lead 主仓库的绝对路径。
        actor_getter: 返回当前本地团队身份的函数。
        integration_getter: 按 team ID 返回当前冲突文件列表的函数。
    """

    def __init__(
        self,
        workspace_root: Path,
        actor_getter: Callable[[], TeamActorContext | None],
        integration_getter: Callable[[str], TeamIntegrationState],
    ) -> None:
        """保存路径判定和实时冲突状态所需的组件。

        Args:
            workspace_root: 当前主仓库绝对路径。
            actor_getter: 无参数调用后返回当前可信 Actor。
            integration_getter: 按 team ID 读取最新 integration 状态。

        Returns:
            不返回数据；实例随后加入工具调度器拦截链。
        """

        self.workspace_root = workspace_root.resolve(strict=True)
        self.actor_getter = actor_getter
        self.integration_getter = integration_getter

    async def before_tool(self, context: ToolRunContext) -> InterceptionDecision:
        """检查 Coordinator 文件写入是否精确命中已登记冲突文件。

        Args:
            context: 包含真实工具名和文件路径参数的调用上下文。

        Returns:
            普通会话、成员和非写文件工具直接放行；Lead 写入真实冲突文件
            时放行，其他路径返回 BLOCKED。
        """

        actor = self.actor_getter()
        name = context.invocation.call.name
        if actor is None or actor.actor_kind != "lead" or name not in DIRECT_WRITE_TOOLS:
            return InterceptionDecision.allow()
        raw_path = context.invocation.call.arguments.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return InterceptionDecision.deny(
                ToolErrorCode.BLOCKED,
                "Coordinator 文件写入必须提供明确路径",
            )
        requested = Path(raw_path)
        if not requested.is_absolute():
            requested = self.workspace_root / requested
        requested = requested.resolve()
        allowed = {
            (
                path.resolve()
                if path.is_absolute()
                else (self.workspace_root / path).resolve()
            )
            for path in self.integration_getter(actor.team_id).conflicted_files
        }
        if requested not in allowed:
            return InterceptionDecision.deny(
                ToolErrorCode.BLOCKED,
                "Coordinator 只能编辑当前 Git 合并登记的冲突文件",
            )
        return InterceptionDecision.allow()


def plan_is_approved(
    approval: MemberPlanApproval | None,
    *,
    task_id: str,
    attempt_number: int,
) -> bool:
    """判断审批是否精确属于当前任务和当前执行次数。

    Args:
        approval: 成员最近一次持久化的计划审批；没有时传 None。
        task_id: 当前 working 任务 ID。
        attempt_number: 当前任务执行次数。

    Returns:
        只有任务、执行次数匹配且决定为 approved 时返回 True。
    """

    return bool(
        approval is not None
        and approval.task_id == task_id
        and approval.attempt_number == attempt_number
        and approval.decision is PlanDecision.APPROVED
    )


def _merge_source(parts: list[str]) -> str | None:
    """从已经拆分的 ``git merge`` 命令中找出唯一来源分支。

    Args:
        parts: 保留参数边界的命令数组，前两项应为 ``git`` 和 ``merge``。

    Returns:
        第一个非选项参数；没有明确来源时返回 ``None``。
    """

    options_with_value = {"-m", "--message", "-s", "--strategy", "-X", "--strategy-option"}
    skip_next = False
    for item in parts[2:]:
        value = item.strip('"')
        if skip_next:
            skip_next = False
            continue
        if value in options_with_value:
            skip_next = True
            continue
        if value.startswith("-"):
            continue
        return value
    return None


def _validation_scope(executable: str, parts: list[str]) -> str | None:
    """判断一条已执行命令是局部验证、最终验证还是普通读取。

    Args:
        executable: 已去除引号并转成小写的可执行程序名。
        parts: ``shlex`` 拆分后的完整命令参数。

    Returns:
        指定测试文件或测试目录时返回 ``focused``；对整个项目运行 pytest
        或 compileall 时返回 ``final``；非验证命令返回 ``None``。
    """

    arguments: list[str]
    if executable in {"python", "python3"}:
        if len(parts) < 3 or parts[1] != "-m":
            return None
        module = parts[2].strip('"').casefold()
        if module == "compileall":
            return "final"
        if module != "pytest":
            return None
        arguments = parts[3:]
    elif executable == "pytest":
        arguments = parts[1:]
    else:
        return None
    selected = [item for item in arguments if not item.startswith("-")]
    return "focused" if selected else "final"
