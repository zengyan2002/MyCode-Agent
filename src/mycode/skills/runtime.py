"""管理当前会话中已激活的 inline Skill 及其工具范围。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from mycode.agent.instructions import RuntimeInstructionManager
from mycode.skills.catalog import SkillCatalog
from mycode.models.skills import (
    ActiveSkill,
    SkillDefinition,
    SkillMode,
    SkillRestoreReport,
    SkillSessionState,
)
from mycode.skills.parser import replace_skill_arguments
from mycode.skills.resources import SkillResourceAccess
from mycode.skills.trust import SkillTrustStore
from mycode.models.tools import ToolView
from mycode.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from mycode.persistence.sessions import SessionManager


class SkillRuntime:
    """保存一个主会话或临时 fork 当前生效的 Skill 状态。

    SkillService 在激活、停用和恢复时调用本类。它同时更新运行时 SOP、
    ReadFile 资源范围、工具白名单和会话信任缓存，但不负责扫描磁盘。
    """

    def __init__(
        self,
        catalog: SkillCatalog,
        instructions: RuntimeInstructionManager,
        resources: SkillResourceAccess,
        trust: SkillTrustStore,
        session_manager: SessionManager | None = None,
        maximum_allowlist: frozenset[str] | None = None,
    ) -> None:
        """连接当前 Catalog 与本会话使用的运行组件。

        Args:
            catalog: 提供当前有效 Skill 定义的共享目录。
            instructions: 为每轮模型请求提供目录和活动 SOP 的管理器。
            resources: 当前会话的目录型 Skill 只读路径映射。
            trust: 当前会话外部 Skill 专属工具的首次信任集合。
            session_manager: 主对话使用的会话管理器；临时 fork 不持久化
                活动状态，因此传 None。
            maximum_allowlist: 顶层 fork Skill 的业务工具上限；主会话为
                None。嵌套 Skill 只能继续收窄，不能放宽这个集合。

        Returns:
            None。创建后会立即把轻量 Skill 目录写入指令管理器。
        """

        # reload 后仍查询同一个 Catalog 对象，避免保存过期快照。
        self._catalog = catalog
        # 激活与停用会同步更新这一个指令管理器。
        self._instructions = instructions
        # 目录型 Skill 激活后只在这个资源映射中获得读取范围。
        self._resources = resources
        # /clear 通过本类清掉当前会话确认记录。
        self._trust = trust
        # 主会话状态变化后写入旁路文件；临时 fork 中为 None。
        self._session_manager = session_manager
        # fork 嵌套加载时与合并结果取交集；主会话不设置上限。
        self._maximum_allowlist = maximum_allowlist
        # 键是规范化 Skill 名，值是会话持久化所需的活动状态。
        self._active: dict[str, ActiveSkill] = {}
        # 保存活动状态对应的当前定义，用来合并白名单和刷新 SOP。
        self._definitions: dict[str, SkillDefinition] = {}
        # 新 Skill 激活时使用的递增序号；停用不会回收旧序号。
        self._next_order = 1
        self.refresh_catalog_instruction()

    @property
    def active_skills(self) -> tuple[ActiveSkill, ...]:
        """按激活顺序返回当前会话全部 inline Skill。

        Returns:
            activated_order 从小到大排列的不可变活动状态元组。
        """

        return tuple(
            sorted(
                self._active.values(),
                key=lambda item: item.activated_order,
            )
        )

    @property
    def active_names(self) -> frozenset[str]:
        """返回当前激活 Skill 的规范化名字集合。

        Returns:
            ToolView 用来判断专属工具归属的只读集合。
        """

        return frozenset(self._active)

    def build_catalog_instruction(self) -> str:
        """生成只含 Skill 名和一句说明的目录文本。

        Returns:
            没有可用 Skill 时返回空字符串；否则返回可供 Agent 判断意图并
            调用 LoadSkill 的逐行列表，不包含任何 SOP 正文。
        """

        skills = self._catalog.list()
        if not skills:
            return ""
        lines = [
            "当前可用 Skill 如下。用户需求匹配时，调用 LoadSkill 按需加载："
        ]
        lines.extend(
            f"- {skill.name}: {skill.description}"
            for skill in skills
        )
        return "\n".join(lines)

    def refresh_catalog_instruction(self) -> None:
        """把 Catalog 当前名字和说明同步到指令管理器。

        Returns:
            None。reload 后调用可让下一轮请求看到最新轻量目录。
        """

        self._instructions.set_skill_catalog(
            self.build_catalog_instruction() or None
        )

    def activate_inline(
        self,
        skill: SkillDefinition,
        arguments: str,
    ) -> ActiveSkill:
        """激活或刷新一个共享主对话的 Skill。

        Args:
            skill: Catalog 中准备激活的最新 SkillDefinition。
            arguments: 替换 SOP 中 $ARGUMENTS 的原始用户参数。

        Returns:
            当前活动状态。重复激活同名 Skill 会更新正文、参数和版本，
            但不增加记录，也不改变第一次激活时的顺序。

        Raises:
            ValueError: 传入的 Skill 使用 fork 模式。
        """

        if skill.mode is not SkillMode.INLINE:
            raise ValueError(f"Skill {skill.name} 不是 inline 模式")
        normalized = skill.name.casefold()
        previous = self._active.get(normalized)
        if previous is None:
            order = self._next_order
            self._next_order += 1
        else:
            order = previous.activated_order
            # 新定义可能从目录型变为单文件，先撤销旧根再按新定义登记。
            self._resources.deactivate(normalized)
        active = ActiveSkill(
            name=skill.name,
            activated_order=order,
            revision=skill.revision,
            arguments=arguments,
        )
        self._active[normalized] = active
        self._definitions[normalized] = skill
        self._resources.activate(skill)
        self._instructions.set_active_skill(
            skill.name,
            replace_skill_arguments(skill.prompt_body, arguments),
            order,
        )
        self._persist_state()
        return active

    def activate_temporary(
        self,
        skill: SkillDefinition,
        arguments: str,
    ) -> ActiveSkill:
        """在临时 fork 生命周期中激活任意模式 Skill 的 SOP。

        Args:
            skill: 顶层 fork Skill 或在 fork 内嵌套加载的当前定义。
            arguments: 用来替换 $ARGUMENTS 的本次参数。

        Returns:
            临时活动状态。重复名字刷新正文但不改变激活顺序。

        Notes:
            本方法不改变 Skill 自身 mode，也不写主会话元数据。调用方必须
            使用没有 SessionManager 的临时 SkillRuntime，并在 finally 清空。
        """

        normalized = skill.name.casefold()
        previous = self._active.get(normalized)
        if previous is None:
            order = self._next_order
            self._next_order += 1
        else:
            order = previous.activated_order
            self._resources.deactivate(normalized)
        active = ActiveSkill(
            name=skill.name,
            activated_order=order,
            revision=skill.revision,
            arguments=arguments,
        )
        self._active[normalized] = active
        self._definitions[normalized] = skill
        self._resources.activate(skill)
        self._instructions.set_active_skill(
            skill.name,
            replace_skill_arguments(skill.prompt_body, arguments),
            order,
        )
        return active

    def deactivate(self, name: str) -> bool:
        """停用一个 inline Skill 并撤销其 SOP 和资源范围。

        Args:
            name: 需要停用的 Skill 名，比较时忽略大小写。

        Returns:
            原来存在活动状态时返回 True；未知或未激活时返回 False。
        """

        normalized = name.strip().casefold()
        active = self._active.pop(normalized, None)
        if active is None:
            return False
        self._definitions.pop(normalized, None)
        self._resources.deactivate(normalized)
        self._instructions.remove_active_skill(normalized)
        self._persist_state()
        return True

    def clear(self) -> None:
        """清空当前会话全部 Skill 状态和临时信任。

        Returns:
            None。活动 SOP、资源根、工具所属状态和首次信任立即失效；
            Catalog 目录继续保留，供新会话按需加载。
        """

        self._active.clear()
        self._definitions.clear()
        self._resources.clear()
        self._trust.clear()
        self._instructions.clear_active_skills()
        self._next_order = 1
        self._persist_state()

    def restore(self, names: Sequence[str]) -> SkillRestoreReport:
        """按会话保存顺序恢复当前仍有效的 inline Skill。

        Args:
            names: 从目标会话 ``.meta.json`` 读出的 Skill 名列表。

        Returns:
            成功恢复的名字和被跳过项目的明确警告。恢复只使用当前 Catalog
            定义，历史 SOP、白名单和专属工具不会从会话文件恢复。
        """

        # 目标会话已经由 SessionManager 切换完成。先撤销上一会话的运行
        # 状态，再按旁路文件顺序从当前 Catalog 重建。
        self._active.clear()
        self._definitions.clear()
        self._resources.clear()
        self._trust.clear()
        self._instructions.clear_active_skills()
        self._next_order = 1
        restored: list[str] = []
        warnings: list[str] = []
        for name in names:
            skill = self._catalog.get(name)
            if skill is None:
                warnings.append(f"会话中的 Skill {name} 当前不存在，已跳过")
                continue
            if skill.mode is not SkillMode.INLINE:
                warnings.append(
                    f"会话中的 Skill {name} 当前是 fork 模式，不能恢复为活动状态"
                )
                continue
            self._activate_restored(skill)
            restored.append(skill.name)
        self._persist_state()
        return SkillRestoreReport(tuple(restored), tuple(warnings))

    def _activate_restored(self, skill: SkillDefinition) -> ActiveSkill:
        """在批量恢复期间激活一个 Skill，但不立即写旁路文件。

        Args:
            skill: 当前 Catalog 中的 inline Skill 定义。

        Returns:
            参数为空字符串、顺序按恢复列表递增的 ActiveSkill。
        """

        normalized = skill.name.casefold()
        order = self._next_order
        self._next_order += 1
        active = ActiveSkill(
            name=skill.name,
            activated_order=order,
            revision=skill.revision,
            arguments="",
        )
        self._active[normalized] = active
        self._definitions[normalized] = skill
        self._resources.activate(skill)
        self._instructions.set_active_skill(
            skill.name,
            replace_skill_arguments(skill.prompt_body, ""),
            order,
        )
        return active

    def _persist_state(self) -> None:
        """把当前活动名字写入主会话旁路文件。

        Returns:
            None。临时 fork 没有 SessionManager 时直接返回。
        """

        if self._session_manager is None:
            return
        self._session_manager.save_skill_state(
            SkillSessionState(
                tuple(active.name for active in self.active_skills)
            )
        )

    def build_active_instructions(self) -> tuple[str, ...]:
        """按激活顺序生成当前活动 Skill 的完整 SOP 文本。

        Returns:
            每项包含 Skill 名和替换好参数的正文，后激活者位于更后面。
        """

        return tuple(
            f"Skill: {active.name}\n\n"
            f"{replace_skill_arguments(self._definitions[active.name.casefold()].prompt_body, active.arguments)}"
            for active in self.active_skills
        )

    def merged_allowlist(self) -> frozenset[str] | None:
        """合并全部活动 Skill 的三态业务工具白名单。

        Returns:
            没有活动 Skill 或任一活动 Skill 未限制工具时返回 None；全部
            显式配置时返回集合并集，所有空数组会得到空集合。
        """

        if not self._definitions:
            return None
        allowlists = tuple(
            definition.allowed_tools
            for definition in self._definitions.values()
        )
        if any(value is None for value in allowlists):
            merged_result: frozenset[str] | None = None
        else:
            merged: set[str] = set()
            for value in allowlists:
                assert value is not None
                merged.update(value)
            merged_result = frozenset(merged)
        if self._maximum_allowlist is None:
            return merged_result
        if merged_result is None:
            return self._maximum_allowlist
        return frozenset(merged_result & self._maximum_allowlist)

    def build_tool_view(self, registry: ToolRegistry) -> ToolView:
        """根据活动 Skill 和白名单取得本轮最终工具快照。

        Args:
            registry: 保存基础、MCP、系统和 Skill 专属工具的注册表。

        Returns:
            包含活动名字、合并白名单和 Provider 实际可见名字的 ToolView。
        """

        requested = ToolView(
            active_skill_names=self.active_names,
            business_allowlist=self.merged_allowlist(),
        )
        _, resolved = registry.definitions_for(requested)
        return resolved
