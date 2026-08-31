"""为独立子 Agent 准备固定工作区、目标目录提示和运行结束收尾。"""

from __future__ import annotations

from dataclasses import replace

from mycode.agent.system_prompt import PromptAssembler
from mycode.agents.prompts import (
    definition_role_section,
    subagent_constraints_section,
)
from mycode.models.agents import IndependentAgentOrigin, IndependentAgentSpec
from mycode.models.prompts import (
    PromptContext,
    RuntimeInstruction,
    RuntimeInstructionKind,
)
from mycode.models.worktrees import (
    WorkspaceAssignment,
    WorkspaceIsolationMode,
    WorktreeFinishReport,
    WorktreeTaskOutcome,
    WorktreeTaskOwner,
)
from mycode.persistence.instructions import ProjectInstructionLoader
from mycode.worktrees.manager import WorktreeManager


_WORKSPACE_SPECIFIC_RUNTIME = {
    RuntimeInstructionKind.ENVIRONMENT_CONTEXT,
    RuntimeInstructionKind.ENVIRONMENT_UPDATE,
    RuntimeInstructionKind.MODEL_CALL_BUDGET,
    RuntimeInstructionKind.FINALIZATION,
}


class AgentWorkspaceService:
    """连接子 Agent 的隔离策略、项目指令重载和 Worktree 收尾。

    Attributes:
        manager: 创建独立目录、租用共享目录并执行变更保护收尾的 Manager。

    定义式角色只有明确声明 ``shared`` 时共享当前目录；匿名 Fork 和 fork
    Skill 始终创建独立 Worktree。该策略没有单次工具调用覆盖参数。
    """

    def __init__(self, manager: WorktreeManager) -> None:
        """创建子 Agent 工作区服务。

        Args:
            manager: 已启动的 Worktree Manager。

        Returns:
            新的工作区服务。

        Raises:
            ValueError: ``manager`` 类型无效。
        """

        if not isinstance(manager, WorktreeManager):
            raise ValueError("AgentWorkspaceService.manager 类型无效")
        self.manager = manager

    async def prepare(self, spec: IndependentAgentSpec) -> IndependentAgentSpec:
        """为一份未准备 spec 分配目录并重建目标工作区上下文。

        Args:
            spec: AgentService 或 SkillForkRunner 生成、``workspace`` 仍为
                ``None`` 的冻结运行输入。

        Returns:
            ``workspace`` 已赋值、稳定提示已从目标目录重建、运行时说明已更新
            的新 ``IndependentAgentSpec``。

        Raises:
            ValueError: spec 类型无效或已经准备过。
            WorktreeManagerError: 创建、租约或目标目录项目指令加载失败。
        """

        if not isinstance(spec, IndependentAgentSpec):
            raise ValueError("prepare spec 类型无效")
        if spec.workspace is not None:
            raise ValueError("IndependentAgentSpec 已经准备过工作区")
        owner = WorktreeTaskOwner(
            session_id=spec.session_id,
            task_id=spec.run_id,
            origin=spec.origin.value,
        )
        shared = (
            spec.origin is IndependentAgentOrigin.DEFINITION
            and spec.role is not None
            and spec.role.isolation is WorkspaceIsolationMode.SHARED
        )
        assignment = (
            await self.manager.lease_current_for_task(owner)
            if shared
            else await self.manager.create_for_task(owner)
        )
        try:
            loaded = ProjectInstructionLoader(assignment.root).load()
            stable = PromptAssembler(
                project_instructions=loaded.content,
            ).build()
            if spec.role is not None:
                sections = (
                    subagent_constraints_section(),
                    definition_role_section(spec.role),
                )
                stable += "".join(
                    f"## {section.name}\n\n{section.content.strip()}\n\n"
                    for section in sections
                )
            inherited = tuple(
                item
                for item in spec.inherited_runtime
                if item.kind not in _WORKSPACE_SPECIFIC_RUNTIME
            )
            notice = RuntimeInstruction(
                RuntimeInstructionKind.RUNTIME_NOTICE,
                self._workspace_notice(assignment),
            )
            warnings = tuple(
                RuntimeInstruction(
                    RuntimeInstructionKind.RUNTIME_NOTICE,
                    f"项目指令警告：{warning.path}：{warning.reason}",
                )
                for warning in loaded.warnings
            )
            runtime = (*inherited, notice, *warnings)
            return replace(
                spec,
                workspace=assignment,
                prompt=PromptContext(stable=stable, runtime=runtime),
                inherited_runtime=runtime,
            )
        except Exception:
            await self.manager.finish_task(
                assignment,
                WorktreeTaskOutcome.FAILED,
            )
            raise

    async def abandon(
        self,
        spec: IndependentAgentSpec,
        reason: str,
    ) -> WorktreeFinishReport | None:
        """收尾一份已经准备但未能进入正常 Runner 生命周期的 spec。

        Args:
            spec: 可能已经包含工作区分配的独立运行输入。
            reason: 排队取消或装配失败原因，只用于调用方日志；Worktree 报告
                的主要原因由 Manager 根据实际变更生成。

        Returns:
            spec 尚未准备时返回 ``None``；否则返回失败终态的工作区收尾报告。

        Raises:
            ValueError: spec 或原因类型无效。
        """

        if not isinstance(spec, IndependentAgentSpec):
            raise ValueError("abandon spec 类型无效")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("abandon reason 必须是非空字符串")
        if spec.workspace is None:
            return None
        return await self.manager.finish_task(
            spec.workspace,
            WorktreeTaskOutcome.FAILED,
        )

    async def finish(
        self,
        spec: IndependentAgentSpec,
        outcome: WorktreeTaskOutcome,
    ) -> WorktreeFinishReport:
        """在子 Agent 资源关闭后检查并收尾其工作区。

        Args:
            spec: Runner 实际使用、包含固定工作区分配的 spec。
            outcome: Agent 终态映射后的 Worktree 任务结果。

        Returns:
            Manager 生成的删除、保留或共享释放报告。

        Raises:
            ValueError: spec 尚未准备或 outcome 类型无效。
        """

        if spec.workspace is None:
            raise ValueError("finish 需要已经准备工作区的 spec")
        if not isinstance(outcome, WorktreeTaskOutcome):
            raise ValueError("finish outcome 类型无效")
        return await self.manager.finish_task(spec.workspace, outcome)

    async def mark_running(self, spec: IndependentAgentSpec) -> None:
        """在 Runner 真正开始模型循环前持久化任务运行状态。

        Args:
            spec: 已经准备工作区、即将开始运行的独立 Agent 输入。

        Returns:
            共享主仓库无需记录时直接返回；临时 Worktree 改为 running 后不
            返回数据。

        Raises:
            ValueError: spec 尚未准备工作区。
            WorktreeManagerError: 任务租约已失效或记录不存在。
        """

        if spec.workspace is None:
            raise ValueError("mark_running 需要已经准备工作区的 spec")
        await self.manager.mark_task_running(spec.workspace)

    @staticmethod
    def _workspace_notice(assignment: WorkspaceAssignment) -> str:
        """构造可信的子 Agent 工作目录说明。

        Args:
            assignment: Manager 冻结的绝对路径、分支、基线和父目录状态。

        Returns:
            不含配置正文的多行说明，要求所有工具使用目标根目录，并明确父目录
            未提交内容不会复制到新 Worktree。
        """

        lines = [
            f"本次子 Agent 的工作目录固定为：{assignment.root}",
            f"Git 分支：{assignment.branch or 'detached'}",
            f"创建基线 commit：{assignment.base_commit}",
            "所有相对路径都以该目录为根；不要把主 Agent 原目录拼到工具路径中。",
        ]
        if assignment.parent_had_changes:
            lines.append(
                "创建时父工作区有未提交内容；这些 staged、unstaged 和 untracked "
                "文件没有复制到本 Worktree。"
            )
        return "\n".join(lines)
