"""保存长期团队成员跨多轮复用的会话和单轮执行入口。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import uuid4

from mycode.agent.conversation import Conversation
from mycode.agents.catalog import AgentCatalog
from mycode.agents.runtime import IndependentAgentRuntimeBuilder
from mycode.constants import DEFAULT_MAX_MODEL_CALLS
from mycode.context import ArtifactStore, ContextManager
from mycode.models.agents import (
    AgentPermissionMode,
    BackgroundTaskStatus,
    IndependentAgentOrigin,
    IndependentAgentSpec,
)
from mycode.models.messages import ChatMessage
from mycode.models.permissions import PermissionMode
from mycode.models.prompts import PromptContext
from mycode.models.teams import TeamActorContext
from mycode.models.tools import ToolView
from mycode.models.worktrees import (
    WorkspaceAssignment,
    WorkspaceIsolationMode,
    WorktreeFinishAction,
    WorktreeFinishReport,
    WorktreeTaskOutcome,
)
from mycode.agents.workspaces import AgentWorkspaceService
from mycode.permissions import PermissionController
from mycode.persistence.sessions import PreparedSession, SessionManager
from mycode.providers.runner import ProviderRequestRunner
from mycode.models.config import ProviderConfig
from mycode.teams.mailbox import TeamMailbox
from mycode.teams.policy import build_team_tool_view, plan_is_approved
from mycode.teams.prompts import build_team_instruction
from mycode.teams.store import TeamStateStore


TeamTurnExecutor = Callable[[str, tuple[ChatMessage, ...]], Awaitable[str]]


@dataclass(slots=True)
class TeamMemberRuntime:
    """代表一个成员已经恢复、可以继续执行新指令的对话运行时。

    Attributes:
        sessions: 把每轮已确认消息落盘到该成员专属目录的会话管理器。
        execute_turn: 使用当前成员 Provider、工具和权限执行一轮的函数。
    """

    sessions: SessionManager
    execute_turn: TeamTurnExecutor

    async def run(self, prompt: str) -> str:
        """用完整历史执行一轮，并返回成员最后的纯文本答复。

        Args:
            prompt: 首次工作说明、任务扫描提示或显式唤醒消息。

        Returns:
            模型结束本轮时返回的最终文本。消息的实际持久化仍由装配出的
            AgentTurnRunner/SessionManager 在模型边界完成。

        Raises:
            RuntimeError: prompt 为空或执行器没有返回文本。
        """

        if not prompt.strip():
            raise RuntimeError("成员运行提示不能为空")
        result = await self.execute_turn(prompt, self.sessions.history)
        if not result.strip():
            raise RuntimeError("成员本轮没有返回最终文本")
        return result

    def close(self) -> None:
        """关闭成员会话文件，不删除已经保存的历史。

        Returns:
            不返回数据；后续恢复会重新打开同一磁盘会话。
        """

        self.sessions.close()


TeamRuntimeLoader = Callable[[str, str], Awaitable[TeamMemberRuntime]]


class TeamAgentWorkspaceService(AgentWorkspaceService):
    """让一次 Agent 轮次复用长期成员 Worktree，不在轮次结束时删除它。

    Attributes:
        manager: 父类保存的 WorktreeManager；TeamDelete 才负责最终清理目录。
    """

    async def prepare(self, spec: IndependentAgentSpec) -> IndependentAgentSpec:
        """确认调用方已经把成员固定 Worktree 写入 spec。

        Args:
            spec: Team runtime 根据成员记录构造的本轮运行输入。

        Returns:
            原样返回包含 workspace 的 spec。

        Raises:
            ValueError: spec 没有成员固定工作区。
        """

        if spec.workspace is None:
            raise ValueError("团队成员运行必须使用已登记的固定 Worktree")
        return spec

    async def mark_running(self, spec: IndependentAgentSpec) -> None:
        """确认成员本轮有固定目录；成员状态由 TeammateHost 单独持久化。

        Args:
            spec: 即将开始一轮模型调用的成员输入。

        Returns:
            workspace 存在时不返回数据。
        """

        if spec.workspace is None:
            raise ValueError("团队成员运行缺少固定 Worktree")

    async def finish(
        self,
        spec: IndependentAgentSpec,
        outcome: WorktreeTaskOutcome,
    ) -> WorktreeFinishReport:
        """结束单轮 Agent 资源，但保留长期成员目录、分支和租约。

        Args:
            spec: 本轮使用的成员固定工作区。
            outcome: 本轮完成、失败、取消或中断状态。

        Returns:
            action 为 retained 的报告；真正删除由 TeamDelete 执行。
        """

        if spec.workspace is None:
            raise ValueError("团队成员运行缺少固定 Worktree")
        return WorktreeFinishReport(
            workspace=spec.workspace,
            action=WorktreeFinishAction.RETAINED,
            terminal_status=outcome,
            changes=None,
            reason="团队成员 Worktree 在团队存续期间保持不变",
        )

    async def abandon(
        self,
        spec: IndependentAgentSpec,
        reason: str,
    ) -> WorktreeFinishReport | None:
        """装配失败时保留成员固定目录，并返回实际保留说明。

        Args:
            spec: 可能已经包含成员工作区的运行输入。
            reason: 装配失败原因，仅用于验证不是空字符串。

        Returns:
            没有 workspace 时返回 None；否则返回 retained 报告。
        """

        if not reason.strip():
            raise ValueError("abandon reason 不能为空")
        if spec.workspace is None:
            return None
        return await self.finish(spec, WorktreeTaskOutcome.FAILED)


class TeamMemberRuntimeFactory:
    """按成员记录恢复磁盘会话，并为每轮创建可执行的 Agent。

    Attributes:
        store: 读取成员、任务、generation 和团队会话目录的入口。
        mailbox: 读取成员最新计划审批回复的邮箱服务。
        catalog: 根据成员 ``role_name`` 取得当前生效角色定义。
        runtime_builder: 创建真实 AgentTurnRunner、工具上下文和权限组件。
        request_runner: 恢复会话的 ContextManager 使用的 Provider 请求器。
        provider_config: 恢复和新一轮上下文估算使用的 Provider 配置。
        stable_prompt: 主应用已装入项目规则的固定系统提示。
        parent_permissions: 角色声明 ``inherit`` 时要快照的当前权限模式。
    """

    def __init__(
        self,
        *,
        store: TeamStateStore,
        mailbox: TeamMailbox,
        catalog: AgentCatalog,
        runtime_builder: IndependentAgentRuntimeBuilder,
        request_runner: ProviderRequestRunner,
        provider_config: ProviderConfig,
        stable_prompt: str,
        parent_permissions: PermissionController,
    ) -> None:
        """保存恢复长期成员所需的真实应用组件。

        Args:
            store: 当前工作区唯一的 TeamStateStore。
            mailbox: 与 Store 共用团队目录的 TeamMailbox。
            catalog: 应用启动后完成加载的 AgentCatalog。
            runtime_builder: 使用 TeamAgentWorkspaceService 装配的运行时 Builder。
            request_runner: 当前 Provider 的共享请求器。
            provider_config: 当前 Provider 的上下文配置。
            stable_prompt: 已包含项目 AGENTS.md 的非空提示。
            parent_permissions: Lead 当前权限控制器。

        Returns:
            不返回数据；调用实例时才打开具体成员会话。
        """

        if not stable_prompt.strip():
            raise ValueError("成员固定提示不能为空")
        self.store = store
        self.mailbox = mailbox
        self.catalog = catalog
        self.runtime_builder = runtime_builder
        self.request_runner = request_runner
        self.provider_config = provider_config
        self.stable_prompt = stable_prompt
        self.parent_permissions = parent_permissions

    async def __call__(self, team_id: str, member_id: str) -> TeamMemberRuntime:
        """恢复指定成员的 JSONL 历史，并返回可续写运行时。

        Args:
            team_id: 成员所属的团队 ID。
            member_id: 团队花名册中的不可变 Agent ID。

        Returns:
            保持成员 SessionManager 打开、且可以串行执行新指令的
            ``TeamMemberRuntime``。

        Raises:
            RuntimeError: 成员或角色不存在，会话无法恢复，或运行结束时
                没有可用的最终文本。
        """

        snapshot = self.store.load_team(team_id)
        member = next(
            (item for item in snapshot.members if item.agent_id == member_id),
            None,
        )
        if member is None:
            raise RuntimeError(f"团队成员不存在：{member_id}")
        role = self.catalog.get(member.role_name)
        if role is None:
            raise RuntimeError(f"成员角色不存在：{member.role_name}")

        conversation = Conversation()
        artifact_store = ArtifactStore(
            member.worktree_path,
            f"team-{member.agent_id}-{uuid4().hex}",
        )
        context_manager = ContextManager(
            self.request_runner,
            conversation,
            self.provider_config,
            artifact_store,
        )
        sessions = SessionManager(
            member.worktree_path,
            conversation,
            context_manager,
            sessions_dir=self.store.team_dir(team_id) / "sessions",
        )
        candidate = sessions.read_candidate(member.session_id)
        sessions.activate(
            PreparedSession(candidate, candidate.messages, None, 0)
        )
        actor = TeamActorContext(
            team_id=team_id,
            actor_id=member.agent_id,
            actor_kind="member",
            generation=member.runtime_generation,
        )
        workspace = WorkspaceAssignment(
            root=member.worktree_path,
            isolation=WorkspaceIsolationMode.WORKTREE,
            worktree_name=member.worktree_name,
            branch=member.branch,
            base_commit="team-member-baseline",
            lease_id=f"team-runtime-{member.agent_id}",
        )

        async def execute_turn(
            prompt: str,
            history: tuple[ChatMessage, ...],
        ) -> str:
            """用成员当前磁盘历史执行一轮，再只追加新消息。

            Args:
                prompt: Host 合并的首次任务说明或未读邮箱消息。
                history: SessionManager 在本轮开始时的已持久化历史。

            Returns:
                Agent 正常或强制收尾时的最终文本。

            Raises:
                RuntimeError: Agent 执行失败、取消或没有最终文本。
            """

            current = self.store.load_team(team_id)
            latest_member = next(
                item for item in current.members if item.agent_id == member_id
            )
            approval_ok = not latest_member.plan_mode_required
            if latest_member.plan_mode_required and latest_member.current_task_id:
                task = next(
                    (
                        item
                        for item in current.tasks
                        if item.task_id == latest_member.current_task_id
                    ),
                    None,
                )
                approval = self.mailbox.latest_plan_approval(team_id, member_id)
                if task is not None and task.attempts and approval is not None:
                    approval_ok = (
                        approval.decided_by_generation
                        == current.team.lead_generation
                        and plan_is_approved(
                            approval,
                            task_id=task.task_id,
                            attempt_number=task.attempts[-1].number,
                        )
                    )
            role_view = ToolView(
                final_allowlist=role.tools,
                denied_tool_names=role.disallowed_tools,
            )
            tool_view = build_team_tool_view(
                actor,
                base=role_view,
                plan_approved=approval_ok,
            )
            permission_mode = (
                self.parent_permissions.mode
                if role.permission_mode is AgentPermissionMode.INHERIT
                else PermissionMode(role.permission_mode.value)
            )
            stable = "\n\n".join(
                (
                    self.stable_prompt,
                    role.prompt_body,
                    build_team_instruction(actor),
                )
            )
            spec = IndependentAgentSpec(
                run_id=f"team-turn-{uuid4().hex}",
                session_id=member.session_id,
                name=member.name,
                description=f"团队成员 {member.name} 的续写回合",
                origin=IndependentAgentOrigin.DEFINITION,
                task_prompt=prompt,
                initial_messages=history,
                prompt=PromptContext(stable=stable),
                inherited_runtime=(),
                initial_tool_names=None,
                role=role,
                model_override=member.model_override or role.model,
                max_model_calls=role.max_model_calls or DEFAULT_MAX_MODEL_CALLS,
                permission_mode=permission_mode,
                background=True,
                tool_view=tool_view,
                workspace=workspace,
                team_actor=actor,
            )
            runner = self.runtime_builder.build(spec)
            result = await runner.start().wait()
            new_messages = runner.history[len(history) :]
            sessions.append(new_messages)
            if result.status not in {
                BackgroundTaskStatus.COMPLETED,
                BackgroundTaskStatus.PARTIAL,
            }:
                raise RuntimeError(result.error or "团队成员运行失败")
            return result.final_text or "团队成员已完成本轮"

        return TeamMemberRuntime(sessions, execute_turn)
