"""基于 ReAct 的 Agent 循环：对外发送过程事件，并可靠维护工具历史。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from mycode.agent.cancellation import CancellationToken
from mycode.agent.instructions import (
    RuntimeInstructionManager,
    deferred_tools_instruction,
)
from mycode.agent.runner import AgentTurnRequest, AgentTurnRunner
from mycode.agent.environment import EnvironmentCollector
from mycode.agent.system_prompt import PromptAssembler
from mycode.agents.snapshots import ParentRunRecorder
from mycode.agents.tasks import TaskManager, TaskNotificationInbox
from mycode.context.manager import (
    CompactionMode,
    CompactionOutcomeKind,
    ContextManager,
)
from mycode.constants import SESSION_GAP_REMINDER_HOURS
from mycode.errors import (
    ConcurrentTurnError,
)
from mycode.hooks.engine import HookEngine
from mycode.hooks.runtime import HookRunScope
from mycode.models.config import SecretValue
from mycode.models.hooks import HookContext, HookEvent
from mycode.models.events import (
    AgentEvent,
    AgentFinalizationProfile,
    AgentRunOptions,
    CompactionStatusEvent,
    CompactionStatusKind,
)
from mycode.models.messages import ChatMessage
from mycode.models.provider import ProviderRequest, ToolChoice
from mycode.models.prompts import PromptContext, RuntimeInstruction
from mycode.models.tools import (
    ToolActivationState,
    ToolView,
)
from mycode.models.memory import CompletedTurn
from mycode.memory.store import MemoryStore
from mycode.memory.worker import MemoryExtractionWorker
from mycode.persistence.sessions import (
    PreparedSession,
    SessionManager,
    SessionRestoreResult,
)
from mycode.persistence.instructions import ProjectInstructionLoader
from mycode.providers.runner import ProviderRequestRunner
from mycode.tools.registry import ToolRegistry
from mycode.tools.scheduler import ToolScheduler
from mycode.models.worktrees import WorkspaceAssignment
from mycode.worktrees.binding import WorkspaceBinding
from mycode.worktrees.manager import WorktreeManager

if TYPE_CHECKING:
    from mycode.skills.runtime import SkillRuntime


class AgentLoop:
    """管理主会话并通过共享 ReAct 核心执行每一轮用户请求。

    Attributes:
        _session_manager: 保存当前主会话 ID 和已经提交的消息历史。
        _registry: 保存主 Agent 可以发现的全部工具实现。
        _scheduler: 执行模型工具调用并应用权限、Hook 等拦截器。
        _hook_engine: 保存应用级 Hook 规则并创建会话自己的运行范围。
        _turn_runner: 主 Agent、定义式、Fork 和 fork Skill 共用的 ReAct
            执行核心；本实例为它传入主会话的状态对象。
        _running: 当前主会话是否已经有一轮请求正在执行。
    """

    def __init__(
        self,
        session_manager: SessionManager,
        # 工具注册中心
        registry: ToolRegistry,
        # 负责调度并执行模型发起的工具调用
        scheduler: ToolScheduler,
        hook_engine: HookEngine,
        *,
        # 负责发送模型请求，并处理模型返回的流式事件和取消信号
        request_runner: ProviderRequestRunner,
        # 管理环境变化通知、一次性通知和 Plan 模式提醒
        instruction_manager: RuntimeInstructionManager,
        # 检查上下文长度，并负责压缩工具结果和较早对话
        context_manager: ContextManager,
        # 每次请求都会发送给模型的固定系统提示词
        stable_prompt: str,
        # 负责读取和保存用户记忆、项目记忆及其索引
        memory_store: MemoryStore,
        # 在完整对话结束后，后台提取并更新跨会话记忆
        memory_worker: MemoryExtractionWorker,
        # 主会话 Skill 状态；未启用 Skill 系统时保持 None。
        skill_runtime: SkillRuntime | None = None,
        tool_activation: ToolActivationState | None = None,
        parent_recorder: ParentRunRecorder | None = None,
        task_manager: TaskManager | None = None,
        notification_inbox: TaskNotificationInbox | None = None,
        worktree_manager: WorktreeManager | None = None,
        team_tool_view_resolver: Callable[[ToolView], ToolView] | None = None,
        team_message_drain: Callable[[], tuple[ChatMessage, ...]] | None = None,
        team_session_refresh: Callable[[], Awaitable[object]] | None = None,
        # 错误对外展示前需要脱敏的密钥集合
        secrets: Iterable[SecretValue] = (),
    ) -> None:
        """保存主会话组件并创建当前会话的 Hook 和 ReAct 运行状态。

        Args:
            session_manager: 保存、恢复和追加主会话消息的管理器。
            registry: 应用已注册的工具实现集合。
            scheduler: 实际执行主模型工具调用的调度器。
            hook_engine: 应用共用的 Hook 规则引擎。
            request_runner: 向当前 Provider 发送请求并产生流式事件的对象。
            instruction_manager: 生成环境、计划、Skill 和一次性运行时指令。
            context_manager: 估算上下文并压缩较早对话和大型工具结果。
            stable_prompt: 每次主模型请求都携带的固定系统提示。
            memory_store: 读取用户记忆和项目记忆索引的存储对象。
            memory_worker: 完整回合结束后提取跨会话记忆的后台工作器。
            skill_runtime: 主会话当前激活的 Skill；未启用时为 ``None``。
            tool_activation: 主会话自己激活的延迟 MCP 工具集合。
            parent_recorder: 记录本轮真实 Provider 请求，供 Agent Fork 使用。
            task_manager: 管理当前进程后台子 Agent；未启用时为 ``None``。
            notification_inbox: 保存尚未进入主模型请求的任务完成通知。
            worktree_manager: 主会话进入、退出和恢复 Worktree 时使用的 Manager；
                未启用隔离功能的兼容调用方可以不传。
            team_tool_view_resolver: 每次 Provider 请求前根据当前
                Team Actor 和合并状态计算工具视图的函数；未启用团队
                功能时可为 ``None``。
            team_message_drain: 每次 Provider 请求前读取当前 Lead 团队邮箱并
                返回可注入消息的函数；普通会话传 ``None``。
            team_session_refresh: 主会话新建或恢复后重新读取 TeamBinding 的
                异步函数；未启用团队功能时传 ``None``。
            secrets: 错误、artifact 和通知对外展示前要遮盖的密钥。

        Returns:
            不返回数据；构造后的实例可以启动 Hook、执行请求和切换会话。

        Raises:
            ValueError: ``stable_prompt`` 为空或只含空白字符。
        """

        self._session_manager = session_manager
        self._registry = registry
        self._scheduler = scheduler
        # 规则定义由应用共用；该 scope 只保存当前主会话自己的运行状态。
        self._hook_engine = hook_engine
        if not stable_prompt.strip():
            raise ValueError("Agent 固定提示词不能为空")
        self._stable_prompt = stable_prompt
        self._memory_store = memory_store
        self._memory_worker = memory_worker
        # 新建、清空和恢复会话时同步活动 SOP、资源和旁路元数据。
        self._skill_runtime = skill_runtime
        # CLI 传入主 ToolContext 使用的同一实例，使 tool_search 的结果在下一轮可见。
        self._tool_activation = tool_activation or scheduler.tool_activation
        # AgentTool 在同一工具批次中读取这份实际父请求快照来创建 Fork。
        self._parent_recorder = parent_recorder or ParentRunRecorder()
        # 会话切换时取消该会话的后台任务；通知只会在 Provider 请求边界排空。
        self._task_manager = task_manager
        self._notification_inbox = notification_inbox
        self._worktree_manager = worktree_manager
        self._team_tool_view_resolver = team_tool_view_resolver or (lambda view: view)
        self._team_message_drain = team_message_drain or (lambda: ())
        self._team_session_refresh = team_session_refresh
        self._instruction_manager = instruction_manager
        self._hook_scope: HookRunScope = hook_engine.create_scope(
            instruction_manager
        )
        # 普通请求和内部摘要共享同一套流关闭及取消逻辑。
        self._request_runner = request_runner
        self._context_manager = context_manager
        self._secrets = tuple(secrets)
        # 主会话和 fork 共用的 ReAct 核心；此处不引入测试专用工厂。
        self._turn_runner = AgentTurnRunner(
            request_runner,
            registry,
            scheduler,
            hook_engine,
            secrets=self._secrets,
        )
        # 同一个 Conversation 同一时间只能处理一条用户请求。
        #当前对话是否已经有一个 Agent 回合正在执行。
        self._running = False

    #清理历史对话
    async def clear(self) -> None:
        """结束当前会话并创建新的空会话。

        Returns:
            None。新会话自己的 Hook scope 已创建并触发 `session_start`。
        """

        await self.new_session()

    async def new_session(self) -> str:
        """结束当前会话，创建空会话并重置会话级运行状态。

        Returns:
            SessionManager 新建的会话 ID；返回前已触发新会话的
            `session_start`。
        """

        if self._running:
            raise ConcurrentTurnError("当前请求运行时不能清空对话")
        previous_session_id = self._session_manager.current_id
        if self._task_manager is not None:
            await self._task_manager.cancel_session(previous_session_id)
        if self._notification_inbox is not None:
            self._notification_inbox.clear_session(previous_session_id)
        await self._hook_engine.dispatch(
            HookContext(HookEvent.SESSION_END),
            self._hook_scope,
        )
        await self._hook_engine.close_scope(self._hook_scope)
        session_id = self._session_manager.create_new()
        if self._worktree_manager is not None:
            resolution = await self._worktree_manager.resolve_session_binding(
                session_id
            )
            stable, _ = await self._load_workspace_prompt(
                resolution.assignment
            )
            await self._worktree_manager.activate_session(
                session_id,
                resolution.assignment,
            )
            self._stable_prompt = stable
        self._instruction_manager.reset()
        self._hook_scope = self._hook_engine.create_scope(
            self._instruction_manager
        )
        if self._skill_runtime is not None:
            self._skill_runtime.clear()
        self._tool_activation.reset()
        await self._hook_engine.dispatch(
            HookContext(HookEvent.SESSION_START),
            self._hook_scope,
        )
        if self._team_session_refresh is not None:
            await self._team_session_refresh()
        return session_id

    async def restore_session(
        self,
        session_id: str,
        cancellation: CancellationToken,
    ) -> SessionRestoreResult:
        """检查并恢复指定的旧会话
        函数读取并修整旧会话消息，检查其是否超过模型上下文限制，必要时压缩较早内容。检查通过后切换会话，重置运行时指令和 MCP 工具启用状态；如果会话超过 24 小时未活动，还会为下一次模型请求添加状态变化提醒

        Args:
            session_id: 需要恢复的旧会话 ID
            cancellation: 用于取消上下文检查和压缩的信号

        Returns:
            本次恢复过程中跳过、修整和压缩消息的情况
        """

        if self._running:
            raise ConcurrentTurnError("同一会话已有请求正在运行")
        self._running = True
        try:
            # 通过会话id取出要恢复的旧会话信息
            candidate = self._session_manager.read_candidate(session_id)
            workspace_assignment: WorkspaceAssignment | None = None
            workspace_stable = self._stable_prompt
            worktree_warnings: tuple[str, ...] = ()
            if self._worktree_manager is not None:
                resolution = await self._worktree_manager.resolve_session_binding(
                    session_id
                )
                workspace_assignment = resolution.assignment
                workspace_stable, instruction_warnings = (
                    await self._load_workspace_prompt(workspace_assignment)
                )
                worktree_warnings = (
                    *resolution.warnings,
                    *instruction_warnings,
                )
            skill_state, skill_metadata_warning = (
                self._session_manager.read_skill_state(session_id)
            )
            memory_runtime = self._memory_store.load_runtime_indexes()
            runtime = (
                RuntimeInstructionManager(
                    EnvironmentCollector(
                        WorkspaceBinding.fixed(workspace_assignment)
                    )
                ).preview(plan_only=False)
                if workspace_assignment is not None
                else self._instruction_manager.preview(plan_only=False)
            )
            deferred = deferred_tools_instruction(
                self._registry.deferred_mcp_names_for(
                    self._tool_activation.active_mcp_names
                )
            )
            if deferred is not None:
                runtime = (*runtime, deferred)

            restored = await self._context_manager.prepare_restored_context(
                candidate.messages,
                cancellation,
                lambda messages: self._build_request(
                    messages,
                    runtime,
                    memory_runtime,
                    include_checkpoint=False,
                    stable_prompt=workspace_stable,
                ),
            )
            previous_session_id = self._session_manager.current_id
            if self._task_manager is not None:
                await self._task_manager.cancel_session(previous_session_id)
            if self._notification_inbox is not None:
                self._notification_inbox.clear_session(previous_session_id)
            await self._hook_engine.dispatch(
                HookContext(HookEvent.SESSION_END),
                self._hook_scope,
            )
            await self._hook_engine.close_scope(self._hook_scope)
            self._session_manager.activate(
                PreparedSession(
                    candidate,
                    restored.messages,
                    restored.checkpoint,
                    restored.compactions,
                )
            )
            if (
                self._worktree_manager is not None
                and workspace_assignment is not None
            ):
                await self._worktree_manager.activate_session(
                    session_id,
                    workspace_assignment,
                )
                self._stable_prompt = workspace_stable
            self._instruction_manager.reset()
            self._hook_scope = self._hook_engine.create_scope(
                self._instruction_manager
            )
            skill_warnings: list[str] = []
            if skill_metadata_warning is not None:
                skill_warnings.append(skill_metadata_warning)
            if self._skill_runtime is not None:
                restore_report = self._skill_runtime.restore(
                    skill_state.active_skills
                )
                skill_warnings.extend(restore_report.warnings)
            self._tool_activation.reset()
            gap = datetime.now().astimezone() - candidate.last_active
            notice_added = gap > timedelta(hours=SESSION_GAP_REMINDER_HOURS)
            if notice_added:
                self.enqueue_runtime_notice(
                    "距离这段会话上次活动已经超过 24 小时。项目文件和外部状态"
                    "可能已经变化；做决定前请重新读取相关文件并确认当前状态。"
                )
            await self._hook_engine.dispatch(
                HookContext(HookEvent.SESSION_START),
                self._hook_scope,
            )
            if self._team_session_refresh is not None:
                await self._team_session_refresh()
            if restored.compactions:
                await self._hook_engine.dispatch(
                    HookContext(
                        HookEvent.COMPACT,
                        message=(
                            f"恢复会话时压缩了 {restored.compactions} 次"
                        ),
                    ),
                    self._hook_scope,
                )
            return SessionRestoreResult(
                session_id=candidate.session_id,
                skipped_lines=candidate.skipped_lines,
                chain_truncated=candidate.chain_truncated,
                compactions=restored.compactions,
                time_gap_notice_added=notice_added,
                skill_warnings=tuple(skill_warnings),
                worktree_warnings=worktree_warnings,
            )
        finally:
            self._running = False

    def close(self) -> None:
        """清理当前会话由上下文管理器创建的 artifact。

        Returns:
            不返回数据。

        Raises:
            ConcurrentTurnError: 当前仍有主 Agent 请求正在运行。
        """

        if self._running:
            raise ConcurrentTurnError("当前请求运行时不能关闭 Agent")
        self._context_manager.close()

    async def start_hooks(self) -> None:
        """在应用资源就绪后触发启动和初始会话事件。

        Returns:
            None。同步提示动作会保留在当前主会话的下一次模型请求中。
        """

        await self._hook_engine.dispatch(
            HookContext(HookEvent.STARTUP),
            self._hook_scope,
        )
        await self._hook_engine.dispatch(
            HookContext(HookEvent.SESSION_START),
            self._hook_scope,
        )

    async def shutdown_hooks(self) -> None:
        """触发当前会话和应用关闭事件，并收尾全部 Hook 资源。

        Returns:
            None。方法按 session_end、shutdown、scope、引擎的顺序关闭。
        """

        await self._hook_engine.dispatch(
            HookContext(HookEvent.SESSION_END),
            self._hook_scope,
        )
        await self._hook_engine.dispatch(
            HookContext(HookEvent.SHUTDOWN),
            self._hook_scope,
        )
        await self._hook_engine.close_scope(self._hook_scope)
        await self._hook_engine.close()

    def enqueue_runtime_notice(self, content: str) -> None:
        """让应用向下一次模型请求追加一条只发送一次的运行时通知。

        Args:
            content: 要注入运行时提示区、且不写成用户消息的通知正文。

        Returns:
            不返回数据；通知会在下一次 ``prepare`` 后自动消费。
        """
        self._instruction_manager.enqueue_notice(content)

    async def activate_workspace(
        self,
        assignment: WorkspaceAssignment,
    ) -> tuple[str, ...]:
        """把当前主会话切换到已经验证过的工作区并重载项目提示。

        Args:
            assignment: WorktreeManager 从 ``bind_session`` 返回的固定目录分配。

        Returns:
            目标目录项目指令中可忽略的加载警告。返回前工具绑定、稳定提示和
            环境采集均已指向目标目录。

        Raises:
            ConcurrentTurnError: 主 Agent 仍有请求正在运行。
            RuntimeError: 当前 AgentLoop 没有装配 WorktreeManager。
            WorktreeManagerError: 会话映射与待激活分配不一致。
        """

        if self._running:
            raise ConcurrentTurnError("当前请求运行时不能切换 Worktree")
        if self._worktree_manager is None:
            raise RuntimeError("Worktree 功能尚未装配")
        stable, warnings = await self._load_workspace_prompt(assignment)
        await self._worktree_manager.activate_session(
            self._session_manager.current_id,
            assignment,
        )
        self._stable_prompt = stable
        self._instruction_manager.reset()
        return warnings

    async def exit_workspace(self) -> tuple[WorkspaceAssignment, tuple[str, ...]]:
        """让当前主会话退出 Worktree，回到主仓库并重载项目提示。

        Returns:
            主仓库工作区分配和项目指令加载警告。

        Raises:
            ConcurrentTurnError: 主 Agent 仍有请求正在运行。
            RuntimeError: 当前 AgentLoop 没有装配 WorktreeManager。
            WorktreeManagerError: 状态不可信或主仓库 HEAD 无法读取。
        """

        if self._running:
            raise ConcurrentTurnError("当前请求运行时不能退出 Worktree")
        if self._worktree_manager is None:
            raise RuntimeError("Worktree 功能尚未装配")
        assignment = await self._worktree_manager.exit_session(
            self._session_manager.current_id
        )
        stable, warnings = await self._load_workspace_prompt(assignment)
        self._stable_prompt = stable
        self._instruction_manager.reset()
        return assignment, warnings

    @staticmethod
    async def _load_workspace_prompt(
        assignment: WorkspaceAssignment,
    ) -> tuple[str, tuple[str, ...]]:
        """读取一个工作目录的项目指令并构造主 Agent 固定提示。

        Args:
            assignment: 已经验证、包含绝对根目录的工作区分配。

        Returns:
            ``(稳定提示, 警告)``。警告只含文件路径和读取失败原因，不含
            配置或指令文件正文。
        """

        loaded = await asyncio.to_thread(
            ProjectInstructionLoader(assignment.root).load
        )
        stable = PromptAssembler(
            project_instructions=loaded.content
        ).build()
        warnings = tuple(
            f"{warning.path}：{warning.reason}"
            for warning in loaded.warnings
        )
        return stable, warnings

    def _build_request(
        self,
        messages: tuple[ChatMessage, ...],
        runtime: tuple[RuntimeInstruction, ...],
        memory_runtime: tuple[RuntimeInstruction, ...] = (),
        *,
        include_checkpoint: bool = True,
        stable_prompt: str | None = None,
    ) -> ProviderRequest:
        """用当前工具视图、检查点和给定消息构造普通模型请求。

        Args:
            messages: 本次请求要发送的主会话消息快照。
            runtime: 环境、计划、Skill 和一次性通知组成的运行时指令。
            memory_runtime: 用户与项目长期记忆索引指令。
            include_checkpoint: 是否把上下文压缩检查点放入提示。
            stable_prompt: 恢复预览时使用的目标工作区提示；未传时使用当前
                已激活工作区的稳定提示。

        Returns:
            已装入稳定提示、运行时提示、消息和当前可见工具的请求对象。
        """

        checkpoint = (
            self._context_manager.checkpoint_instructions
            if include_checkpoint
            else ()
        )
        return ProviderRequest(
            messages=messages,
            tools=self._registry.definitions_for(
                ToolView(
                    active_mcp_names=frozenset(
                        self._tool_activation.active_mcp_names
                    )
                )
            )[0],
            tool_choice=ToolChoice.AUTO,
            prompt=PromptContext(
                stable=stable_prompt or self._stable_prompt,
                runtime=(*checkpoint, *memory_runtime, *runtime),
            ),
        )

    def estimate_input_tokens(self, *, plan_only: bool) -> int:
        """估算当前会话下一次普通 Agent 请求的输入 Token 数量。

        Args:
            plan_only: 下一次请求是否使用计划模式运行时指令。

        Returns:
            包含历史、记忆、运行时指令和工具定义的近似 Token 数量。
        """

        memory_runtime = self._memory_store.load_runtime_indexes()
        runtime = self._instruction_manager.preview(plan_only=plan_only)
        deferred = deferred_tools_instruction(
            self._registry.deferred_mcp_names_for(
                self._tool_activation.active_mcp_names
            )
        )
        if deferred is not None:
            runtime = (*runtime, deferred)
        request = self._build_request(
            self._session_manager.history,
            runtime,
            memory_runtime,
        )
        return self._context_manager.estimate_request(request)

    async def stream_compact(
        self,
        *,
        retention_focus: str | None = None,
        cancellation: CancellationToken | None = None,
    ) -> AsyncIterator[CompactionStatusEvent]:
        """响应本地 ``/compact``，不产生用户消息也不调用工具。

        Args:
            retention_focus: 用户希望摘要额外保留的重点；不需要时为 None。
            cancellation: 外部用于中止摘要请求的取消信号。

        Yields:
            摘要开始和结束时供界面展示的状态事件。

        Returns:
            迭代结束时不额外返回数据；结果通过 ``Yields`` 逐项产生。
        """

        if self._running:
            raise ConcurrentTurnError("同一会话已有请求正在运行")
        self._running = True
        try:
            yield CompactionStatusEvent(
                CompactionStatusKind.STARTED,
                "正在压缩较早对话…",
            )
            outcome = await self._context_manager.compact(
                CompactionMode.MANUAL,
                cancellation or CancellationToken(),
                retention_focus=retention_focus,
            )
            if outcome.kind is CompactionOutcomeKind.SUCCEEDED:
                kind = CompactionStatusKind.SUCCEEDED
                await self._hook_engine.dispatch(
                    HookContext(HookEvent.COMPACT, message=outcome.message),
                    self._hook_scope,
                )
            elif outcome.kind is CompactionOutcomeKind.NO_CONTENT:
                kind = CompactionStatusKind.NO_CONTENT
            elif outcome.kind is CompactionOutcomeKind.CIRCUIT_OPEN:
                kind = CompactionStatusKind.CIRCUIT_OPEN
            else:
                kind = CompactionStatusKind.FAILED
            yield CompactionStatusEvent(kind, outcome.message)
        finally:
            self._running = False

    async def stream_turn(
        self,
        user_text: str,
        *,
        options: AgentRunOptions | None = None,
        cancellation: CancellationToken | None = None,
        emit_user_event: bool = True,
    ) -> AsyncIterator[AgentEvent]:
        """通过共享 AgentTurnRunner 处理一条主会话用户消息。

        Args:
            user_text: 用户发送的非空文本。
            options: 可选的 Plan、模型调用上限、并发和整条任务超时配置。
            cancellation: UI 提供的取消令牌；未传时创建本轮独立令牌。
            emit_user_event: False 时消息仍写入会话，但不向 UI 发出用户消息；
                后台通知唤醒回合使用该值。

        Yields:
            与改造前一致的用户、模型、工具、压缩、警告和结束事件。

        Returns:
            迭代结束时不额外返回数据；最终回答或错误通过事件产生。

        Raises:
            ConcurrentTurnError: 当前 AgentLoop 已有请求在运行。
            ValueError: 用户输入为空白。
        """

        if not user_text.strip():
            raise ValueError("用户消息不能为空")
        if self._running:
            raise ConcurrentTurnError("同一会话已有请求正在运行")
        run_options = options or AgentRunOptions()
        external = cancellation or CancellationToken()

        def completed_turn(messages: tuple[ChatMessage, ...]) -> None:
            """把主会话已经落盘的完整回合交给记忆后台。

            Args:
                messages: 从本轮用户消息到最终助手回答的已保存消息。

            Returns:
                None。后台已关闭时由共享 Runner 忽略 RuntimeError。
            """

            self._memory_worker.enqueue(
                CompletedTurn(self._session_manager.current_id, messages)
            )

        self._parent_recorder.clear()
        request = AgentTurnRequest(
            user_text=user_text,
            history=lambda: self._session_manager.history,
            append_messages=lambda messages: self._session_manager.append(
                messages
            ),
            context_manager=self._context_manager,
            instruction_manager=self._instruction_manager,
            stable_prompt=self._stable_prompt,
            finalization_profile=AgentFinalizationProfile.MAIN,
            options=run_options,
            cancellation=external,
            hook_scope=self._hook_scope,
            load_memory_runtime=self._memory_store.load_runtime_indexes,
            skill_runtime=self._skill_runtime,
            completed_turn=completed_turn,
            tool_activation=self._tool_activation,
            parent_recorder=self._parent_recorder,
            resolve_tool_view=self._team_tool_view_resolver,
            drain_external_messages=lambda: (
                *(
                    ()
                    if self._notification_inbox is None
                    else self._notification_inbox.drain_messages(
                        self._session_manager.current_id
                    )
                ),
                *self._team_message_drain(),
            ),
            emit_user_event=emit_user_event,
        )
        self._running = True
        stream = self._turn_runner.stream(request)
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()
            self._parent_recorder.clear()
            self._running = False

    @property
    def parent_recorder(self) -> ParentRunRecorder:
        """返回主 Agent 本轮用于 Fork 的父请求记录器。

        Returns:
            AgentTurnRunner 写入、AgentService 在工具调用期间读取的同一对象。
        """

        return self._parent_recorder
