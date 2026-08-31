"""装配并执行拥有独立对话、权限、缓存和 Hook 状态的子 Agent。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from mycode.agent.cancellation import CancellationToken
from mycode.agent.conversation import Conversation
from mycode.agent.environment import EnvironmentCollector
from mycode.agent.instructions import RuntimeInstructionManager
from mycode.agent.runner import AgentTurnRequest, AgentTurnRunner
from mycode.context.artifacts import ArtifactStore
from mycode.context.manager import ContextManager
from mycode.hooks.adapters import PostToolHookObserver, PreToolHookInterceptor
from mycode.hooks.engine import HookEngine
from mycode.hooks.runtime import HookRunScope
from mycode.errors import MyCodeError
from mycode.models.agents import (
    AgentRunResult,
    AgentUsage,
    BackgroundTaskStatus,
    IndependentAgentOrigin,
    IndependentAgentSpec,
)
from mycode.models.config import ProviderConfig, SecretValue
from mycode.models.events import (
    AgentCompletionMode,
    AgentErrorCode,
    AgentErrorEvent,
    AgentRunOptions,
    AgentFinalizationProfile,
    FinalReplyEvent,
    ModelTextDeltaEvent,
    ModelUsageEvent,
    ToolResultEvent,
)
from mycode.models.permissions import (
    ApprovalChoice,
    LoadedPermissionSettings,
    PermissionRequest,
    PermissionStore,
)
from mycode.permissions.interceptor import (
    AgentPermissionApprover,
    PermissionInterceptor,
)
from mycode.permissions.policy import PermissionController, PermissionPolicy
from mycode.permissions.rules import PermissionRuleResolver
from mycode.providers.runner import ProviderRequestRunner
from mycode.skills.catalog import SkillCatalog
from mycode.skills.resources import SkillResourceAccess
from mycode.skills.runtime import SkillRuntime
from mycode.skills.trust import SkillTrustInterceptor, SkillTrustStore
from mycode.tools.base import ToolContext
from mycode.tools.executor import ToolExecutor
from mycode.tools.interceptors import PlanOnlyInterceptor
from mycode.tools.registry import ToolRegistry
from mycode.tools.scheduler import ToolScheduler
from mycode.worktrees.binding import WorkspaceBinding, shared_workspace_binding
from mycode.agents.workspaces import AgentWorkspaceService
from mycode.models.worktrees import (
    WorktreeFinishAction,
    WorktreeFinishReport,
    WorktreeTaskOutcome,
)


class AgentInteractionApprover(Protocol):
    """描述前台子 Agent 可能调用的两种真实用户确认入口。"""

    async def request_permission(
        self,
        request: PermissionRequest,
    ) -> ApprovalChoice:
        """显示一项工具权限请求并返回用户选择。

        Args:
            request: 工具名、风险说明和建议规则组成的权限请求。

        Returns:
            用户在前台审批界面选择的允许、拒绝或持久化选项。
        """

        ...

    async def confirm(self, message: str) -> bool:
        """显示外部 Skill 工具的信任说明并返回是否确认。

        Args:
            message: 包含 Skill 来源及命令/读写风险的说明。

        Returns:
            用户确认信任时返回 ``True``，否则返回 ``False``。
        """

        ...


class _AgentSkillApprover:
    """前台转发 Skill 信任确认，后台固定拒绝。

    Attributes:
        background: 当前运行是否已经进入后台。进入后台后，新的 Skill
            信任请求不会再显示终端确认界面。
        _background_event: 前台确认等待期间用于通知“已经移交后台”的事件。
    """

    def __init__(
        self,
        approver: AgentInteractionApprover,
        *,
        background: bool,
    ) -> None:
        """保存确认界面并设置初始前后台状态。

        Args:
            approver: 实际显示 Skill 信任说明并接收用户选择的终端界面。
            background: ``True`` 表示当前运行从启动时就不能询问用户。

        Returns:
            不返回数据；实例随后由 SkillTrustInterceptor 调用。
        """

        self._approver = approver
        self.background = background
        self._background_event = asyncio.Event()
        if background:
            self._background_event.set()

    def move_to_background(self) -> None:
        """让后续外部 Skill 信任请求不再访问 UI。

        Returns:
            不返回数据；正在等待的确认也会按拒绝结束。
        """

        self.background = True
        self._background_event.set()

    async def confirm(self, message: str) -> bool:
        """根据当前前后台状态处理一次 Skill 信任请求。

        Args:
            message: 包含 Skill 来源、命令和读写类别的确认说明。

        Returns:
            后台返回 ``False``；前台返回真实界面的确认结果。
        """

        if self.background:
            return False
        confirmation = asyncio.create_task(self._approver.confirm(message))
        moved = asyncio.create_task(self._background_event.wait())
        try:
            done, _ = await asyncio.wait(
                {confirmation, moved},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if moved in done:
                confirmation.cancel()
                await asyncio.gather(confirmation, return_exceptions=True)
                return False
            return await confirmation
        finally:
            moved.cancel()
            await asyncio.gather(moved, return_exceptions=True)


class _ChildSkillLoadRouter:
    """让独立子 Agent 的 LoadSkill 只激活当前运行中的 SOP。

    Attributes:
        _catalog: 当前应用已加载的 Skill 定义目录。
        _runtime: 当前子 Agent 独享的 Skill 活动状态。
    """

    def __init__(self, catalog: SkillCatalog, runtime: SkillRuntime) -> None:
        """保存 Skill 目录和当前子 Agent 的 Skill 运行状态。

        Args:
            catalog: 按名字查找 Skill 定义的目录。
            runtime: 接收临时 SOP 的当前子 Agent SkillRuntime。

        Returns:
            不返回数据。
        """

        self._catalog = catalog
        self._runtime = runtime

    async def load(self, scope: str, name: str, arguments: str) -> str:
        """在当前子 Agent 中激活目标 Skill，不创建下一层 Agent。

        Args:
            scope: ToolContext 传入的范围名称；独立运行固定使用 ``fork``。
            name: Skill Catalog 中的名字。
            arguments: 替换 SOP 中 ``$ARGUMENTS`` 的原始参数。

        Returns:
            不包含 SOP 正文的激活确认。

        Raises:
            MyCodeError: scope 无效或 Skill 名不存在。
        """

        if scope != "fork":
            raise MyCodeError("独立 Agent 的 Skill 路由只接受 fork 范围")
        skill = self._catalog.get(name)
        if skill is None:
            raise MyCodeError(f"未知 Skill：{name}")
        # activate_temporary 内部会注入替换参数后的 SOP；即使原定义为 fork，
        # 此处也只激活流程文本，不调用 SkillForkRunner。
        self._runtime.activate_temporary(skill, arguments)
        return f"Skill {skill.name} 已在当前独立任务中激活"


class AgentRunHandle:
    """控制一个已经启动、可以前台等待或移交后台的子 Agent。

    Attributes:
        run_id: 与 IndependentAgentSpec 相同的运行 ID。
        cancellation: cancel 方法触发的协作式取消令牌。
        permission_approver: 前后台切换时同步更新的权限审批器。
        task: 实际执行子 Agent 并最终返回 AgentRunResult 的 asyncio Task。
    """

    def __init__(
        self,
        run_id: str,
        cancellation: CancellationToken,
        permission_approver: AgentPermissionApprover,
        skill_approver: _AgentSkillApprover,
        task: asyncio.Task[AgentRunResult],
    ) -> None:
        """保存一次已经启动的运行任务及其取消、审批控制器。

        Args:
            run_id: 与本次 ``IndependentAgentSpec`` 相同的运行 ID。
            cancellation: Provider 和工具执行共同观察的协作式取消令牌。
            permission_approver: 工具权限未命中静态规则时使用的审批器。
            skill_approver: 外部 Skill 请求信任确认时使用的审批器。
            task: 正在执行子 Agent 并最终返回结果的 asyncio Task。

        Returns:
            不返回数据；调用方通过新实例等待、移交或取消运行。
        """

        self.run_id = run_id
        self.cancellation = cancellation
        self.permission_approver = permission_approver
        self._skill_approver = skill_approver
        self.task = task

    async def wait(self) -> AgentRunResult:
        """等待当前运行结束并返回最终结果。

        Returns:
            完成、失败或取消状态，以及最终/部分文本和用量。
        """

        return await self.task

    def move_to_background(self) -> None:
        """把同一个运行实例切成非交互模式，不重启任务。

        Returns:
            不返回数据；之后权限和 Skill 信任的 ASK 都直接拒绝。
        """

        self.permission_approver.move_to_background()
        self._skill_approver.move_to_background()

    def cancel(self) -> None:
        """请求当前子 Agent 协作式停止 Provider 或工具工作。

        Returns:
            不返回数据；多次调用效果相同。
        """

        self.cancellation.cancel()


class IndependentAgentRunner:
    """持有一次独立运行的真实组件，并把过程事件汇总成结果。

    Attributes:
        _spec: AgentService 或 SkillForkRunner 生成的冻结运行输入。
        _turn_runner: 使用当前独立 Scheduler 的共享 ReAct 核心。
        _conversation: 只保存当前子 Agent 消息的内存对话。
        _context_manager: 当前运行自己的压缩、artifact 和用量状态。
        _hook_scope: 当前运行自己的 Hook once 和通知状态。
        _tool_context: 当前运行自己的文件缓存与 MCP 激活状态。
    """

    def __init__(
        self,
        spec: IndependentAgentSpec,
        turn_runner: AgentTurnRunner,
        conversation: Conversation,
        context_manager: ContextManager,
        instruction_manager: RuntimeInstructionManager,
        hook_engine: HookEngine,
        hook_scope: HookRunScope,
        skill_runtime: SkillRuntime,
        tool_context: ToolContext,
        permission_approver: AgentPermissionApprover,
        skill_approver: _AgentSkillApprover,
        workspace_service: AgentWorkspaceService,
    ) -> None:
        """保存 Builder 为本次运行创建的全部独立对象。

        Args:
            spec: 本次运行的冻结输入。
            turn_runner: 使用独立 Scheduler 的 ReAct 执行器。
            conversation: 仅保存本次子 Agent 消息的内存对话。
            context_manager: 本次运行独享的压缩和 artifact 状态。
            instruction_manager: 本次运行独享的环境与 Skill 指令状态。
            hook_engine: 应用共用 Hook 规则的执行引擎。
            hook_scope: 本次运行自己的 once 和通知状态。
            skill_runtime: 本次运行自己的活动 SOP 和资源范围。
            tool_context: 本次运行自己的文件缓存和 MCP 激活状态。
            permission_approver: 可以在移交后台时关闭交互的权限审批器。
            skill_approver: 可以在移交后台时关闭 Skill 信任确认的审批器。
            workspace_service: 标记运行状态并在资源关闭后收尾 Worktree 的服务。

        Returns:
            不返回数据；``start`` 会使用这些对象启动同一次运行。
        """

        self._spec = spec
        self._turn_runner = turn_runner
        self._conversation = conversation
        self._context_manager = context_manager
        self._instruction_manager = instruction_manager
        self._hook_engine = hook_engine
        self._hook_scope = hook_scope
        self._skill_runtime = skill_runtime
        self._tool_context = tool_context
        self._permission_approver = permission_approver
        self._skill_approver = skill_approver
        self._workspace_service = workspace_service

    @property
    def history(self):
        """返回本轮运行结束时的完整对话历史。

        Returns:
            初始历史以及本轮新增的用户、助手和工具结果消息。长期团队成员
            用它把新增部分追加到成员自己的 SessionManager。
        """

        return self._conversation.history

    def _finalization_profile(self) -> AgentFinalizationProfile:
        """根据内置角色名选择最后一次调用的报告要求。

        Returns:
            Explore、Plan、Verification 使用各自要求；Fork、general-purpose
            和用户自定义角色使用通用要求。
        """

        if self._spec.role is None:
            return AgentFinalizationProfile.GENERIC
        profiles = {
            "explore": AgentFinalizationProfile.EXPLORE,
            "plan": AgentFinalizationProfile.PLAN,
            "verification": AgentFinalizationProfile.VERIFICATION,
        }
        return profiles.get(
            self._spec.role.key,
            AgentFinalizationProfile.GENERIC,
        )

    def start(self) -> AgentRunHandle:
        """在当前事件循环启动子 Agent，并返回可等待或移交的句柄。

        Returns:
            持有同一 asyncio Task、取消令牌和审批器的 AgentRunHandle。
        """

        cancellation = CancellationToken()
        task = asyncio.create_task(self._run(cancellation))
        return AgentRunHandle(
            self._spec.run_id,
            cancellation,
            self._permission_approver,
            self._skill_approver,
            task,
        )

    async def _run(self, cancellation: CancellationToken) -> AgentRunResult:
        """执行 ReAct 循环、累计用量，并在所有出口清理独立资源。

        Args:
            cancellation: AgentRunHandle 暴露给前台和 TaskManager 的令牌。

        Returns:
            成功时带最终文本；错误或取消时带部分文本和原因的 AgentRunResult。
        """

        started = time.monotonic()
        usage = AgentUsage()
        visible_parts: list[str] = []
        final_text: str | None = None
        error: str | None = None
        status = BackgroundTaskStatus.FAILED
        try:
            await self._workspace_service.mark_running(self._spec)
            self._conversation.extend(self._spec.initial_messages)
            request = AgentTurnRequest(
                user_text=self._spec.task_prompt,
                history=lambda: self._conversation.history,
                append_messages=self._conversation.extend,
                context_manager=self._context_manager,
                instruction_manager=self._instruction_manager,
                stable_prompt=self._spec.prompt.stable,
                finalization_profile=self._finalization_profile(),
                options=AgentRunOptions(max_model_calls=self._spec.max_model_calls),
                cancellation=cancellation,
                hook_scope=self._hook_scope,
                load_memory_runtime=lambda: (),
                skill_runtime=self._skill_runtime,
                model_override=self._spec.model_override,
                tool_activation=self._tool_context.tool_activation,
                base_tool_view=self._spec.tool_view,
                fixed_runtime=(
                    self._spec.inherited_runtime
                    if self._spec.origin is IndependentAgentOrigin.FORK
                    else None
                ),
                user_already_in_history=(
                    self._spec.origin is IndependentAgentOrigin.FORK
                ),
            )
            async for event in self._turn_runner.stream(request):
                if isinstance(event, ModelTextDeltaEvent):
                    visible_parts.append(event.text)
                elif isinstance(event, ModelUsageEvent):
                    if event.usage is not None:
                        usage.input_tokens += event.usage.input_tokens
                        usage.output_tokens += event.usage.output_tokens
                        if event.usage.cached_input_tokens is not None:
                            usage.cached_input_tokens = (
                                (usage.cached_input_tokens or 0)
                                + event.usage.cached_input_tokens
                            )
                elif isinstance(event, ToolResultEvent):
                    usage.tool_calls += 1
                elif isinstance(event, FinalReplyEvent):
                    final_text = event.text
                    usage.model_calls = event.model_calls
                    status = (
                        BackgroundTaskStatus.PARTIAL
                        if event.completion_mode
                        is AgentCompletionMode.FORCED_FINALIZATION
                        else BackgroundTaskStatus.COMPLETED
                    )
                elif isinstance(event, AgentErrorEvent):
                    usage.model_calls = event.model_calls
                    error = event.message
                    status = (
                        BackgroundTaskStatus.CANCELLED
                        if event.code is AgentErrorCode.CANCELLED
                        else BackgroundTaskStatus.FAILED
                    )
            if final_text is None and error is None:
                error = "子 Agent 结束时没有返回最终文本"
        except asyncio.CancelledError:
            cancellation.cancel()
            error = "子 Agent 任务被取消"
            status = BackgroundTaskStatus.CANCELLED
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            status = BackgroundTaskStatus.FAILED
        finally:
            usage.duration_ms = int((time.monotonic() - started) * 1000)
            self._skill_runtime.clear()
            await self._hook_engine.close_scope(self._hook_scope)
            self._context_manager.close()
            self._tool_context.file_cache.clear()

        outcome = {
            BackgroundTaskStatus.COMPLETED: WorktreeTaskOutcome.COMPLETED,
            BackgroundTaskStatus.PARTIAL: WorktreeTaskOutcome.PARTIAL,
            BackgroundTaskStatus.FAILED: WorktreeTaskOutcome.FAILED,
            BackgroundTaskStatus.CANCELLED: WorktreeTaskOutcome.CANCELLED,
            BackgroundTaskStatus.INTERRUPTED: WorktreeTaskOutcome.INTERRUPTED,
        }[status]
        workspace_report = None
        try:
            workspace_report = await self._workspace_service.finish(
                self._spec,
                outcome,
            )
        except Exception as exc:
            assert self._spec.workspace is not None
            workspace_report = WorktreeFinishReport(
                workspace=self._spec.workspace,
                action=WorktreeFinishAction.SKIPPED,
                terminal_status=outcome,
                changes=None,
                reason="子 Agent 已结束，但工作区收尾没有完成",
                warnings=(str(exc) or type(exc).__name__,),
            )

        partial = "".join(visible_parts).strip() or None
        if status in {
            BackgroundTaskStatus.COMPLETED,
            BackgroundTaskStatus.PARTIAL,
        }:
            return AgentRunResult(
                status,
                final_text,
                None,
                None,
                usage,
                workspace_report,
            )
        return AgentRunResult(
            status,
            None,
            partial,
            error or "子 Agent 运行失败",
            usage,
            workspace_report,
        )

class IndependentAgentRuntimeBuilder:
    """为每次委派创建真实的独立运行组件，不复用父 Agent 可变状态。

    Attributes:
        _request_runner: 应用共用的 Provider 连接和请求事件转换器。
        _registry: 应用共用、只保存工具实现与注册顺序的 ToolRegistry。
        _hook_engine: 应用共用的 Hook 规则引擎。
        _workspace_root: 所有 Agent 实际操作的同一个项目根目录。
        _permission_settings: 子 Agent 也要读取的静态权限规则。
        _skill_catalog: LoadSkill 在独立运行中查询的有效 Skill 目录。
    """

    def __init__(
        self,
        request_runner: ProviderRequestRunner,
        registry: ToolRegistry,
        hook_engine: HookEngine,
        provider_config: ProviderConfig,
        workspace_root: Path,
        permission_settings: LoadedPermissionSettings,
        parent_permission_controller: PermissionController,
        permission_store: PermissionStore,
        interaction_approver: AgentInteractionApprover,
        skill_catalog: SkillCatalog,
        workspace_service: AgentWorkspaceService,
        *,
        user_memory_root: Path | None = None,
        secrets: Iterable[SecretValue] = (),
    ) -> None:
        """保存所有独立运行会共享的静态基础设施。

        Args:
            request_runner: 应用共用的 Provider 连接和流处理器。
            registry: 应用共用的工具实现注册表。
            hook_engine: 应用共用的 Hook 规则引擎。
            provider_config: 当前 Provider 的上下文和 artifact 预算。
            workspace_root: 文件工具和 artifact 所属的项目根目录。
            permission_settings: 用户级、项目级和本地静态权限规则。
            parent_permission_controller: 提供共享 LOCAL 状态，但不会复制其
                SESSION 临时批准。
            permission_store: 用户永久授权时写入本地配置的真实存储。
            interaction_approver: 前台子 Agent 使用的终端权限和信任界面。
            skill_catalog: LoadSkill 在子运行中查询的当前有效 Skill 目录。
            workspace_service: 为每次子运行标记和收尾固定工作区的服务。
            user_memory_root: read_file 可读取的用户记忆目录。
            secrets: artifact 和错误输出需要遮盖的敏感值。

        Returns:
            不返回数据；每次 ``build`` 才创建可变的独立运行状态。
        """

        self._request_runner = request_runner
        self._registry = registry
        self._hook_engine = hook_engine
        self._provider_config = provider_config
        self._workspace_root = workspace_root.resolve()
        self._permission_settings = permission_settings
        self._parent_permissions = parent_permission_controller
        self._permission_store = permission_store
        self._interaction_approver = interaction_approver
        self._skill_catalog = skill_catalog
        if not isinstance(workspace_service, AgentWorkspaceService):
            raise ValueError("workspace_service 类型无效")
        self._workspace_service = workspace_service
        self._user_memory_root = user_memory_root
        self._secrets = tuple(secrets)

    def build(self, spec: IndependentAgentSpec) -> IndependentAgentRunner:
        """为一份冻结 spec 创建完整且互不共享的运行对象。

        Args:
            spec: AgentService 已经解析好角色、模型、权限、工具和前后台的
                一次运行输入。

        Returns:
            尚未启动的 IndependentAgentRunner；调用 ``start`` 才创建任务。
        """

        if spec.workspace is None:
            raise ValueError("独立 Agent 必须先准备工作区再装配 Runner")
        conversation = Conversation()
        workspace_binding = (
            WorkspaceBinding.fixed(spec.workspace)
            if spec.workspace is not None
            else shared_workspace_binding(self._workspace_root)
        )
        workspace_root = workspace_binding.snapshot().root
        artifact_store = ArtifactStore(
            workspace_root,
            spec.run_id,
            secrets=self._secrets,
        )
        context_manager = ContextManager(
            self._request_runner,
            conversation,
            self._provider_config,
            artifact_store,
        )
        instructions = RuntimeInstructionManager(
            EnvironmentCollector(workspace_binding)
        )
        resources = SkillResourceAccess()
        trust = SkillTrustStore()
        skill_runtime = SkillRuntime(
            self._skill_catalog,
            instructions,
            resources,
            trust,
            maximum_allowlist=(
                spec.skill.allowed_tools if spec.skill is not None else None
            ),
        )
        if spec.skill is not None:
            skill_runtime.activate_temporary(
                spec.skill,
                spec.skill_arguments,
            )
        skill_router = _ChildSkillLoadRouter(
            self._skill_catalog,
            skill_runtime,
        )
        tool_context = ToolContext(
            workspace=workspace_binding,
            user_memory_root=self._user_memory_root,
            skill_resources=resources,
            skill_load_router=skill_router,  # type: ignore[arg-type]
            skill_load_scope="fork",
            team_actor=spec.team_actor,
        )
        child_permissions = self._parent_permissions.child(spec.permission_mode)
        permission_policy = PermissionPolicy(
            PermissionRuleResolver(
                self._permission_settings.user,
                self._permission_settings.project,
            ),
            child_permissions,
        )
        permission_approver = AgentPermissionApprover(
            self._interaction_approver,
            background=spec.background,
        )
        skill_approver = _AgentSkillApprover(
            self._interaction_approver,
            background=spec.background,
        )
        permission_interceptor = PermissionInterceptor(
            self._registry,
            tool_context,
            permission_policy,
            child_permissions,
            permission_approver,
            self._permission_store,
        )
        pre_hooks = PreToolHookInterceptor(self._hook_engine)
        post_hooks = PostToolHookObserver(self._hook_engine)
        executor = ToolExecutor(self._registry, tool_context)
        scheduler = ToolScheduler(
            self._registry,
            executor,
            interceptors=(
                PlanOnlyInterceptor(),
                SkillTrustInterceptor(
                    self._registry,
                    skill_approver,
                    trust,
                ),
                pre_hooks,
                permission_interceptor,
            ),
            observers=(post_hooks,),
        )
        hook_scope = self._hook_engine.create_scope(instructions)
        turn_runner = AgentTurnRunner(
            self._request_runner,
            self._registry,
            scheduler,
            self._hook_engine,
            secrets=self._secrets,
        )
        return IndependentAgentRunner(
            spec,
            turn_runner,
            conversation,
            context_manager,
            instructions,
            self._hook_engine,
            hook_scope,
            skill_runtime,
            tool_context,
            permission_approver,
            skill_approver,
            self._workspace_service,
        )

    async def abandon(
        self,
        spec: IndependentAgentSpec,
        reason: str,
    ) -> WorktreeFinishReport | None:
        """收尾一份已经准备但在 Runner 启动前被取消的 spec。

        Args:
            spec: TaskManager 队列中尚未开始运行的冻结输入。
            reason: 排队取消、应用关闭或装配失败的具体原因。

        Returns:
            spec 未准备时返回 ``None``；否则返回工作区实际删除或保留报告。
        """

        return await self._workspace_service.abandon(spec, reason)
