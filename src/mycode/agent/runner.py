"""执行主会话和 fork 共用的 ReAct 模型与工具循环。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field

from mycode.agent.cancellation import (
    CancellationController,
    CancellationReason,
    CancellationToken,
)
from mycode.agent.instructions import (
    RuntimeInstructionManager,
    deferred_tools_instruction,
)
from mycode.agent.finalization import (
    budget_instruction,
    finalization_instruction,
    parse_final_report,
)
from mycode.agents.snapshots import ParentRunRecorder
from mycode.context.manager import (
    CompactionMode,
    CompactionOutcomeKind,
    ContextManager,
)
from mycode.context.tool_results import ToolResultSaveFailure
from mycode.errors import (
    ContextWindowExceededError,
    MyCodeError,
    StreamProtocolError,
    TransportError,
    redact_secrets,
)
from mycode.hooks.engine import HookEngine
from mycode.hooks.runtime import HookRunScope
from mycode.models.config import SecretValue
from mycode.models.events import (
    AgentCompletionMode,
    AgentErrorCode,
    AgentErrorEvent,
    AgentEvent,
    AgentRunOptions,
    AgentFinalizationProfile,
    AgentWarningEvent,
    CompactionStatusEvent,
    CompactionStatusKind,
    FinalReplyEvent,
    ModelTextDeltaEvent,
    ModelUsageEvent,
    ThinkingDeltaEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from mycode.models.messages import (
    AssistantMessage,
    ChatMessage,
    TextBlock,
    ToolResultMessage,
    UserMessage,
)
from mycode.models.model_calls import ModelCallBudget, ModelCallPurpose
from mycode.models.hooks import HookContext, HookEvent
from mycode.models.prompts import PromptContext, RuntimeInstruction
from mycode.models.provider import (
    ModelStopReason,
    ProviderCompleted,
    ProviderRequest,
    ProviderTextDelta,
    ProviderThinkingDelta,
    ToolChoice,
)
from mycode.models.tools import (
    ToolErrorCode,
    ToolExecutionResult,
    ToolActivationState,
    ToolView,
)
from mycode.providers.runner import (
    ProviderRequestCancelled,
    ProviderRequestRunner,
)
from mycode.skills.runtime import SkillRuntime
from mycode.tools.registry import ToolRegistry
from mycode.tools.scheduler import ToolScheduleSession, ToolScheduler


class _CancellationObserved(Exception):
    """把 Provider 深层取消快速送回当前运行循环。"""


@dataclass(frozen=True)
class AgentTurnRequest:
    """保存主会话或 fork 执行一条用户请求所需的真实运行对象。

    AgentLoop 使用 SessionManager 的 history 和 append；SkillForkRunner 使用
    临时 Conversation 的 history 和 extend。两者共享同一套 ContextManager、
    工具调度和 Provider 事件处理逻辑。
    """

    # 用户本轮输入；主会话是原始文本，fork 是触发 Skill 的任务说明。
    user_text: str
    # 读取当前对话历史。压缩后再次调用会拿到更新后的消息。
    history: Callable[[], tuple[ChatMessage, ...]]
    # 把已确认发生的用户、助手和工具结果写入当前主会话或临时会话。
    append_messages: Callable[[tuple[ChatMessage, ...]], None]
    # 管理当前对话的 Token、摘要和工具结果 artifact。
    context_manager: ContextManager
    # 提供环境、Skill 目录、活动 SOP、通知和 Plan 规则。
    instruction_manager: RuntimeInstructionManager
    # 每次请求都会发送的固定系统提示词。
    stable_prompt: str
    # 最后一次无工具请求采用的报告要求。
    finalization_profile: AgentFinalizationProfile
    # 当前运行配置，包括 Plan、模型调用上限、并发和整条任务超时。
    options: AgentRunOptions
    # 外部取消令牌；主 UI 和 fork 调用方都可中止运行。
    cancellation: CancellationToken
    # 当前主会话或 fork 独享的 once、提示词和后台 Hook 状态。
    hook_scope: HookRunScope
    # 主会话读取长期记忆索引；fork 不需要时返回空元组。
    load_memory_runtime: Callable[[], tuple[RuntimeInstruction, ...]]
    # 当前会话 Skill 状态。未启用 Skill 系统时为 None。
    skill_runtime: SkillRuntime | None = None
    # fork frontmatter 指定的单次模型；主会话为 None。
    model_override: str | None = None
    # 主会话用来把完整回合交给记忆后台；fork 为 None。
    completed_turn: Callable[[tuple[ChatMessage, ...]], None] | None = None
    # 当前运行自己的 MCP 激活集合；tool_search 只修改对应 ToolContext 中的同一实例。
    tool_activation: ToolActivationState = field(
        default_factory=ToolActivationState
    )
    # 子 Agent 工具策略提前写入的全来源白名单和禁止集合。
    base_tool_view: ToolView = field(default_factory=ToolView)
    # 每次 Provider 请求前重新叠加的运行时工具策略。团队
    # 创建或删除可能发生在同一个 ReAct 回合中，因此不能只在
    # 用户输入开始时计算一次。
    resolve_tool_view: Callable[[ToolView], ToolView] = lambda view: view
    # 主会话记录实际 Provider 前缀时使用；独立子 Agent 传 None，不能继续 Fork。
    parent_recorder: ParentRunRecorder | None = None
    # Fork 首次及后续请求原样复用父 PromptContext.runtime；None 才动态收集环境。
    fixed_runtime: tuple[RuntimeInstruction, ...] | None = None
    # Fork 的 initial_messages 已包含最后一条任务 UserMessage；True 时不重复追加。
    user_already_in_history: bool = False
    # 每次 Provider 请求前排空当前主会话的后台通知；独立运行返回空元组。
    drain_external_messages: Callable[[], tuple[ChatMessage, ...]] = lambda: ()
    # False 用于后台通知唤醒回合：消息仍写入历史，但 UI 不显示成用户手输。
    emit_user_event: bool = True

    def __post_init__(self) -> None:
        """检查运行所需的文本字段不是空白。

        Returns:
            请求字段可直接进入 ReAct 循环时不返回数据。

        Raises:
            ValueError: 用户输入或固定系统提示词为空。
        """

        if not self.user_text.strip():
            raise ValueError("用户消息不能为空")
        if not self.stable_prompt.strip():
            raise ValueError("Agent 固定提示词不能为空")
        if not isinstance(self.finalization_profile, AgentFinalizationProfile):
            raise ValueError("Agent 强制收尾 profile 无效")


class AgentTurnRunner:
    """执行一条消息的多轮模型请求、工具调用和错误收尾。

    Attributes:
        _request_runner: 发送 ProviderRequest 并处理流关闭与取消的对象。
        _registry: 按当前 ToolView 生成 definitions 的共享工具注册表。
        _scheduler: 执行工具并应用权限、Skill 信任和 Hook 的调度器。
        _hook_engine: 派发 turn、message、compact 和 error 事件的规则引擎。
        _secrets: 用户可见错误和警告中要遮盖的敏感值。
    """

    def __init__(
        self,
        request_runner: ProviderRequestRunner,
        registry: ToolRegistry,
        scheduler: ToolScheduler,
        hook_engine: HookEngine,
        *,
        secrets: Iterable[SecretValue] = (),
    ) -> None:
        """连接主会话和 fork 共用的 Provider 与工具基础设施。

        Args:
            request_runner: 发送统一 ProviderRequest 并处理流关闭和取消。
            registry: 每轮计算实际模型可见工具定义。
            scheduler: 按读写顺序执行模型返回的工具调用。
            hook_engine: 派发主会话和 fork 共用的轮次、消息与压缩事件。
            secrets: 用户可见错误输出前需要替换的密钥集合。

        Returns:
            不返回数据；每次 ``stream`` 接收一份具体运行状态后开始工作。

        Returns:
            None。
        """

        # 主会话与 fork 共用同一个 Provider 连接池。
        self._request_runner = request_runner
        # LoadSkill 或 MCP 激活后，下一模型轮从这里取得最新工具视图。
        self._registry = registry
        # 调度器负责拦截器顺序、并发和可靠取消。
        self._scheduler = scheduler
        # 规则定义共用，具体 once 和提示词状态从每个 AgentTurnRequest 取得。
        self._hooks = hook_engine
        # 错误事件只保存脱敏后的说明。
        self._secrets = tuple(secrets)

    @staticmethod
    def _runtime_for(
        run: AgentTurnRequest,
        *,
        preview: bool,
    ) -> tuple[RuntimeInstruction, ...]:
        """取得本轮应发送的运行时指令。

        Args:
            run: 当前主会话、定义式或 Fork 的冻结运行对象。
            preview: ``True`` 表示只估算请求、不消费一次性通知；``False``
                表示正式发送并消费通知。

        Returns:
            Fork 指定 ``fixed_runtime`` 时原样返回父请求指令；其他运行从
            RuntimeInstructionManager 预览或正式取得当前指令。
        """

        if run.fixed_runtime is not None:
            return run.fixed_runtime
        if preview:
            return run.instruction_manager.preview(
                plan_only=run.options.plan_only
            )
        return run.instruction_manager.prepare(
            plan_only=run.options.plan_only
        )

    def build_request(
        self,
        run: AgentTurnRequest,
        messages: tuple[ChatMessage, ...],
        runtime: tuple[RuntimeInstruction, ...],
        memory_runtime: tuple[RuntimeInstruction, ...],
        *,
        include_checkpoint: bool = True,
        finalizing: bool = False,
    ) -> tuple[ProviderRequest, ToolView]:
        """构造本轮 ProviderRequest 和对应的工具名快照。

        Args:
            run: 当前主会话或 fork 的运行对象。
            messages: 当前上下文中仍可见的对话消息。
            runtime: 本轮环境、活动 SOP、通知和模式指令。
            memory_runtime: 主会话按需加载的长期记忆索引。
            include_checkpoint: False 时不加入当前压缩摘要，供恢复预检使用。
            finalizing: True 时不向模型暴露工具，供最后一次正式回答使用。

        Returns:
            ProviderRequest 与同一批定义对应的 ToolView。Scheduler 必须使用
            返回视图中的 visible_tool_names 复查模型调用。
        """

        checkpoint = (
            run.context_manager.checkpoint_instructions
            if include_checkpoint
            else ()
        )
        skill_allowlist = (
            run.skill_runtime.merged_allowlist()
            if run.skill_runtime is not None
            else None
        )
        base_view = run.resolve_tool_view(run.base_tool_view)
        base_allowlist = base_view.business_allowlist
        if skill_allowlist is None:
            business_allowlist = base_allowlist
        elif base_allowlist is None:
            business_allowlist = skill_allowlist
        else:
            business_allowlist = skill_allowlist & base_allowlist
        requested_view = ToolView(
            active_skill_names=(
                run.skill_runtime.active_names
                if run.skill_runtime is not None
                else base_view.active_skill_names
            ),
            business_allowlist=business_allowlist,
            active_mcp_names=frozenset(
                run.tool_activation.active_mcp_names
            ),
            final_allowlist=base_view.final_allowlist,
            denied_tool_names=base_view.denied_tool_names,
        )
        if finalizing:
            definitions = ()
            resolved_view = ToolView()
        else:
            definitions, resolved_view = self._registry.definitions_for(
                requested_view
            )
        return (
            ProviderRequest(
                messages=messages,
                tools=definitions,
                tool_choice=(ToolChoice.NONE if finalizing else ToolChoice.AUTO),
                prompt=PromptContext(
                    stable=run.stable_prompt,
                    runtime=(*checkpoint, *memory_runtime, *runtime),
                ),
                model_override=run.model_override,
            ),
            resolved_view,
        )

    @staticmethod
    def _tool_messages(
        results: tuple[ToolExecutionResult, ...],
    ) -> tuple[ToolResultMessage, ...]:
        """把工具执行结果转换成下一轮模型能读取的消息。

        Args:
            results: Scheduler 按原调用顺序返回的完整结构化结果。

        Returns:
            保留成功状态、错误码和截断信息的 ToolResultMessage 元组。
        """

        return tuple(
            ToolResultMessage(
                result.tool_call_id,
                result.tool_name,
                result.to_model_json(),
                not result.success,
            )
            for result in results
        )

    def _safe_error(
        self,
        code: AgentErrorCode,
        message: str,
        model_calls: int,
    ) -> AgentErrorEvent:
        """生成不含已配置密钥的用户可见错误事件。

        Args:
            code: UI 用来分类展示的稳定错误码。
            message: 内部产生、可能包含密钥的原始说明。
            model_calls: 错误发生前已发送的模型请求数。

        Returns:
            已完成密钥替换的 AgentErrorEvent。
        """

        return AgentErrorEvent(
            code,
            redact_secrets(message, self._secrets),
            model_calls,
        )

    def _cancel_error(
        self,
        controller: CancellationController,
        model_calls: int,
    ) -> AgentErrorEvent:
        """根据用户取消或整轮截止时间生成错误事件。

        Args:
            controller: 保存本次取消来源的整轮控制器。
            model_calls: 取消前已发送的模型请求数。

        Returns:
            DEADLINE_EXCEEDED 或 CANCELLED 事件。
        """

        if controller.reason is CancellationReason.DEADLINE:
            return self._safe_error(
                AgentErrorCode.DEADLINE_EXCEEDED,
                "Agent 请求超过整轮截止时间",
                model_calls,
            )
        return self._safe_error(
            AgentErrorCode.CANCELLED,
            "Agent 请求已取消",
            model_calls,
        )

    def _tool_result_warning(
        self,
        failures: tuple[ToolResultSaveFailure, ...],
    ) -> AgentWarningEvent:
        """把本批 artifact 保存失败合成一条脱敏警告。

        Args:
            failures: 每个未能落盘的工具名、调用 ID 和原因。

        Returns:
            可直接交给 UI 的 AgentWarningEvent。
        """

        lines = ["部分工具结果无法保存为 artifact，已改用完整正文："]
        lines.extend(
            f"- {failure.tool_name}（调用 ID：{failure.tool_call_id}）："
            f"{failure.reason}"
            for failure in failures
        )
        return AgentWarningEvent(
            redact_secrets("\n".join(lines), self._secrets)
        )

    def _commit_tool_round(
        self,
        run: AgentTurnRequest,
        assistant: AssistantMessage,
        results: tuple[ToolExecutionResult, ...],
    ) -> tuple[
        tuple[ToolResultSaveFailure, ...],
        tuple[ChatMessage, ...],
    ]:
        """压缩过长工具结果并提交助手调用与对应结果。

        Args:
            run: 提供 ContextManager 和当前对话写入函数的运行对象。
            assistant: 包含本批 ToolCall 的完整助手消息。
            results: 工具实际执行结果，顺序与 ToolCall 一致。

        Returns:
            artifact 保存失败列表和实际写入对话的消息。
        """

        original_results = results
        try:
            outcome = run.context_manager.compact_tool_results(results)
        except MyCodeError:
            messages: tuple[ChatMessage, ...] = (
                assistant,
                *self._tool_messages(original_results),
            )
            run.append_messages(messages)
            raise
        messages = (assistant, *self._tool_messages(outcome.results))
        run.append_messages(messages)
        return outcome.failures, messages

    @staticmethod
    def _request_message(
        run: AgentTurnRequest,
        history: tuple[ChatMessage, ...],
    ) -> str:
        """返回 `pre_send` 条件和模板可读取的本次请求摘要。

        Args:
            run: 提供本轮原始用户输入。
            history: 即将发送给 Provider 的当前消息历史。

        Returns:
            首次请求返回用户输入；工具回灌后返回最近一条工具结果正文。
        """

        if history and isinstance(history[-1], ToolResultMessage):
            return history[-1].content
        return run.user_text

    @staticmethod
    def _response_message(assistant: AssistantMessage) -> str:
        """生成模型完整响应对应的 Hook 消息文本。

        Args:
            assistant: Provider 完成事件携带的完整助手消息。

        Returns:
            有回复正文时返回完整正文；只有工具调用时返回逗号分隔的工具名。
        """

        if assistant.text:
            return assistant.text
        return ", ".join(call.name for call in assistant.tool_calls)

    async def stream(
        self,
        run: AgentTurnRequest,
    ) -> AsyncIterator[AgentEvent]:
        """在共享 ReAct 核心外派发每轮只发生一次的 Hook。

        Args:
            run: 当前主会话或 fork 的完整运行对象及 Hook scope。

        Yields:
            与核心循环相同的 AgentEvent；错误事件会先触发 `error` Hook。
        """

        await self._hooks.dispatch(
            HookContext(HookEvent.TURN_START, message=run.user_text),
            run.hook_scope,
        )
        core = self._stream_core(run)
        try:
            async for event in core:
                if isinstance(event, AgentErrorEvent):
                    await self._hooks.dispatch(
                        HookContext(HookEvent.ERROR, error=event.message),
                        run.hook_scope,
                    )
                yield event
        finally:
            await core.aclose()
            await asyncio.shield(
                self._hooks.dispatch(
                    HookContext(HookEvent.TURN_END, message=run.user_text),
                    run.hook_scope,
                )
            )

    async def _stream_core(
        self,
        run: AgentTurnRequest,
    ) -> AsyncIterator[AgentEvent]:
        """执行一条用户请求，直到最终回答、错误或取消。

        Args:
            run: 当前主会话或 fork 的历史、上下文、Skill、配置和取消对象。

        Yields:
            用户消息、模型增量、工具事件、压缩状态、警告及唯一结束事件。
        """

        user_message = UserMessage(run.user_text)
        turn_messages: list[ChatMessage] = [user_message]
        budget = ModelCallBudget(run.options.max_model_calls)
        current_session: ToolScheduleSession | None = None
        current_assistant: AssistantMessage | None = None
        current_result_indexes: set[int] = set()
        active_model_call_number: int | None = None

        try:
            if run.emit_user_event:
                yield UserMessageEvent(run.user_text)
            if not run.user_already_in_history:
                run.append_messages((user_message,))
            async with CancellationController(
                run.cancellation,
                run.options.overall_timeout_seconds,
            ) as controller:
                try:
                    while True:
                        if controller.token.is_cancelled:
                            yield self._cancel_error(
                                controller,
                                budget.used_model_calls,
                            )
                            return

                        external_messages = run.drain_external_messages()
                        if external_messages:
                            run.append_messages(external_messages)
                            turn_messages.extend(external_messages)
                        candidate = run.history()
                        try:
                            memory_runtime = run.load_memory_runtime()
                        except MyCodeError as exc:
                            memory_runtime = ()
                            yield AgentWarningEvent(
                                redact_secrets(
                                    f"无法加载长期记忆索引：{exc}",
                                    self._secrets,
                                )
                            )
                        preview_runtime = self._runtime_for(run, preview=True)
                        deferred = (
                            None
                            if run.fixed_runtime is not None
                            else deferred_tools_instruction(
                                self._registry.deferred_mcp_names_for(
                                    run.tool_activation.active_mcp_names
                                )
                            )
                        )
                        if deferred is not None:
                            preview_runtime = (*preview_runtime, deferred)
                        preview_request, _ = self.build_request(
                            run,
                            candidate,
                            preview_runtime,
                            memory_runtime,
                        )
                        if run.context_manager.should_auto_compact(
                            preview_request
                        ):
                            yield CompactionStatusEvent(
                                CompactionStatusKind.STARTED,
                                "对话接近上下文上限，正在自动压缩较早内容…",
                            )
                            outcome = await run.context_manager.compact(
                                CompactionMode.AUTO,
                                controller.token,
                                model_call_budget=budget,
                            )
                            for record in outcome.model_call_records:
                                yield ModelUsageEvent(
                                    record.model_call_number,
                                    record.usage,
                                    record.purpose,
                                )
                            if outcome.kind is CompactionOutcomeKind.SUCCEEDED:
                                yield CompactionStatusEvent(
                                    CompactionStatusKind.SUCCEEDED,
                                    outcome.message,
                                )
                                await self._hooks.dispatch(
                                    HookContext(
                                        HookEvent.COMPACT,
                                        message=outcome.message,
                                    ),
                                    run.hook_scope,
                                )
                                candidate = run.history()
                            elif outcome.kind is CompactionOutcomeKind.CANCELLED:
                                yield self._cancel_error(
                                    controller,
                                    budget.used_model_calls,
                                )
                                return
                            elif outcome.kind not in {
                                CompactionOutcomeKind.NO_CONTENT,
                                CompactionOutcomeKind.FINAL_CALL_RESERVED,
                            }:
                                kind = (
                                    CompactionStatusKind.CIRCUIT_OPEN
                                    if outcome.kind
                                    is CompactionOutcomeKind.CIRCUIT_OPEN
                                    else CompactionStatusKind.FAILED
                                )
                                yield CompactionStatusEvent(kind, outcome.message)
                                return

                        await self._hooks.dispatch(
                            HookContext(
                                HookEvent.PRE_SEND,
                                message=self._request_message(run, candidate),
                            ),
                            run.hook_scope,
                        )
                        runtime = self._runtime_for(run, preview=False)
                        if deferred is not None:
                            runtime = (*runtime, deferred)
                        finalizing = budget.finalization_required
                        if finalizing:
                            runtime = (
                                *runtime,
                                finalization_instruction(
                                    run.finalization_profile
                                ),
                            )
                        else:
                            reminder = budget_instruction(
                                budget.remaining_model_calls
                            )
                            if reminder is not None:
                                runtime = (*runtime, reminder)
                        request, tool_view = self.build_request(
                            run,
                            candidate,
                            runtime,
                            memory_runtime,
                            finalizing=finalizing,
                        )
                        if run.parent_recorder is not None:
                            run.parent_recorder.record_request(request, tool_view)
                        model_call_number = budget.begin(
                            ModelCallPurpose.AGENT
                        )
                        active_model_call_number = model_call_number
                        completed: ProviderCompleted | None = None
                        emergency_retried = False
                        while True:
                            try:
                                async for event in self._request_runner.events(
                                    request,
                                    controller.token,
                                ):
                                    if isinstance(event, ProviderThinkingDelta):
                                        if not finalizing:
                                            yield ThinkingDeltaEvent(
                                                model_call_number,
                                                event.text,
                                            )
                                    elif isinstance(event, ProviderTextDelta):
                                        if not finalizing:
                                            yield ModelTextDeltaEvent(
                                                model_call_number,
                                                event.text,
                                            )
                                    else:
                                        completed = event
                                break
                            except ProviderRequestCancelled as exc:
                                raise _CancellationObserved from exc
                            except ContextWindowExceededError:
                                record = budget.finish(
                                    model_call_number,
                                    None,
                                )
                                active_model_call_number = None
                                yield ModelUsageEvent(
                                    record.model_call_number,
                                    record.usage,
                                    record.purpose,
                                )
                                if finalizing:
                                    yield self._safe_error(
                                        AgentErrorCode.MAX_MODEL_CALLS,
                                        "最后一次模型调用因上下文超限失败，已没有额度再次生成正式报告",
                                        budget.used_model_calls,
                                    )
                                    return
                                if emergency_retried:
                                    raise
                                emergency_retried = True
                                yield CompactionStatusEvent(
                                    CompactionStatusKind.STARTED,
                                    "Provider 拒绝了过长请求，正在紧急压缩…",
                                )
                                outcome = await run.context_manager.compact(
                                    CompactionMode.EMERGENCY,
                                    controller.token,
                                    model_call_budget=budget,
                                )
                                for record in outcome.model_call_records:
                                    yield ModelUsageEvent(
                                        record.model_call_number,
                                        record.usage,
                                        record.purpose,
                                    )
                                if outcome.kind is not CompactionOutcomeKind.SUCCEEDED:
                                    if outcome.kind is CompactionOutcomeKind.CANCELLED:
                                        yield self._cancel_error(
                                            controller,
                                            budget.used_model_calls,
                                        )
                                        return
                                    if outcome.kind is CompactionOutcomeKind.FINAL_CALL_RESERVED:
                                        pass
                                    else:
                                        kind = (
                                            CompactionStatusKind.CIRCUIT_OPEN
                                            if outcome.kind
                                            is CompactionOutcomeKind.CIRCUIT_OPEN
                                            else CompactionStatusKind.FAILED
                                        )
                                        yield CompactionStatusEvent(
                                            kind,
                                            outcome.message,
                                        )
                                        return
                                if outcome.kind is CompactionOutcomeKind.SUCCEEDED:
                                    yield CompactionStatusEvent(
                                        CompactionStatusKind.SUCCEEDED,
                                        outcome.message,
                                    )
                                    await self._hooks.dispatch(
                                        HookContext(
                                            HookEvent.COMPACT,
                                            message=outcome.message,
                                        ),
                                        run.hook_scope,
                                    )
                                candidate = run.history()
                                await self._hooks.dispatch(
                                    HookContext(
                                        HookEvent.PRE_SEND,
                                        message=self._request_message(
                                            run,
                                            candidate,
                                        ),
                                    ),
                                    run.hook_scope,
                                )
                                runtime = self._runtime_for(run, preview=False)
                                if deferred is not None:
                                    runtime = (*runtime, deferred)
                                finalizing = budget.finalization_required
                                if finalizing:
                                    runtime = (
                                        *runtime,
                                        finalization_instruction(
                                            run.finalization_profile
                                        ),
                                    )
                                else:
                                    reminder = budget_instruction(
                                        budget.remaining_model_calls
                                    )
                                    if reminder is not None:
                                        runtime = (*runtime, reminder)
                                request, tool_view = self.build_request(
                                    run,
                                    candidate,
                                    runtime,
                                    memory_runtime,
                                    finalizing=finalizing,
                                )
                                if run.parent_recorder is not None:
                                    run.parent_recorder.record_request(
                                        request,
                                        tool_view,
                                    )
                                completed = None
                                model_call_number = budget.begin(
                                    ModelCallPurpose.AGENT
                                )
                                active_model_call_number = model_call_number

                        if completed is None:
                            raise StreamProtocolError(
                                "Provider 完成事件缺少响应"
                            )
                        record = budget.finish(
                            model_call_number,
                            completed.usage,
                        )
                        active_model_call_number = None
                        yield ModelUsageEvent(
                            record.model_call_number,
                            record.usage,
                            record.purpose,
                        )
                        run.context_manager.record_normal_usage(
                            request,
                            completed.usage,
                        )
                        assistant = completed.assistant_message
                        if run.parent_recorder is not None:
                            run.parent_recorder.record_response(assistant)
                        calls = assistant.tool_calls
                        await self._hooks.dispatch(
                            HookContext(
                                HookEvent.POST_RECEIVE,
                                message=self._response_message(assistant),
                            ),
                            run.hook_scope,
                        )

                        if completed.stop_reason is ModelStopReason.MAX_TOKENS:
                            yield self._safe_error(
                                AgentErrorCode.MODEL_OUTPUT_TRUNCATED,
                                "模型输出达到长度限制，本次调用未形成完整回复",
                                budget.used_model_calls,
                            )
                            return
                        if finalizing:
                            report = parse_final_report(assistant.text)
                            if report is None:
                                yield self._safe_error(
                                    AgentErrorCode.MAX_MODEL_CALLS,
                                    "已达到最大模型调用次数，且最后一次调用没有生成可用的正式报告",
                                    budget.used_model_calls,
                                )
                                return
                            clean_assistant = AssistantMessage(
                                (TextBlock(report),)
                            )
                            run.append_messages((clean_assistant,))
                            turn_messages.append(clean_assistant)
                            if run.completed_turn is not None:
                                try:
                                    run.completed_turn(tuple(turn_messages))
                                except RuntimeError:
                                    pass
                            yield ModelTextDeltaEvent(
                                model_call_number,
                                report,
                            )
                            yield FinalReplyEvent(
                                report,
                                budget.used_model_calls,
                                AgentCompletionMode.FORCED_FINALIZATION,
                            )
                            return
                        if completed.stop_reason is ModelStopReason.END_TURN:
                            if calls:
                                raise StreamProtocolError(
                                    "模型正常结束时仍包含工具调用"
                                )
                            if not assistant.text:
                                raise StreamProtocolError(
                                    "模型正常结束但没有回答文本"
                                )
                            run.append_messages((assistant,))
                            turn_messages.append(assistant)
                            if run.completed_turn is not None:
                                try:
                                    run.completed_turn(tuple(turn_messages))
                                except RuntimeError:
                                    pass
                            yield FinalReplyEvent(
                                assistant.text,
                                budget.used_model_calls,
                            )
                            return
                        if (
                            completed.stop_reason is not ModelStopReason.TOOL_USE
                            or not calls
                        ):
                            raise StreamProtocolError(
                                "模型工具结束原因与调用内容不一致"
                            )

                        current_assistant = assistant
                        current_session = self._scheduler.schedule(
                            calls,
                            model_call_number=model_call_number,
                            options=run.options,
                            cancellation=controller.token,
                            hook_scope=run.hook_scope,
                            visible_tool_names=(
                                tool_view.visible_tool_names
                                if run.skill_runtime is not None
                                else None
                            ),
                        )
                        async for tool_event in current_session.stream():
                            if isinstance(tool_event, ToolResultEvent):
                                current_result_indexes.add(
                                    tool_event.invocation.call_index
                                )
                            yield tool_event

                        if controller.token.is_cancelled:
                            results = await current_session.finalize(
                                ToolErrorCode.CANCELLED
                            )
                            for invocation, result in zip(
                                current_session.invocations,
                                results,
                                strict=True,
                            ):
                                if invocation.call_index not in current_result_indexes:
                                    yield ToolResultEvent(invocation, result)
                            current_session = None
                            current_assistant = None
                            failures, committed = self._commit_tool_round(
                                run,
                                assistant,
                                results,
                            )
                            turn_messages.extend(committed)
                            if failures:
                                yield self._tool_result_warning(failures)
                            current_result_indexes.clear()
                            yield self._cancel_error(
                                controller,
                                budget.used_model_calls,
                            )
                            return

                        results = await current_session.finalize()
                        current_session = None
                        current_assistant = None
                        failures, committed = self._commit_tool_round(
                            run,
                            assistant,
                            results,
                        )
                        turn_messages.extend(committed)
                        if failures:
                            yield self._tool_result_warning(failures)
                        current_result_indexes.clear()
                except _CancellationObserved:
                    if active_model_call_number is not None:
                        record = budget.finish(active_model_call_number, None)
                        active_model_call_number = None
                        yield ModelUsageEvent(
                            record.model_call_number,
                            record.usage,
                            record.purpose,
                        )
                    yield self._cancel_error(
                        controller,
                        budget.used_model_calls,
                    )
                    return
                except StreamProtocolError as exc:
                    if active_model_call_number is not None:
                        record = budget.finish(active_model_call_number, None)
                        active_model_call_number = None
                        yield ModelUsageEvent(
                            record.model_call_number,
                            record.usage,
                            record.purpose,
                        )
                    yield self._safe_error(
                        AgentErrorCode.PROTOCOL_ERROR,
                        str(exc),
                        budget.used_model_calls,
                    )
                    return
                except TransportError as exc:
                    if active_model_call_number is not None:
                        record = budget.finish(active_model_call_number, None)
                        active_model_call_number = None
                        yield ModelUsageEvent(
                            record.model_call_number,
                            record.usage,
                            record.purpose,
                        )
                    yield self._safe_error(
                        AgentErrorCode.PROVIDER_ERROR,
                        str(exc),
                        budget.used_model_calls,
                    )
                    return
                except MyCodeError as exc:
                    if active_model_call_number is not None:
                        record = budget.finish(active_model_call_number, None)
                        active_model_call_number = None
                        yield ModelUsageEvent(
                            record.model_call_number,
                            record.usage,
                            record.purpose,
                        )
                    yield self._safe_error(
                        AgentErrorCode.INTERNAL_ERROR,
                        str(exc),
                        budget.used_model_calls,
                    )
                    return
                except asyncio.CancelledError:
                    if active_model_call_number is not None:
                        record = budget.finish(active_model_call_number, None)
                        active_model_call_number = None
                        yield ModelUsageEvent(
                            record.model_call_number,
                            record.usage,
                            record.purpose,
                        )
                    if (
                        current_session is not None
                        and current_assistant is not None
                    ):
                        results = await current_session.finalize(
                            ToolErrorCode.CANCELLED
                        )
                        for invocation, result in zip(
                            current_session.invocations,
                            results,
                            strict=True,
                        ):
                            if invocation.call_index not in current_result_indexes:
                                yield ToolResultEvent(invocation, result)
                        failures, committed = self._commit_tool_round(
                            run,
                            current_assistant,
                            results,
                        )
                        turn_messages.extend(committed)
                        if failures:
                            yield self._tool_result_warning(failures)
                        current_session = None
                        current_assistant = None
                        current_result_indexes.clear()
                    yield self._safe_error(
                        AgentErrorCode.CANCELLED,
                        "Agent 请求已取消",
                        budget.used_model_calls,
                    )
                    return
        finally:
            if current_session is not None and current_assistant is not None:
                results = await current_session.finalize(
                    ToolErrorCode.CANCELLED
                )
                self._commit_tool_round(run, current_assistant, results)
