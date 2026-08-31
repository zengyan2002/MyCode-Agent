"""协调工具落盘、Token 预算、全量摘要、重试和熔断。"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum

from mycode.agent.cancellation import CancellationToken
from mycode.agent.conversation import Conversation
from mycode.constants import (
    AUTO_COMPACTION_MARGIN_TOKENS,
    MANUAL_COMPACTION_MARGIN_TOKENS,
    MAX_COMPACTION_ATTEMPTS,
    RECENT_MESSAGE_GROUPS,
    RECENT_MESSAGE_TOKENS,
    SESSION_RESTORE_MAX_COMPACTIONS,
)
from mycode.context.artifacts import ArtifactStore, StagedArtifact
from mycode.context.estimator import TokenEstimator, estimate_text
from mycode.context.history import HistoryPartitioner, MessageGroup, flatten_groups
from mycode.context.summary import CompactionMaterial, SummaryCodec
from mycode.context.tool_results import (
    ToolResultCompactionOutcome,
    ToolResultCompactor,
)
from mycode.errors import (
    AuthenticationError,
    ContextWindowExceededError,
    MyCodeError,
    TransportError,
    redact_secrets,
)
from mycode.models.config import ProviderConfig
from mycode.models.messages import ChatMessage, UserMessage
from mycode.models.model_calls import (
    ModelCallBudget,
    ModelCallPurpose,
    ModelCallRecord,
)
from mycode.models.prompts import (
    PromptContext,
    RuntimeInstruction,
    RuntimeInstructionKind,
)
from mycode.models.provider import ProviderRequest, ProviderUsage
from mycode.models.tools import ToolExecutionResult
from mycode.providers.runner import (
    ProviderRequestCancelled,
    ProviderRequestRunner,
)

_BOUNDARY_TEXT = (
    "较早对话已经被结构化摘要替代。摘要可能省略文件和工具输出细节；"
    "需要具体代码、日志或 artifact 内容时，必须使用 read_file 重新读取，"
    "不要根据摘要补写或猜测。"
)


class CompactionMode(str, Enum):
    """记录这次上下文摘要是自动触发、用户手动触发，还是超限后紧急触发"""

    # 上下文接近模型上限时，由程序自动触发
    AUTO = "auto"
    # 用户执行 /compact 时触发
    MANUAL = "manual"
    # 模型服务返回上下文超限后，为了重试请求而触发
    EMERGENCY = "emergency"


class CompactionOutcomeKind(str, Enum):
    """记录一次上下文摘要是成功、无内容、失败、停止重试还是被取消"""

    # 摘要成功，较早对话已经被摘要替换
    SUCCEEDED = "succeeded"
    # 当前没有足够的较早对话可以摘要
    NO_CONTENT = "no_content"
    # 本次摘要失败
    FAILED = "failed"
    # 连续失败达到上限，程序已经停止自动重试
    CIRCUIT_OPEN = "circuit_open"
    # 摘要过程被用户取消
    CANCELLED = "cancelled"
    # 当前 Agent 只剩最后一次模型调用，压缩没有占用它。
    FINAL_CALL_RESERVED = "final_call_reserved"


@dataclass(frozen=True)
class CompactionOutcome:
    """保存一次上下文摘要完成后的状态和用户提示。

    这个对象返回给 AgentLoop。message 可以显示给用户，但不包含生成的摘要正文
    """

    # 摘要成功、失败、无内容、停止重试或被取消
    kind: CompactionOutcomeKind
    # 程序生成的用户提示，例如摘要成功、取消或失败原因
    message: str
    # 这次压缩实际发出的 Provider 请求，供 Agent 运行器上报调用次数和 Token。
    model_call_records: tuple[ModelCallRecord, ...] = ()


@dataclass(frozen=True)
class RestoredContext:
    """保存候选会话通过 Token 检查后可正式启用的上下文。"""

    messages: tuple[ChatMessage, ...]
    checkpoint: str | None
    compactions: int


class ContextManager:
    """管理一个 Agent 会话的两层上下文压缩状态。

    AgentLoop 在提交工具结果前调用 ``compact_tool_results``，每次普通请求
    前调用 ``should_auto_compact``，完成后调用 ``record_normal_usage``。
    手动和紧急摘要也走同一个 ``compact``，因此共享重试与熔断状态。
    """

    def __init__(
        self,
        request_runner: ProviderRequestRunner,
        conversation: Conversation,
        config: ProviderConfig,
        artifact_store: ArtifactStore,
    ) -> None:
        # 把摘要请求发给模型服务，收完整个响应，最后返回模型的完成结果
        self._runner = request_runner
        # AgentLoop 使用的对话记录；摘要成功后，删除已被摘要替代的旧消息，只保留近期原文
        self._conversation = conversation
        # 窗口、摘要输出及工具阈值来自当前 Provider 配置。
        self._config = config
        # 负责用户原文暂存、正式提交和会话文件清理。
        self._artifacts = artifact_store
        # 负责按当前 Provider 的字符数上限保存过长的工具结果并生成预览。
        self._tool_results = ToolResultCompactor(
            artifact_store,
            result_threshold_chars=config.tool_result_spill_chars,
            batch_threshold_chars=config.tool_batch_spill_chars,
        )
        # 估算完整模型请求会占多少 Token，用来判断是否需要自动生成摘要
        self._estimator = TokenEstimator()
        # 把对话分成“需要摘要的较早消息”和“继续保留的近期消息”
        self._partitioner = HistoryPartitioner()
        # 负责生成摘要请求，并检查模型返回的摘要格式
        self._codec = SummaryCodec()
        # 最近一次成功生成的上下文摘要；还没有摘要时为 None。
        self._summary: str | None = None
        # 从上次摘要成功或清空对话后，摘要请求连续失败的次数
        self._consecutive_failures = 0
        # 普通压缩沿用固定近期窗口；仅候选会话的后续恢复轮次会临时调小。
        self._recent_group_floor = RECENT_MESSAGE_GROUPS
        self._recent_token_target = RECENT_MESSAGE_TOKENS

    @property
    def checkpoint_instructions(self) -> tuple[RuntimeInstruction, ...]:
        """返回后续模型请求要携带的上下文摘要和重新读取提醒

        还没有生成摘要时返回空元组
        """

        if self._summary is None:
            return ()
        return (
            RuntimeInstruction(
                RuntimeInstructionKind.COMPACTION_CHECKPOINT,
                self._summary,
            ),
            RuntimeInstruction(
                RuntimeInstructionKind.COMPACTION_BOUNDARY,
                _BOUNDARY_TEXT,
            ),
        )

    def compact_tool_results(
        self,
        results: tuple[ToolExecutionResult, ...],
    ) -> ToolResultCompactionOutcome:
        """返回同序工具结果，以及本批首次发生的 artifact 保存失败。"""

        return self._tool_results.compact_batch(results)

    def should_auto_compact(self, request: ProviderRequest) -> bool:
        """判断完整请求估算是否已达到自动压缩触发线。"""

        if self._consecutive_failures >= MAX_COMPACTION_ATTEMPTS:
            return False
        threshold = (
            self._config.context_window_tokens
            - self._config.compaction_output_tokens
            - AUTO_COMPACTION_MARGIN_TOKENS
        )
        return self._estimator.estimate_request(request) >= threshold

    def estimate_request(self, request: ProviderRequest) -> int:
        """估算一次完整模型请求占用的 Token 数量。

        Args:
            request: 已包含提示词、消息和工具定义的模型请求。

        Returns:
            按当前本地估算规则计算出的近似 Token 数量。
        """

        return self._estimator.estimate_request(request)

    def _user_material(
        self,
        mode: CompactionMode,
    ) -> tuple[tuple[UserMessage, ...], int, StagedArtifact | None]:
        """选择能放进摘要请求并原样写入摘要的近期用户消息

        如果全部用户消息放不下，就把完整用户原话写入临时文

        Returns:
            选中的近期用户消息、未选中的消息数量，以及完整原话临时文件；没有省略消息时，临时文件为 None。
        """

        # 取出历史消息中的所有用户消息
        all_users = tuple(
            message
            for message in self._conversation.history
            if isinstance(message, UserMessage)
        )

        # 根据模式选择安全余量
        margin = (
            AUTO_COMPACTION_MARGIN_TOKENS
            if mode is CompactionMode.AUTO
            else MANUAL_COMPACTION_MARGIN_TOKENS
        )

        # 计算摘要请求最多能使用多少输入 Token
        # 整个摘要请求的输入部分，最多允许占用多少 Token = 上下文窗口-摘要输出预留-安全余量
        input_budget = (
            self._config.context_window_tokens
            - self._config.compaction_output_tokens
            - margin
        )
        # 选中的用户消息既要作为摘要材料发送给模型
        # 又必须在摘要第六部分原样返回，因此同时受输入和输出空间限制
        # 两个约束取较小值，并在输出摘要中至少给草稿、标签和其他八部分留 4000 Token。
        budget = max(
            min(
                self._config.compaction_output_tokens - 4_000,
                input_budget,
            ),
            0,
        )
        # 暂时保存选中的用户消息（按时间从新到旧排列）
        selected_reversed: list[UserMessage] = []
        # 统计已选中的消息大约花了多少token了
        used = 0
        # 从新到旧遍历用户消息
        for message in reversed(all_users):
            # 估算当前文本的token数
            cost = estimate_text(message.content)
            if used + cost > budget:
                # 到达阈值停止
                break
            # 未到达阈值，继续追加用户消息
            selected_reversed.append(message)
            used += cost
        # 再把顺序调整回去，按从旧到新
        selected = tuple(reversed(selected_reversed))
        # 计算未被选中的消息个数
        omitted = len(all_users) - len(selected)
        # 将未被选中发给模型的用户原话落盘
        staged = (
            self._artifacts.write_user_transcript_temp_file(all_users)
            if omitted and all_users
            else None
        )
        return selected, omitted, staged

    def _shrink_groups(
        self,
        groups: tuple[MessageGroup, ...],
        percentage: int,
    ) -> tuple[MessageGroup, ...]:
        """从下一次摘要请求要发送的较早对话中，删除最早的一部分非用户消息组

        用户消息组始终保留。包含工具调用的助手消息及其工具结果会作为一个
        完整消息组一起保留或删除，不会从组中单独删除某条消息。删除数量按照
        非用户消息组总数和给定百分比计算，并向上取整。

        Args:
            groups: 按时间从旧到新排列的摘要消息组。
            percentage: 要删除的非用户消息组比例，例如 10 表示删除约 10%。

        Returns:
            删除后剩余的消息组，原有顺序保持不变。
        """
        # 统计不含用户消息的消息组数量
        non_user_count = sum(
            not group.contains_user_message for group in groups
        )
        # 需要去除的非用户消息组数量
        remove_count = math.ceil(non_user_count * percentage / 100)
        # 保存不需要删除的消息组
        remaining: list[MessageGroup] = []
        # 记录已经删除了多少组
        removed = 0
        for group in groups:
            if not group.contains_user_message and removed < remove_count:
                # 当前组如果不含用户消息，并且删除数量不够，就接着删除
                removed += 1
                continue
            remaining.append(group)
        return tuple(remaining)

    def _failure(
        self,
        exc: BaseException,
    ) -> CompactionOutcome:
        """记录一次摘要请求失败，并返回要交给 AgentLoop 的失败结果

        每次调用都会增加连续失败次数。达到最大失败次数时，返回停止自动重试的
        结果；否则返回普通失败结果，并替换异常信息中出现的 API Key

        Args:
            exc: 本次摘要请求抛出的异常。

        Returns:
            普通失败或停止自动重试的摘要结果，其中包含可显示给用户的失败说明。
        """

        self._consecutive_failures += 1
        if self._consecutive_failures >= MAX_COMPACTION_ATTEMPTS:
            return CompactionOutcome(
                CompactionOutcomeKind.CIRCUIT_OPEN,
                "上下文摘要连续失败 3 次，已停止自动重试",
            )
        message = redact_secrets(str(exc), (self._config.api_key,))
        return CompactionOutcome(
            CompactionOutcomeKind.FAILED,
            f"上下文摘要失败：{message}",
        )

    async def compact(
        self,
        mode: CompactionMode,
        cancellation: CancellationToken,
        *,
        retention_focus: str | None = None,
        model_call_budget: ModelCallBudget | None = None,
    ) -> CompactionOutcome:
        """调用模型摘要较早对话，成功后只保留消息和工具结果后重试。
        摘要失败或被取消时不修改原对话，并删除尚未提交的近期原文和新摘要。

        一次最多请求三次。上下文超限时逐步减少较早的助手用户原话临时文件。

        Args:
            mode: 摘要的触发方式，包括自动、手动和上下文超限后的紧急摘要。
            cancellation: 用于中止摘要请求的取消信号。
            retention_focus: 用户手动压缩时希望摘要额外保留的重点；其他模式忽略。
            model_call_budget: 当前 Agent 运行的共享模型调用预算；手动压缩和
                会话恢复不属于某条 Agent 请求，因此不传。

        Returns:
            摘要的最终状态和可以显示给用户的说明。
        """

        if (
            self._consecutive_failures >= MAX_COMPACTION_ATTEMPTS
            and mode is not CompactionMode.MANUAL
        ):
            return CompactionOutcome(
                CompactionOutcomeKind.CIRCUIT_OPEN,
                "上下文摘要熔断中；可使用 /compact 手动重试或 /clear 重置",
            )
        partition = self._partitioner.partition(
            self._conversation.history,
            minimum_recent_groups=self._recent_group_floor,
            recent_token_target=self._recent_token_target,
        )
        if not partition.compactable_groups:
            return CompactionOutcome(
                CompactionOutcomeKind.NO_CONTENT,
                "当前没有可压缩的较早对话",
            )

        selected_users, omitted_users, staged = self._user_material(mode)
        attempts = 0
        shrink_percentage = 0
        records: list[ModelCallRecord] = []
        try:
            while attempts < MAX_COMPACTION_ATTEMPTS:
                attempts += 1
                material = CompactionMaterial(
                    previous_summary=self._summary,
                    groups=self._shrink_groups(
                        partition.compactable_groups,
                        shrink_percentage,
                    ),
                    user_messages=selected_users,
                    omitted_user_messages=omitted_users,
                    user_transcript_path=(
                        staged.record.relative_path
                        if staged is not None
                        else None
                    ),
                    retention_focus=(
                        retention_focus.strip()
                        if mode is CompactionMode.MANUAL
                        and retention_focus is not None
                        and retention_focus.strip()
                        else None
                    ),
                )
                request = self._codec.build_request(
                    material,
                    max_output_tokens=self._config.compaction_output_tokens,
                )
                model_call_number: int | None = None
                if model_call_budget is not None:
                    try:
                        model_call_number = model_call_budget.begin(
                            ModelCallPurpose.COMPACTION,
                            preserve_final_call=True,
                        )
                    except RuntimeError:
                        return CompactionOutcome(
                            CompactionOutcomeKind.FINAL_CALL_RESERVED,
                            "只剩最后一次模型调用，已跳过上下文压缩并保留正式回答额度",
                            tuple(records),
                        )
                try:
                    completed = await self._runner.collect(request, cancellation)
                except ProviderRequestCancelled:
                    if model_call_number is not None:
                        records.append(
                            model_call_budget.finish(model_call_number, None)
                        )
                    return CompactionOutcome(
                        CompactionOutcomeKind.CANCELLED,
                        "上下文摘要已取消，原对话保持不变",
                        tuple(records),
                    )
                except ContextWindowExceededError as exc:
                    if model_call_number is not None:
                        records.append(
                            model_call_budget.finish(model_call_number, None)
                        )
                    failure = self._failure(exc)
                    if (
                        self._consecutive_failures >= MAX_COMPACTION_ATTEMPTS
                        or attempts >= MAX_COMPACTION_ATTEMPTS
                    ):
                        return replace(
                            failure,
                            model_call_records=tuple(records),
                        )
                    shrink_percentage = 10 if attempts == 1 else 20
                    continue
                except AuthenticationError as exc:
                    if model_call_number is not None:
                        records.append(
                            model_call_budget.finish(model_call_number, None)
                        )
                    return replace(
                        self._failure(exc),
                        model_call_records=tuple(records),
                    )
                except TransportError as exc:
                    if model_call_number is not None:
                        records.append(
                            model_call_budget.finish(model_call_number, None)
                        )
                    failure = self._failure(exc)
                    if (
                        self._consecutive_failures >= MAX_COMPACTION_ATTEMPTS
                        or attempts >= MAX_COMPACTION_ATTEMPTS
                    ):
                        return replace(
                            failure,
                            model_call_records=tuple(records),
                        )
                    # 网络或服务暂时失败时复用当前缩减级别，不能因为连接
                    # 波动继续丢弃更多较早材料。
                    continue
                except MyCodeError as exc:
                    if model_call_number is not None:
                        records.append(
                            model_call_budget.finish(model_call_number, None)
                        )
                    return replace(
                        self._failure(exc),
                        model_call_records=tuple(records),
                    )
                except Exception as exc:
                    if model_call_number is not None:
                        records.append(
                            model_call_budget.finish(model_call_number, None)
                        )
                    failure = self._failure(exc)
                    if (
                        self._consecutive_failures >= MAX_COMPACTION_ATTEMPTS
                        or attempts >= MAX_COMPACTION_ATTEMPTS
                    ):
                        return replace(
                            failure,
                            model_call_records=tuple(records),
                        )
                    continue

                if model_call_number is not None:
                    records.append(
                        model_call_budget.finish(
                            model_call_number,
                            completed.usage,
                        )
                    )
                try:
                    summary = self._codec.parse(completed)
                except MyCodeError as exc:
                    return replace(
                        self._failure(exc),
                        model_call_records=tuple(records),
                    )

                if staged is not None:
                    self._artifacts.commit_staged(staged)
                    staged = None
                self._conversation.replace(
                    flatten_groups(partition.recent_groups)
                )
                self._summary = summary
                self._estimator.reset()
                self._consecutive_failures = 0
                return CompactionOutcome(
                    CompactionOutcomeKind.SUCCEEDED,
                    "较早对话已压缩，近期原文已保留",
                    tuple(records),
                )
            return CompactionOutcome(
                CompactionOutcomeKind.FAILED,
                "上下文摘要未完成",
                tuple(records),
            )
        finally:
            if staged is not None:
                self._artifacts.discard_staged(staged)

    def record_normal_usage(
        self,
        request: ProviderRequest,
        usage: ProviderUsage | None,
    ) -> None:
        """记录普通 Agent 请求的 usage；摘要请求不会调用此方法。"""

        self._estimator.record_usage(request, usage)

    @staticmethod
    def _restore_request(
        request: ProviderRequest,
        checkpoint: tuple[RuntimeInstruction, ...],
    ) -> ProviderRequest:
        """把候选摘要放入本次估算请求，不改动调用方构造的请求。"""

        prompt = request.prompt
        return ProviderRequest(
            messages=request.messages,
            tools=request.tools,
            tool_choice=request.tool_choice,
            prompt=PromptContext(
                stable=prompt.stable,
                runtime=(*checkpoint, *prompt.runtime),
            ),
            max_output_tokens=request.max_output_tokens,
        )

    async def prepare_restored_context(
        self,
        messages: tuple[ChatMessage, ...],
        cancellation: CancellationToken,
        build_request: Callable[
            [tuple[ChatMessage, ...]], ProviderRequest
        ],
        *,
        max_compactions: int = SESSION_RESTORE_MAX_COMPACTIONS,
    ) -> RestoredContext:
        """检查并按需压缩候选会话，不修改当前对话和当前摘要。"""

        if (
            isinstance(max_compactions, bool)
            or not isinstance(max_compactions, int)
            or not 1 <= max_compactions <= SESSION_RESTORE_MAX_COMPACTIONS
        ):
            raise ValueError("恢复压缩轮数必须是 1 到 3 的整数")

        candidate = Conversation()
        candidate.extend(messages)
        scratch = ContextManager(
            self._runner,
            candidate,
            self._config,
            self._artifacts,
        )
        compactions = 0

        while True:
            request = self._restore_request(
                build_request(candidate.history),
                scratch.checkpoint_instructions,
            )
            if not scratch.should_auto_compact(request):
                return RestoredContext(
                    messages=candidate.history,
                    checkpoint=scratch._summary,
                    compactions=compactions,
                )
            if compactions >= max_compactions:
                raise MyCodeError(
                    f"候选会话压缩 {max_compactions} 次后仍超过当前模型预算"
                )

            if compactions:
                scratch._recent_group_floor = max(
                    1,
                    RECENT_MESSAGE_GROUPS - compactions * 2,
                )
                scratch._recent_token_target = max(
                    0,
                    RECENT_MESSAGE_TOKENS // (2**compactions),
                )
            outcome = await scratch.compact(
                CompactionMode.AUTO,
                cancellation,
            )
            if outcome.kind is not CompactionOutcomeKind.SUCCEEDED:
                raise MyCodeError(f"无法恢复候选会话：{outcome.message}")
            compactions += 1

    def adopt_restored_context(self, checkpoint: str | None) -> None:
        """在会话正式切换后采用候选摘要并清空旧估算状态。"""

        self._summary = checkpoint
        self._consecutive_failures = 0
        self._estimator.reset()

    def reset(self) -> None:
        """清除摘要状态和当前会话的 artifact 文件。"""

        self._summary = None
        self._consecutive_failures = 0
        self._estimator.reset()
        self._artifacts.cleanup()

    def close(self) -> None:
        """在应用正常退出时删除当前会话 artifact。"""

        self._artifacts.cleanup()
