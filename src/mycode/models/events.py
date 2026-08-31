"""Agent 运行选项和对外过程事件。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from mycode.constants import DEFAULT_MAX_READ_CONCURRENCY, DEFAULT_MAX_MODEL_CALLS
from mycode.models.model_calls import ModelCallPurpose
from mycode.models.provider import ProviderUsage
from mycode.models.tools import ToolExecutionResult, ToolInvocation

@dataclass(frozen=True)
class AgentRunOptions:
    """保存 Agent 处理一条用户消息时使用的执行限制。

    ``AgentLoop`` 和独立子 Agent 运行器把该对象交给共享执行循环。对象创建
    后不能修改，确保一次运行中的模式、模型调用上限、工具并发和超时不变。
    """

    # 是否开启仅规划模式；开启后可以读取和分析，但不能执行写工具。
    plan_only: bool = False
    # 处理一条用户消息时，最多向 Provider 发出多少次模型请求；工具调用不单独计数。
    max_model_calls: int = DEFAULT_MAX_MODEL_CALLS
    # 同一批只读工具最多同时执行几个；写工具始终按顺序执行。
    max_read_concurrency: int = DEFAULT_MAX_READ_CONCURRENCY
    # 整条用户请求最多运行多少秒；None表示不设置整体超时。
    overall_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        """拒绝无法作为运行限制使用的布尔值、非数值和非正数。

        Returns:
            所有运行选项都可直接交给 Agent 循环时不返回数据。

        Raises:
            ValueError: 模型调用上限、读工具并发数或可选超时时间无效。
        """

        if (
            isinstance(self.max_model_calls, bool)
            or not isinstance(self.max_model_calls, int)
            or self.max_model_calls <= 0
        ):
            raise ValueError("Agent 最大模型调用次数必须是正整数")
        if (
            isinstance(self.max_read_concurrency, bool)
            or not isinstance(self.max_read_concurrency, int)
            or self.max_read_concurrency <= 0
        ):
            raise ValueError("读工具并发上限必须是正整数")
        if (
            self.overall_timeout_seconds is not None
            and (
                isinstance(self.overall_timeout_seconds, bool)
                or not isinstance(self.overall_timeout_seconds, (int, float))
                or self.overall_timeout_seconds <= 0
            )
        ):
            raise ValueError("Agent 超时时间必须为正数")

# 对外错误码把不同内部异常归一化，UI 无需依赖具体异常类型。
class AgentErrorCode(str, Enum):
    PROVIDER_ERROR = "provider_error"
    PROTOCOL_ERROR = "protocol_error"
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    MAX_MODEL_CALLS = "max_model_calls"
    MODEL_OUTPUT_TRUNCATED = "model_output_truncated"
    INTERNAL_ERROR = "internal_error"


class AgentFinalizationProfile(str, Enum):
    """选择强制收尾请求应采用哪一种报告要求。"""

    MAIN = "main"
    EXPLORE = "explore"
    PLAN = "plan"
    VERIFICATION = "verification"
    GENERIC = "generic"


class AgentCompletionMode(str, Enum):
    """说明最终文本是正常结束产生的，还是预算耗尽前强制收尾产生的。"""

    NORMAL = "normal"
    FORCED_FINALIZATION = "forced_finalization"


class CompactionStatusKind(str, Enum):
    """终端需要展示的上下文压缩阶段。"""

    STARTED = "started"
    SUCCEEDED = "succeeded"
    NO_CONTENT = "no_content"
    FAILED = "failed"
    CIRCUIT_OPEN = "circuit_open"

# 用户消息事件只标记本轮开始，不代表消息已经提交到 Conversation。
@dataclass(frozen=True)
class UserMessageEvent:
    text: str

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("用户消息事件必须包含文本")

# thinking 只用于实时展示；需要回传的完整 thinking 块保存在消息模型中。
@dataclass(frozen=True)
class ThinkingDeltaEvent:
    # 这段思考内容属于当前运行的第几次模型调用。
    model_call_number: int
    # 思考文本内容。
    text: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.model_call_number, bool)
            or not isinstance(self.model_call_number, int)
            or self.model_call_number <= 0
            or not self.text
        ):
            raise ValueError("思考事件必须包含有效模型调用序号和文本")

# 模型每返回一段新的可见文字，就通过该事件交给 UI 立即追加显示；完整回答结束时另发 FinalReplyEvent
@dataclass(frozen=True)
class ModelTextDeltaEvent:
    model_call_number: int
    text: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.model_call_number, bool)
            or not isinstance(self.model_call_number, int)
            or self.model_call_number <= 0
            or not self.text
        ):
            raise ValueError("模型文本事件必须包含有效模型调用序号和文本")


@dataclass(frozen=True)
class ModelUsageEvent:
    """把一轮 Provider 完成时的实际 token 统计交给独立运行器。

    Attributes:
        model_call_number: 这份统计属于当前运行的第几次模型请求，从 1 开始。
        purpose: 这次请求用于 Agent 推理还是上下文压缩。
        usage: Provider 返回的统计；服务未提供可信统计时为 ``None``。
    """

    model_call_number: int
    usage: ProviderUsage | None
    purpose: ModelCallPurpose = ModelCallPurpose.AGENT

    def __post_init__(self) -> None:
        if (
            isinstance(self.model_call_number, bool)
            or not isinstance(self.model_call_number, int)
            or self.model_call_number <= 0
        ):
            raise ValueError("模型用量事件必须包含有效模型调用序号")
        if not isinstance(self.purpose, ModelCallPurpose):
            raise ValueError("模型用量事件 purpose 类型无效")
        if self.usage is not None and not isinstance(self.usage, ProviderUsage):
            raise ValueError("模型用量事件 usage 类型无效")


#工具开始事件
@dataclass(frozen=True)
class ToolStartedEvent:
    invocation: ToolInvocation

#工具结果事件
@dataclass(frozen=True)
class ToolResultEvent:
    invocation: ToolInvocation
    result: ToolExecutionResult

    def __post_init__(self) -> None:
        if (
            self.invocation.call.id != self.result.tool_call_id
            or self.invocation.call.name != self.result.tool_name
        ):
            raise ValueError("工具结果事件与对应调用不匹配")

@dataclass(frozen=True)
class FinalReplyEvent:
    """把已经可以展示和保存的最终正文交给主循环或独立运行器。

    Attributes:
        text: 模型主动完成或强制收尾后得到的可见正文；外层报告标记已移除。
        model_calls: 当前运行实际已经发出的 Provider 请求总数。
        completion_mode: 正常提前完成或使用最后一次额度强制收尾。
    """

    text: str
    model_calls: int
    completion_mode: AgentCompletionMode = AgentCompletionMode.NORMAL

    def __post_init__(self) -> None:
        if (
            not self.text
            or isinstance(self.model_calls, bool)
            or not isinstance(self.model_calls, int)
            or self.model_calls <= 0
        ):
            raise ValueError("最终回复必须包含文本和已完成模型调用次数")
        if not isinstance(self.completion_mode, AgentCompletionMode):
            raise ValueError("最终回复 completion_mode 类型无效")

@dataclass(frozen=True)
class AgentErrorEvent:
    """把终止当前运行的错误码、说明和实际模型调用数交给 UI。

    Attributes:
        code: UI 和独立运行器用来选择错误处理方式的枚举。
        message: 可以直接显示给用户或写入后台任务结果的中文说明。
        model_calls: 错误发生前实际发出的 Provider 请求数；请求前失败可为零。
    """

    code: AgentErrorCode
    message: str
    model_calls: int

    def __post_init__(self) -> None:
        if (
            not self.message
            or isinstance(self.model_calls, bool)
            or not isinstance(self.model_calls, int)
            or self.model_calls < 0
        ):
            raise ValueError("Agent 错误必须包含消息和有效模型调用次数")


@dataclass(frozen=True)
class AgentWarningEvent:
    """把不终止当前请求的警告交给应用显示。"""

    message: str

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("Agent 警告必须包含消息")


@dataclass(frozen=True)
class CompactionStatusEvent:
    """告诉应用压缩开始或结束，不携带摘要正文。"""

    # 压缩开始、成功、失败或熔断阶段。
    kind: CompactionStatusKind
    # UI 可直接展示且不包含摘要正文的说明。
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CompactionStatusKind):
            raise ValueError("压缩状态类型无效")
        if not self.message.strip():
            raise ValueError("压缩状态必须包含说明")


# AgentEvent 是应用层唯一需要消费的运行协议。工具的完整结果也通过
# 事件暴露，方便未来接入非终端消费者、审计器或更丰富的界面。
AgentEvent: TypeAlias = (
    UserMessageEvent
    | ThinkingDeltaEvent
    | ModelTextDeltaEvent
    | ModelUsageEvent
    | ToolStartedEvent
    | ToolResultEvent
    | FinalReplyEvent
    | AgentErrorEvent
    | AgentWarningEvent
    | CompactionStatusEvent
)
