"""Provider 请求、流事件和标准化结束原因。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from mycode.models.messages import AssistantMessage, ChatMessage
from mycode.models.prompts import PromptContext
from mycode.models.tools import ToolDefinition


class ToolChoice(str, Enum):
    """模型工具选择模式。"""

    AUTO = "auto"
    NONE = "none"


@dataclass(frozen=True)
class ProviderUsage:
    """一次成功模型请求实际消耗的 Token 数。

    Provider 适配器把协议专有统计转换成该对象。ContextManager 只使用
    ``input_tokens`` 作为下一次普通请求的估算锚点。
    """

    # Provider 统计的本次请求输入量，TokenEstimator 把它保存为增量锚点。
    input_tokens: int
    # Provider 统计的本次响应输出量，目前用于诊断，不参与输入估算。
    output_tokens: int
    # Provider 明确报告的缓存命中输入量；未提供该统计时为 None，报告零时保留 0。
    cached_input_tokens: int | None = None

    def __post_init__(self) -> None:
        for value in (self.input_tokens, self.output_tokens):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Provider usage 必须是非负整数")
        if self.cached_input_tokens is not None and (
            isinstance(self.cached_input_tokens, bool)
            or not isinstance(self.cached_input_tokens, int)
            or self.cached_input_tokens < 0
        ):
            raise ValueError("Provider cached usage 必须是非负整数或 None")


@dataclass(frozen=True)
class ProviderRequest:
    # 发送给模型的对话消息，包括：以前的用户问题、以前的模型回答、工具调用和工具结果、当前用户的新问题
    messages: tuple[ChatMessage, ...]
    # 模型当前可以使用哪些工具，以及每个工具需要什么参数
    tools: tuple[ToolDefinition, ...]
    # 控制这次请求是否允许模型调用工具
    tool_choice: ToolChoice
    # 发给模型的系统提示词和运行时说明
    prompt: PromptContext | str
    # 模型本次最多可以输出多少 Token；None 表示使用模型服务的默认上限
    max_output_tokens: int | None = None
    # fork Skill 可为这一次请求指定模型；None 表示沿用 Provider 全局配置。
    model_override: str | None = None

    def __post_init__(self) -> None:
        # 接受旧调用方传入的纯字符串作为平滑迁移路径；进入 Provider 前统一
        # 转成没有运行时补充的 PromptContext。
        if isinstance(self.prompt, str):
            object.__setattr__(self, "prompt", PromptContext(self.prompt))
        if not isinstance(self.prompt, PromptContext):
            raise ValueError("Provider 请求必须包含提示上下文")
        if (
            self.max_output_tokens is not None
            and (
                isinstance(self.max_output_tokens, bool)
                or not isinstance(self.max_output_tokens, int)
                or self.max_output_tokens <= 0
            )
        ):
            raise ValueError("Provider 输出 Token 上限必须是正整数")
        if (
            self.model_override is not None
            and not self.model_override.strip()
        ):
            raise ValueError("Provider 临时模型名称不能为空")

class ModelStopReason(str, Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"


@dataclass(frozen=True)
class ProviderThinkingDelta:
    text: str

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("Provider 思考增量必须包含文本")


@dataclass(frozen=True)
class ProviderTextDelta:
    text: str

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("Provider 文本增量必须包含文本")


@dataclass(frozen=True)
class ProviderCompleted:
    # 结束原因
    stop_reason: ModelStopReason
    # Provider 把流式片段拼成的助手消息，可能包含文字、思考内容或工具调用
    assistant_message: AssistantMessage
    # 服务未提供可信统计时为 None；这不影响已经完成的模型回答。
    usage: ProviderUsage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stop_reason, ModelStopReason):
            raise ValueError("Provider 完成事件必须包含结束原因")
        if not isinstance(self.assistant_message, AssistantMessage):
            raise ValueError("Provider 完成事件必须包含助手消息")
        if self.usage is not None and not isinstance(self.usage, ProviderUsage):
            raise ValueError("Provider 完成事件 usage 类型无效")


# Provider 只向 Agent 暴露两类可实时展示的增量和一个唯一完成事件；
# 协议专有的 SSE 事件名称、索引和 JSON 结构不会泄漏到上层。
ProviderEvent: TypeAlias = (
    ProviderThinkingDelta | ProviderTextDelta | ProviderCompleted
)
