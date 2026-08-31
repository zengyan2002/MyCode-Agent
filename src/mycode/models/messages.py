"""对话历史和 Provider 请求共用的消息模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from mycode.models.json_types import JsonObject


@dataclass(frozen=True)
class UserMessage:
    content: str


@dataclass(frozen=True)
class TextBlock:
    text: str


@dataclass(frozen=True)
class ThinkingBlock:
    """主要为Anthropic准备 要求原样回传的不透明签名思考内容。"""

    # Anthropic 返回的思考文本
    thinking: str
    # Anthropic 为这段思考提供的签名；程序不解析，后续请求原样回传
    signature: str


@dataclass(frozen=True)
class RedactedThinkingBlock:
    """Anthropic 生成的脱敏思考内容；本地不得解释或修改。"""

    # Anthropic 返回的不透明数据；本地不解析、不修改，也不展示
    data: str

# 工具参数必须是 JSON 对象；字符串分片的拼接和解析由具体 Provider 完成，
# 进入统一消息模型时已经是完整、可供本地 Schema 校验的结构。
@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: JsonObject


# 保留内容块的原始顺序十分关键：Anthropic 要求 thinking、文本和工具
# 调用按服务端生成顺序回传，不能拆成独立字段后再任意重排。
AssistantBlock: TypeAlias = (
    TextBlock | ThinkingBlock | RedactedThinkingBlock | ToolCall
)


@dataclass(frozen=True)
class AssistantMessage:
    content: tuple[AssistantBlock, ...]

    @property
    def text(self) -> str:
        """拼接可展示文本，不暴露思考内容或工具参数。"""

        return "".join(
            block.text for block in self.content if isinstance(block, TextBlock)
        )

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        # Agent 调度器只关心工具块，但完整 content 仍会进入历史，确保文本、
        # thinking 签名和工具调用在下一次 Provider 请求中都能正确重建。
        return tuple(
            block for block in self.content if isinstance(block, ToolCall)
        )


@dataclass(frozen=True)
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool


# 统一历史刻意不包含 OpenAI/Anthropic 的 role 字典。协议转换只发生在
# Provider 边界，Agent 与工具层因此可以共享同一套消息状态机。
ChatMessage: TypeAlias = UserMessage | AssistantMessage | ToolResultMessage
