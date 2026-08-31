"""估算一次模型请求会占用多少 Token。

模型返回实际 Token 用量时，用实际值修正后续估算；没有实际用量时，根据
文本转换成 UTF-8 后的字节数进行估算。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from mycode.models.messages import (
    AssistantMessage,
    ChatMessage,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from mycode.models.provider import ProviderRequest, ProviderUsage
from mycode.models.tools import ToolDefinition

#  每次完整估算请求时固定增加一次，用来估算请求最外层字段和格式占用的 Token
_REQUEST_OVERHEAD = 24
# 每条用户消息、助手消息或工具结果固定增加一次，用来估算 role、content 等消息结构
_MESSAGE_OVERHEAD = 8
# 助手消息中的每个文字块、思考块或工具调用块固定增加一次，用来估算内容块的外层结构
_BLOCK_OVERHEAD = 4
# 每个工具定义固定增加一次，用来估算工具定义的外层结构；工具名、说明和参数 Schema 会另外计算
_TOOL_OVERHEAD = 12
# 每条临时加入提示词的环境信息、模式要求或对话摘要额外预留的 Token
_RUNTIME_OVERHEAD = 4


def estimate_text(text: str) -> int:
    """按 UTF-8 字节数除以 3 向上取整，估算一段文本的 Token。"""

    return math.ceil(len(text.encode("utf-8")) / 3)


def _json_tokens(value: object) -> int:
    """估算一份紧凑 JSON 数据的 Token。"""

    rendered = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return estimate_text(rendered)


def estimate_message(message: ChatMessage) -> int:
    """估算一条协议中立消息及其内容块的 Token。"""

    total = _MESSAGE_OVERHEAD
    if isinstance(message, UserMessage):
        return total + estimate_text(message.content)
    if isinstance(message, ToolResultMessage):
        return (
            total
            + estimate_text(message.tool_call_id)
            + estimate_text(message.tool_name)
            + estimate_text(message.content)
        )
    if isinstance(message, AssistantMessage):
        for block in message.content:
            total += _BLOCK_OVERHEAD
            if isinstance(block, TextBlock):
                total += estimate_text(block.text)
            elif isinstance(block, ThinkingBlock):
                total += estimate_text(block.thinking)
                total += estimate_text(block.signature)
            elif isinstance(block, RedactedThinkingBlock):
                total += estimate_text(block.data)
            elif isinstance(block, ToolCall):
                total += estimate_text(block.id) + estimate_text(block.name)
                total += _json_tokens(block.arguments)
        return total
    raise TypeError(f"不支持的消息类型：{type(message).__name__}")


def _tool_tokens(tool: ToolDefinition) -> int:
    """估算一个工具名称、描述和 JSON Schema 的 Token。"""

    return (
        _TOOL_OVERHEAD
        + estimate_text(tool.name)
        + estimate_text(tool.description)
        + _json_tokens(tool.input_schema)
    )


@dataclass(frozen=True)
class RequestSnapshot:
    """保存一次普通请求中可用于判断“只追加消息”的稳定部分。

    ``runtime`` 已渲染成字符串，避免调用方后来修改运行时对象；``messages``
    和 ``tools`` 都是不可变领域对象，可直接比较完整前缀。
    """

    # 请求中的稳定系统提示词。
    stable_prompt: str
    # 本次运行时指令渲染后的文本和顺序。
    runtime: tuple[str, ...]
    # 请求发送的真实消息快照。
    messages: tuple[ChatMessage, ...]
    # 请求可见的完整工具定义快照。
    tools: tuple[ToolDefinition, ...]


@dataclass(frozen=True)
class UsageAnchor:
    """记录最近一次普通请求的真实输入量和对应请求快照。"""

    # Provider 返回 usage 时对应的普通请求。
    request: RequestSnapshot
    # 该普通请求由 Provider 统计的真实输入 Token。
    input_tokens: int


class TokenEstimator:
    """估算完整 Provider 请求，并用最近 usage 只估算新增消息。

    AgentLoop 在每次普通请求前调用 ``estimate_request``，完成响应后调用
    ``record_usage``。只要 Prompt、运行时指令、工具定义或旧消息发生变化，
    就会放弃增量路径并重新估算完整请求。
    """

    def __init__(self) -> None:
        # 最近一次带合法 usage 的普通请求；摘要请求不会写入这里。
        self._anchor: UsageAnchor | None = None

    def _snapshot(self, request: ProviderRequest) -> RequestSnapshot:
        """把 ProviderRequest 转成可比较且不含协议细节的快照。"""

        prompt = request.prompt
        if isinstance(prompt, str):
            stable = prompt
            runtime: tuple[str, ...] = ()
        else:
            stable = prompt.stable
            runtime = tuple(item.render() for item in prompt.runtime)
        return RequestSnapshot(stable, runtime, request.messages, request.tools)

    def _estimate_full(self, snapshot: RequestSnapshot) -> int:
        """估算快照中的 Prompt、运行时、消息和工具定义。"""

        total = _REQUEST_OVERHEAD + estimate_text(snapshot.stable_prompt)
        total += sum(
            _RUNTIME_OVERHEAD + estimate_text(item)
            for item in snapshot.runtime
        )
        total += sum(estimate_message(message) for message in snapshot.messages)
        total += sum(_tool_tokens(tool) for tool in snapshot.tools)
        return total

    def estimate_request(self, request: ProviderRequest) -> int:
        """返回本次请求输入 Token 的近似值。"""

        snapshot = self._snapshot(request)
        anchor = self._anchor
        if anchor is not None:
            old = anchor.request
            prefix_length = len(old.messages)
            if (
                old.stable_prompt == snapshot.stable_prompt
                and old.runtime == snapshot.runtime
                and old.tools == snapshot.tools
                and snapshot.messages[:prefix_length] == old.messages
            ):
                appended = snapshot.messages[prefix_length:]
                return anchor.input_tokens + sum(
                    estimate_message(message) for message in appended
                )
        return self._estimate_full(snapshot)

    def record_usage(
        self,
        request: ProviderRequest,
        usage: ProviderUsage | None,
    ) -> None:
        """把一次已完成普通请求的真实输入量保存为后续估算锚点。"""

        if usage is None:
            return
        self._anchor = UsageAnchor(self._snapshot(request), usage.input_tokens)

    def reset(self) -> None:
        """清除旧请求锚点，下一次请求将执行完整估算。"""

        self._anchor = None
