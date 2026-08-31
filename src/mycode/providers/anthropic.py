"""Anthropic Messages 流式响应、思考内容与工具调用适配器。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from mycode.constants import (
    ANTHROPIC_API_VERSION,
    ANTHROPIC_MAX_TOKENS,
    ANTHROPIC_THINKING_BUDGET,
)
from mycode.errors import (
    ContextWindowExceededError,
    HttpServiceError,
    HttpStatusError,
    ServiceError,
    StreamProtocolError,
    redact_secrets,
)
from mycode.models.config import ProviderConfig, ThinkingMode
from mycode.models.messages import (
    AssistantBlock,
    AssistantMessage,
    ChatMessage,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from mycode.models.provider import (
    ModelStopReason,
    ProviderCompleted,
    ProviderEvent,
    ProviderRequest,
    ProviderTextDelta,
    ProviderThinkingDelta,
    ProviderUsage,
)
from mycode.models.tools import ToolDefinition
from mycode.providers.transport import HttpTransport


def _mapping(value: object, message: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StreamProtocolError(message)
    return dict(value)


@dataclass
class _AnthropicBlockBuilder:
    # Anthropic 以内容块为单位流式传输。一个 builder 只对应一个 block index，
    # 不同类型使用不同字段累积，结束后再转换成协议中立 AssistantBlock。
    block_type: str
    text_parts: list[str] = field(default_factory=list)
    thinking_parts: list[str] = field(default_factory=list)
    signature_parts: list[str] = field(default_factory=list)
    redacted_data: str = ""
    tool_id: str = ""
    tool_name: str = ""
    initial_input: dict[str, Any] = field(default_factory=dict)
    json_parts: list[str] = field(default_factory=list)


class AnthropicProvider:
    def __init__(
        self,
        config: ProviderConfig,
        transport: HttpTransport,
    ) -> None:
        self._config = config
        self._transport = transport

    def _assistant_block(self, block: AssistantBlock) -> dict[str, object]:
        # thinking 和 redacted_thinking 对本地是不透明数据，必须连同签名原样
        # 回传；不能只保留 UI 展示的摘要文本或自行改写内容。
        if isinstance(block, TextBlock):
            return {"type": "text", "text": block.text}
        if isinstance(block, ThinkingBlock):
            return {
                "type": "thinking",
                "thinking": block.thinking,
                "signature": block.signature,
            }
        if isinstance(block, RedactedThinkingBlock):
            return {"type": "redacted_thinking", "data": block.data}
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.arguments,
        }

    def _message(self, message: ChatMessage) -> dict[str, object]:
        # Anthropic 的工具结果使用 user role 下的 tool_result 内容块，而不是
        # OpenAI 的独立 tool role。协议差异到这里才展开。
        if isinstance(message, UserMessage):
            return {"role": "user", "content": message.content}
        if isinstance(message, ToolResultMessage):
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content,
                        "is_error": message.is_error,
                    }
                ],
            }
        return {
            "role": "assistant",
            "content": [
                self._assistant_block(block) for block in message.content
            ],
        }

    def _tool(self, definition: ToolDefinition) -> dict[str, object]:
        return {
            "name": definition.name,
            "description": definition.description,
            "input_schema": definition.input_schema,
            "strict": True,
        }

    def _request_body(
        self,
        request: ProviderRequest,
    ) -> dict[str, object]:
        # system 是顶层字段，不进入 messages。工具列表仅在 AUTO 阶段发送；
        # thinking 配置则属于 Provider 能力，与是否开放工具相互独立。
        system = [
            {
                "type": "text",
                "text": request.prompt.stable,
            },
            *(
                {"type": "text", "text": instruction.render()}
                for instruction in request.prompt.runtime
            ),
        ]
        body: dict[str, object] = {
            "model": request.model_override or self._config.model,
            "system": system,
            "messages": [
                self._message(message) for message in request.messages
            ],
            "stream": True,
            "max_tokens": (
                request.max_output_tokens
                if request.max_output_tokens is not None
                else ANTHROPIC_MAX_TOKENS
            ),
        }
        # NONE 表示最终回答阶段：不暴露工具定义，确保模型没有可继续
        # 调用的工具，并与 Conversation 的单工具回合约束保持一致。
        if request.tools and request.tool_choice.value == "auto":
            tools = [
                self._tool(definition) for definition in request.tools
            ]
            body["tools"] = tools
            body["tool_choice"] = {"type": request.tool_choice.value}
        if self._config.thinking is ThinkingMode.ENABLED:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": ANTHROPIC_THINKING_BUDGET,
                "display": "summarized",
            }
        elif self._config.thinking is ThinkingMode.ADAPTIVE:
            body["thinking"] = {
                "type": "adaptive",
                "display": "summarized",
            }
        return body

    def _stop_reason(self, raw_reason: str | None) -> ModelStopReason:
        mapping = {
            "end_turn": ModelStopReason.END_TURN,
            "tool_use": ModelStopReason.TOOL_USE,
            "max_tokens": ModelStopReason.MAX_TOKENS,
        }
        if raw_reason not in mapping:
            raise StreamProtocolError(
                f"Anthropic 返回不支持的 stop_reason：{raw_reason}"
            )
        return mapping[raw_reason]

    def _service_error(self, payload: Any) -> ServiceError:
        if isinstance(payload, Mapping):
            message = payload.get("message")
            if isinstance(message, str) and message:
                safe = redact_secrets(message, [self._config.api_key])
                lowered = message.casefold()
                if (
                    "prompt is too long" in lowered
                    or (
                        "context window" in lowered
                        and (
                            "too long" in lowered
                            or "exceed" in lowered
                            or "maximum" in lowered
                        )
                    )
                    or (
                        "input tokens" in lowered
                        and "maximum" in lowered
                        and ("exceed" in lowered or ">" in lowered)
                    )
                ):
                    return ContextWindowExceededError(
                        f"Anthropic 上下文长度超过限制：{safe}"
                    )
                return ServiceError(f"Anthropic 服务返回错误：{safe}")
        return ServiceError("Anthropic 服务返回流内错误")

    def _http_error(self, error: HttpStatusError) -> ServiceError:
        """解析 Anthropic HTTP 错误，并保留状态码供记忆任务判断是否重试。"""
        try:
            payload = json.loads(error.body)
        except json.JSONDecodeError:
            return HttpServiceError(
                error.status_code,
                f"Anthropic 服务请求失败（HTTP {error.status_code}）",
            )
        if isinstance(payload, Mapping) and "error" in payload:
            classified = self._service_error(payload["error"])
            if isinstance(classified, ContextWindowExceededError):
                return classified
            return HttpServiceError(error.status_code, str(classified))
        return HttpServiceError(
            error.status_code,
            f"Anthropic 服务请求失败（HTTP {error.status_code}）",
        )

    async def _transport_events(
        self,
        request: ProviderRequest,
        headers: Mapping[str, str],
    ) -> AsyncIterator[Any]:
        """读取 HTTP SSE，并在协议边界分类 Anthropic 错误。"""
        try:
            async for event in self._transport.stream_sse(
                url=self._config.base_url,
                headers=headers,
                json_body=self._request_body(request),
            ):
                yield event
        except HttpStatusError as exc:
            raise self._http_error(exc) from exc

    def _usage_value(self, raw: object, name: str) -> int | None:
        """读取一个可选非负 Anthropic usage 字段。

        字段缺失表示该类 Token 为零；字段存在但类型或数值非法时返回
        ``None``，调用方会丢弃整份 usage，不能把坏数据当作真实的零。
        """
        if not isinstance(raw, Mapping):
            return None
        value = raw.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    def _start_block(self, block: dict[str, Any]) -> _AnthropicBlockBuilder:
        # start 事件可能携带初始文本、完整工具 input 或空壳。先严格验证类型，
        # 后续 delta 才能依据 builder.block_type 判断事件是否合法。
        block_type = block.get("type")
        if block_type not in {
            "text",
            "thinking",
            "redacted_thinking",
            "tool_use",
        }:
            raise StreamProtocolError(f"未知 Anthropic 内容块：{block_type}")
        builder = _AnthropicBlockBuilder(block_type)
        if block_type == "text":
            text = block.get("text", "")
            if not isinstance(text, str):
                raise StreamProtocolError("Anthropic 文本块初始值必须是字符串")
            if text:
                builder.text_parts.append(text)
        elif block_type == "thinking":
            thinking = block.get("thinking", "")
            signature = block.get("signature", "")
            if not isinstance(thinking, str) or not isinstance(signature, str):
                raise StreamProtocolError("Anthropic 思考块初始值无效")
            if thinking:
                builder.thinking_parts.append(thinking)
            if signature:
                builder.signature_parts.append(signature)
        elif block_type == "redacted_thinking":
            data = block.get("data")
            if not isinstance(data, str):
                raise StreamProtocolError("Anthropic 脱敏思考块数据无效")
            builder.redacted_data = data
        else:
            tool_id = block.get("id")
            name = block.get("name")
            initial_input = block.get("input", {})
            if (
                not isinstance(tool_id, str)
                or not isinstance(name, str)
                or not isinstance(initial_input, Mapping)
            ):
                raise StreamProtocolError("Anthropic tool_use 初始内容无效")
            builder.tool_id = tool_id
            builder.tool_name = name
            builder.initial_input = dict(initial_input)
        return builder

    def _apply_delta(
        self,
        builder: _AnthropicBlockBuilder,
        delta: dict[str, Any],
    ) -> ProviderEvent | None:
        # 只有可展示的 text/thinking 增量向上产生事件；signature 和工具 JSON
        # 仍需累积，但不应暴露给终端或在尚未完整时交给 Agent。
        delta_type = delta.get("type")
        if delta_type == "text_delta" and builder.block_type == "text":
            text = delta.get("text")
            if not isinstance(text, str):
                raise StreamProtocolError("Anthropic 文本增量必须是字符串")
            if text:
                builder.text_parts.append(text)
                return ProviderTextDelta(text)
            return None
        if delta_type == "thinking_delta" and builder.block_type == "thinking":
            thinking = delta.get("thinking")
            if not isinstance(thinking, str):
                raise StreamProtocolError("Anthropic 思考增量必须是字符串")
            if thinking:
                builder.thinking_parts.append(thinking)
                return ProviderThinkingDelta(thinking)
            return None
        if delta_type == "signature_delta" and builder.block_type == "thinking":
            signature = delta.get("signature")
            if not isinstance(signature, str):
                raise StreamProtocolError("Anthropic 签名增量必须是字符串")
            builder.signature_parts.append(signature)
            return None
        if (
            delta_type == "input_json_delta"
            and builder.block_type == "tool_use"
        ):
            partial = delta.get("partial_json")
            if not isinstance(partial, str):
                raise StreamProtocolError("Anthropic 工具参数片段必须是字符串")
            builder.json_parts.append(partial)
            return None
        raise StreamProtocolError(
            f"Anthropic 增量 {delta_type} 与内容块类型不一致"
        )

    def _finish_block(
        self,
        builder: _AnthropicBlockBuilder,
    ) -> AssistantBlock:
        # block_stop 是解析工具参数的最早安全时机。完整 input 与分片 JSON
        # 互斥；同时出现说明兼容服务给出了语义冲突的响应。
        if builder.block_type == "text":
            return TextBlock("".join(builder.text_parts))
        if builder.block_type == "thinking":
            return ThinkingBlock(
                "".join(builder.thinking_parts),
                "".join(builder.signature_parts),
            )
        if builder.block_type == "redacted_thinking":
            return RedactedThinkingBlock(builder.redacted_data)

        if builder.json_parts and builder.initial_input:
            raise StreamProtocolError(
                "Anthropic tool_use 同时包含完整和分片参数"
            )
        if builder.json_parts:
            try:
                arguments = json.loads("".join(builder.json_parts))
            except json.JSONDecodeError as exc:
                raise StreamProtocolError(
                    f"Anthropic 工具 {builder.tool_name} 的参数不是完整 JSON"
                ) from exc
        else:
            arguments = builder.initial_input
        if not builder.tool_id or not builder.tool_name:
            raise StreamProtocolError("Anthropic tool_use 缺少 ID 或名称")
        if not isinstance(arguments, dict):
            raise StreamProtocolError("Anthropic 工具参数必须是 JSON 对象")
        return ToolCall(builder.tool_id, builder.tool_name, arguments)

    async def stream(
        self,
        request: ProviderRequest,
    ) -> AsyncIterator[ProviderEvent]:
        headers = {
            "x-api-key": self._config.api_key.reveal(),
            "anthropic-version": ANTHROPIC_API_VERSION,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        # builders 保存已开始的块，stopped 单独记录已结束索引。二者组合可
        # 检测重复 start/stop、停止后继续 delta，以及消息过早结束。
        builders: dict[int, _AnthropicBlockBuilder] = {}
        stopped: set[int] = set()
        stop_reason: str | None = None
        input_tokens: int | None = None
        cached_input_tokens: int | None = None
        output_tokens: int | None = None
        completed = False
        async for event in self._transport_events(request, headers):
            try:
                payload = json.loads(event.data)
            except json.JSONDecodeError as exc:
                raise StreamProtocolError("Anthropic 流包含无效 JSON") from exc
            payload = _mapping(payload, "Anthropic 流事件必须是 JSON 对象")
            event_type = payload.get("type")
            if not isinstance(event_type, str):
                raise StreamProtocolError("Anthropic 流事件缺少 type")
            if event.event and event.event != event_type:
                # Anthropic 同时在 SSE event 字段和 JSON type 中声明事件名；
                # 两者冲突通常意味着代理改写或流解析错位，不能择一猜测。
                raise StreamProtocolError("Anthropic SSE event 与 JSON type 不一致")
            if completed and event_type != "ping":
                raise StreamProtocolError("Anthropic 流在 message_stop 后仍包含数据")

            if event_type == "ping":
                continue
            if event_type == "message_start":
                message = payload.get("message")
                if not isinstance(message, Mapping):
                    raise StreamProtocolError("Anthropic message_start 无效")
                raw_usage = message.get("usage")
                if isinstance(raw_usage, Mapping):
                    cached_input_tokens = self._usage_value(
                        raw_usage,
                        "cache_read_input_tokens",
                    )
                    values = tuple(
                        self._usage_value(raw_usage, name)
                        for name in (
                            "input_tokens",
                            "cache_creation_input_tokens",
                            "cache_read_input_tokens",
                        )
                    )
                    if all(value is not None for value in values):
                        input_tokens = sum(
                            value for value in values if value is not None
                        )
                continue
            if event_type == "error":
                raise self._service_error(payload.get("error"))
            if event_type == "content_block_start":
                index = payload.get("index")
                block = payload.get("content_block")
                if not isinstance(index, int) or not isinstance(block, Mapping):
                    raise StreamProtocolError("Anthropic 内容块开始事件无效")
                if index in builders or index != len(builders):
                    # 要求索引从 0 连续增长，使最终按 range 重建内容块时不会
                    # 静默丢失块，也能尽早发现服务端乱序。
                    raise StreamProtocolError("Anthropic 内容块索引重复或不连续")
                builders[index] = self._start_block(dict(block))
                continue
            if event_type == "content_block_delta":
                index = payload.get("index")
                delta = payload.get("delta")
                if (
                    not isinstance(index, int)
                    or index not in builders
                    or index in stopped
                    or not isinstance(delta, Mapping)
                ):
                    raise StreamProtocolError("Anthropic 内容块增量顺序无效")
                display_event = self._apply_delta(
                    builders[index],
                    dict(delta),
                )
                if display_event is not None:
                    yield display_event
                continue
            if event_type == "content_block_stop":
                index = payload.get("index")
                if (
                    not isinstance(index, int)
                    or index not in builders
                    or index in stopped
                ):
                    raise StreamProtocolError("Anthropic 内容块结束顺序无效")
                # 在内容块结束时立即解析工具 JSON；Provider 一旦声明该块完整，
                # 就应立刻拒绝格式错误的参数，而不是拖到整条消息结束。
                self._finish_block(builders[index])
                stopped.add(index)
                continue
            if event_type == "message_delta":
                # stop_reason 属于消息级信息，可能晚于全部内容块到达。重复
                # 声明允许，但前后值必须一致。
                delta = payload.get("delta")
                if not isinstance(delta, Mapping):
                    raise StreamProtocolError("Anthropic message_delta 无效")
                current_reason = delta.get("stop_reason")
                if current_reason is not None:
                    if not isinstance(current_reason, str):
                        raise StreamProtocolError(
                            "Anthropic stop_reason 必须是字符串"
                        )
                    if stop_reason is not None and stop_reason != current_reason:
                        raise StreamProtocolError("Anthropic stop_reason 前后冲突")
                    stop_reason = current_reason
                raw_usage = payload.get("usage")
                if isinstance(raw_usage, Mapping):
                    output_tokens = self._usage_value(
                        raw_usage,
                        "output_tokens",
                    )
                continue
            if event_type == "message_stop":
                if completed:
                    raise StreamProtocolError("Anthropic 流包含重复 message_stop")
                if set(builders) != stopped:
                    raise StreamProtocolError("Anthropic 流在内容块结束前停止")
                normalized_reason = self._stop_reason(stop_reason)
                blocks = tuple(
                    self._finish_block(builders[index])
                    for index in range(len(builders))
                    if (
                        # MAX_TOKENS 可能截断 tool_use 参数。保留可展示文本和
                        # thinking，但绝不把潜在半成品工具调用交给 Agent 执行。
                        normalized_reason is not ModelStopReason.MAX_TOKENS
                        or builders[index].block_type != "tool_use"
                    )
                )
                completed = True
                yield ProviderCompleted(
                    stop_reason=normalized_reason,
                    assistant_message=AssistantMessage(blocks),
                    usage=(
                        ProviderUsage(
                            input_tokens,
                            output_tokens,
                            cached_input_tokens,
                        )
                        if input_tokens is not None
                        and output_tokens is not None
                        else None
                    ),
                )
                continue
            raise StreamProtocolError(f"未知 Anthropic 流事件：{event_type}")

        if not completed:
            raise StreamProtocolError("Anthropic 流在收到 message_stop 前中断")
