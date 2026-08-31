"""OpenAI Chat Completions 流式响应与工具调用适配器。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from mycode.errors import (
    ContextWindowExceededError,
    HttpServiceError,
    HttpStatusError,
    ServiceError,
    StreamProtocolError,
    redact_secrets,
)
from mycode.models.config import ProviderConfig
from mycode.models.messages import (
    AssistantMessage,
    ChatMessage,
    TextBlock,
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
    ProviderUsage,
)
from mycode.models.tools import ToolDefinition
from mycode.providers.transport import HttpTransport


def _json_object(value: object, message: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StreamProtocolError(message)
    return dict(value)


@dataclass
class _OpenAIToolBuilder:
    # Chat Completions 会把一次工具调用拆成多个 delta。builder 按调用索引
    # 暂存元数据和参数片段，只有收到 [DONE] 后才生成不可变 ToolCall。
    call_id: str | None = None
    name: str | None = None
    argument_parts: list[str] = field(default_factory=list)
    saw_function_type: bool = False


class OpenAIProvider:
    def __init__(
        self,
        config: ProviderConfig,
        transport: HttpTransport,
    ) -> None:
        self._config = config
        self._transport = transport

    def _message(self, message: ChatMessage) -> dict[str, object]:
        # 统一历史在这里才获得 OpenAI role。ToolResultMessage 必须使用
        # tool_call_id 与先前 assistant.tool_calls 精确配对。
        if isinstance(message, UserMessage):
            return {"role": "user", "content": message.content}
        if isinstance(message, ToolResultMessage):
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }

        payload: dict[str, object] = {
            "role": "assistant",
            "content": message.text or None,
        }
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    },
                }
                for call in message.tool_calls
            ]
        return payload

    def _tool(self, definition: ToolDefinition) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.input_schema,
                "strict": True,
            },
        }

    def _request_body(
        self,
        request: ProviderRequest,
    ) -> dict[str, object]:
        request_messages = [
            {"role": "system", "content": request.prompt.stable},
            *(
                {"role": "system", "content": instruction.render()}
                for instruction in request.prompt.runtime
            ),
            *(self._message(message) for message in request.messages),
        ]
        body: dict[str, object] = {
            "model": request.model_override or self._config.model,
            "messages": request_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.max_output_tokens is not None:
            body["max_tokens"] = request.max_output_tokens
        # 最终回答阶段必须彻底移除工具，而不只是依赖
        # tool_choice="none"；部分 OpenAI 兼容接口会忽略该字段。
        if request.tools and request.tool_choice.value == "auto":
            body["tools"] = [
                self._tool(definition) for definition in request.tools
            ]
            body["tool_choice"] = request.tool_choice.value
        return body

    def _stop_reason(self, raw_reason: str | None) -> ModelStopReason:
        mapping = {
            "stop": ModelStopReason.END_TURN,
            "tool_calls": ModelStopReason.TOOL_USE,
            "length": ModelStopReason.MAX_TOKENS,
        }
        if raw_reason not in mapping:
            raise StreamProtocolError(
                f"OpenAI 返回不支持的 finish_reason：{raw_reason}"
            )
        return mapping[raw_reason]

    def _service_error(self, payload: Any) -> ServiceError:
        if isinstance(payload, Mapping):
            message = payload.get("message")
            code = payload.get("code")
            if isinstance(message, str) and message:
                safe = redact_secrets(message, [self._config.api_key])
                lowered = message.casefold()
                if code == "context_length_exceeded" or (
                    "context length" in lowered
                    and ("exceed" in lowered or "maximum" in lowered)
                ):
                    return ContextWindowExceededError(
                        f"OpenAI 上下文长度超过限制：{safe}"
                    )
                return ServiceError(f"OpenAI 服务返回错误：{safe}")
        return ServiceError("OpenAI 服务返回流内错误")

    def _http_error(self, error: HttpStatusError) -> ServiceError:
        """解析 OpenAI HTTP 错误，并保留状态码供记忆任务判断是否重试。"""
        try:
            payload = json.loads(error.body)
        except json.JSONDecodeError:
            return HttpServiceError(
                error.status_code,
                f"OpenAI 服务请求失败（HTTP {error.status_code}）",
            )
        if isinstance(payload, Mapping) and "error" in payload:
            classified = self._service_error(payload["error"])
            if isinstance(classified, ContextWindowExceededError):
                return classified
            return HttpServiceError(error.status_code, str(classified))
        return HttpServiceError(
            error.status_code,
            f"OpenAI 服务请求失败（HTTP {error.status_code}）",
        )

    async def _transport_events(
        self,
        request: ProviderRequest,
        headers: Mapping[str, str],
    ) -> AsyncIterator[Any]:
        """读取 HTTP SSE，并在协议边界分类 OpenAI 错误。"""
        try:
            async for event in self._transport.stream_sse(
                url=self._config.base_url,
                headers=headers,
                json_body=self._request_body(request),
            ):
                yield event
        except HttpStatusError as exc:
            raise self._http_error(exc) from exc

    def _usage(self, raw: object) -> ProviderUsage | None:
        """把 OpenAI usage 对象转换成统一统计。

        Args:
            raw: usage-only 流事件中的原始 JSON 值。

        Returns:
            输入和输出统计合法时返回 ProviderUsage；基础字段缺失或无效时
            返回 ``None``。缓存明细缺失或无效只会让该字段为 ``None``。
        """
        if not isinstance(raw, Mapping):
            return None
        input_tokens = raw.get("prompt_tokens")
        output_tokens = raw.get("completion_tokens")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in (input_tokens, output_tokens)
        ):
            return None
        cached_input_tokens: int | None = None
        details = raw.get("prompt_tokens_details")
        if isinstance(details, Mapping):
            cached = details.get("cached_tokens")
            if (
                not isinstance(cached, bool)
                and isinstance(cached, int)
                and cached >= 0
            ):
                cached_input_tokens = cached
        return ProviderUsage(
            input_tokens,
            output_tokens,
            cached_input_tokens,
        )

    def _consume_tool_deltas(
        self,
        raw_calls: object,
        builders: dict[int, _OpenAIToolBuilder],
    ) -> None:
        # index 是并行工具调用的稳定身份。服务端可以在一个 chunk 中同时
        # 推进多个 index，但新 index 必须连续出现，不能跳号或回填未知槽位。
        if not isinstance(raw_calls, list) or not raw_calls:
            raise StreamProtocolError("OpenAI tool_calls 增量必须是非空数组")
        for raw_call in raw_calls:
            call = _json_object(raw_call, "OpenAI 工具调用增量必须是对象")
            index = call.get("index")
            if not isinstance(index, int) or index < 0:
                raise StreamProtocolError("OpenAI 工具调用缺少有效 index")
            if index not in builders:
                if index != len(builders):
                    raise StreamProtocolError("OpenAI 工具调用索引不连续")
                builders[index] = _OpenAIToolBuilder()
            builder = builders[index]

            call_id = call.get("id")
            if call_id is not None:
                if not isinstance(call_id, str):
                    raise StreamProtocolError("OpenAI 工具调用 ID 必须是字符串")
                # 部分 Qwen 兼容接口在首个增量给出真实 ID，后续增量则
                # 固定发送空字符串。这里把空字符串视为“本段没有 ID”，
                # 继续保留已经收集到的值。
                if call_id:
                    # 非空 ID 重复且一致也是合法的；只有两个非空值前后
                    # 不一致时，才说明流式响应本身互相矛盾。
                    if builder.call_id is None:
                        builder.call_id = call_id
                    elif builder.call_id != call_id:
                        raise StreamProtocolError("OpenAI 工具调用 ID 前后冲突")

            call_type = call.get("type")
            if call_type is not None:
                if call_type != "function":
                    raise StreamProtocolError("OpenAI 工具调用类型必须为 function")
                # 同一个 type 也可能随每个增量重复发送。当前协议只接受
                # function，因此重复的相同值可以安全忽略。
                builder.saw_function_type = True

            function = call.get("function")
            if function is not None:
                function_object = _json_object(
                    function,
                    "OpenAI 工具 function 增量必须是对象",
                )
                name = function_object.get("name")
                if name is not None:
                    if not isinstance(name, str):
                        raise StreamProtocolError("OpenAI 工具名称必须是字符串")
                    # 允许重复声明同一个工具名，以兼容每个 chunk 都附带
                    # 完整元数据的实现；切换为另一个名字仍是协议错误。
                    if builder.name is None:
                        builder.name = name
                    elif builder.name != name:
                        raise StreamProtocolError("OpenAI 工具名称前后冲突")
                arguments = function_object.get("arguments")
                if arguments is not None:
                    if not isinstance(arguments, str):
                        raise StreamProtocolError("OpenAI 工具参数片段必须是字符串")
                    builder.argument_parts.append(arguments)

    def _finish_tools(
        self,
        builders: dict[int, _OpenAIToolBuilder],
    ) -> tuple[ToolCall, ...]:
        # 只有流结束后才能断言 ID、类型、名称和 JSON 参数全部到齐。过早解析
        # argument_parts 会把合法的半段 JSON 错报为协议错误。
        calls: list[ToolCall] = []
        if sorted(builders) != list(range(len(builders))):
            raise StreamProtocolError("OpenAI 工具调用索引不连续")
        for index in range(len(builders)):
            builder = builders[index]
            call_id = builder.call_id or ""
            name = builder.name or ""
            arguments_text = "".join(builder.argument_parts)
            if not call_id:
                raise StreamProtocolError("OpenAI 工具调用缺少 ID")
            if not builder.saw_function_type:
                raise StreamProtocolError("OpenAI 工具调用缺少 function 类型")
            if not name:
                raise StreamProtocolError("OpenAI 工具调用缺少名称")
            try:
                arguments = json.loads(arguments_text)
            except json.JSONDecodeError as exc:
                raise StreamProtocolError(
                    f"OpenAI 工具 {name} 的参数不是完整 JSON"
                ) from exc
            if not isinstance(arguments, dict):
                raise StreamProtocolError(
                    f"OpenAI 工具 {name} 的参数必须是 JSON 对象"
                )
            calls.append(ToolCall(call_id, name, arguments))
        return tuple(calls)

    async def stream(
        self,
        request: ProviderRequest,
    ) -> AsyncIterator[ProviderEvent]:
        # text_parts 用于构造最终 AssistantMessage，文本增量则立即向上 yield；
        # 因此 UI 能实时显示，同时历史仍只保存完整消息。
        headers = {
            "Authorization": f"Bearer {self._config.api_key.reveal()}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        text_parts: list[str] = []
        tool_builders: dict[int, _OpenAIToolBuilder] = {}
        finish_reason: str | None = None
        usage: ProviderUsage | None = None
        completed = False

        async for event in self._transport_events(request, headers):
            if event.data.strip() == "[DONE]":
                if completed:
                    raise StreamProtocolError("OpenAI 流包含重复的 [DONE] 标记")
                normalized_reason = self._stop_reason(finish_reason)
                calls = (
                    # 长度截断时工具参数可能不完整，不能执行已收集到的半个
                    # 调用；Agent 会将 MAX_TOKENS 作为失败结束本轮。
                    ()
                    if normalized_reason is ModelStopReason.MAX_TOKENS
                    else self._finish_tools(tool_builders)
                )
                blocks = []
                if text_parts:
                    blocks.append(TextBlock("".join(text_parts)))
                blocks.extend(calls)
                completed = True
                yield ProviderCompleted(
                    stop_reason=normalized_reason,
                    assistant_message=AssistantMessage(tuple(blocks)),
                    usage=usage,
                )
                continue
            if completed:
                raise StreamProtocolError("OpenAI 流在 [DONE] 后仍包含数据")
            try:
                payload = json.loads(event.data)
            except json.JSONDecodeError as exc:
                raise StreamProtocolError("OpenAI 流包含无效 JSON") from exc
            payload = _json_object(payload, "OpenAI 流事件必须是 JSON 对象")
            if "error" in payload:
                raise self._service_error(payload["error"])

            choices = payload.get("choices")
            # 启用流式 usage 的兼容服务可能额外发送 choices=[] 的统计块；
            # 它不包含模型语义事件，可以安全忽略。
            if choices == [] and "usage" in payload:
                current_usage = self._usage(payload.get("usage"))
                if current_usage is not None:
                    usage = current_usage
                continue
            if (
                not isinstance(choices, list)
                or not choices
                or not isinstance(choices[0], Mapping)
            ):
                raise StreamProtocolError("OpenAI 流事件缺少有效 choices")
            choice = choices[0]
            delta = _json_object(
                choice.get("delta"),
                "OpenAI 流事件缺少有效 delta",
            )
            if delta.get("function_call") is not None:
                # 项目只支持现代 tool_calls，不尝试把旧 function_call 与多工具
                # 索引模型混用，否则历史回灌和调用 ID 语义会变得不确定。
                raise StreamProtocolError("不支持 OpenAI 旧式 function_call")

            content = delta.get("content")
            if content is not None and not isinstance(content, str):
                raise StreamProtocolError("OpenAI 文本增量必须是字符串")
            if content:
                text_parts.append(content)
                yield ProviderTextDelta(content)

            if delta.get("tool_calls") is not None:
                self._consume_tool_deltas(delta["tool_calls"], tool_builders)

            current_reason = choice.get("finish_reason")
            # finish_reason 可能只在最后一个 JSON chunk 出现；若服务端重复
            # 声明，值必须一致，否则无法可靠判断是回答、工具还是截断。
            if current_reason is not None:
                if not isinstance(current_reason, str):
                    raise StreamProtocolError("OpenAI finish_reason 必须是字符串")
                if finish_reason is not None and finish_reason != current_reason:
                    raise StreamProtocolError("OpenAI finish_reason 前后冲突")
                finish_reason = current_reason

        if not completed:
            raise StreamProtocolError("OpenAI 流在收到 [DONE] 前中断")
