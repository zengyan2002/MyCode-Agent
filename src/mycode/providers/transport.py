"""基于 HTTPX 的异步 SSE 网络传输层。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from mycode.constants import (
    ERROR_BODY_LIMIT_BYTES,
    HTTP_CONNECT_TIMEOUT_SECONDS,
    HTTP_POOL_TIMEOUT_SECONDS,
    HTTP_WRITE_TIMEOUT_SECONDS,
)
from mycode.errors import (
    AuthenticationError,
    HttpStatusError,
    MyCodeError,
    StreamProtocolError,
    TransportError,
)
from mycode.providers.sse import SSEEvent, iter_sse


class HttpTransport:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # 测试可以注入完整 client 或仅注入 MockTransport，但二者不能同时
        # 提供，否则无法明确由谁拥有连接池及关闭责任。
        if client is not None and transport is not None:
            raise ValueError("client 与 transport 只能传入其中一个")
        timeout = httpx.Timeout(
            connect=HTTP_CONNECT_TIMEOUT_SECONDS,
            read=None,
            write=HTTP_WRITE_TIMEOUT_SECONDS,
            pool=HTTP_POOL_TIMEOUT_SECONDS,
        )
        # read=None 是流式响应的必要选择：模型可能长时间思考而不产生新块。
        # 整轮截止时间由 Agent 层控制，不能用普通 HTTP 读超时替代。
        self._client = client or httpx.AsyncClient(
            timeout=timeout,
            transport=transport,
        )
        self._closed = False

    # 错误响应体可能包含服务端给出的具体原因；限制读取长度，避免异常页
    # 或代理返回的大段内容污染终端输出。
    async def _read_error_body(self, response: httpx.Response) -> str:
        chunks = bytearray()
        async for chunk in response.aiter_bytes():
            remaining = ERROR_BODY_LIMIT_BYTES - len(chunks)
            if remaining <= 0:
                break
            chunks.extend(chunk[:remaining])
            if len(chunks) >= ERROR_BODY_LIMIT_BYTES:
                break
        return bytes(chunks).decode("utf-8", errors="replace").strip()

    # 向模型服务发送流式 HTTP 请求，并把响应中的 SSE 文本逐条解析成事件。
    async def stream_sse(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        json_body: Mapping[str, object],
    ) -> AsyncIterator[SSEEvent]:
        if self._closed:
            raise TransportError("HTTP 客户端已经关闭")
        try:
            async with self._client.stream(
                "POST",
                url,
                headers=dict(headers),
                json=dict(json_body),
            ) as response:
                if response.status_code >= 400:
                    body = await self._read_error_body(response)
                    detail = f"：{body}" if body else ""
                    if response.status_code in {401, 403}:
                        raise AuthenticationError(
                            f"认证失败（HTTP {response.status_code}）{detail}"
                        )
                    raise HttpStatusError(response.status_code, body)
                content_type = response.headers.get("content-type", "").lower()
                # 某些测试响应可能省略 Content-Type，因此只在服务端明确给出
                # 非 SSE 类型时拒绝；这兼顾严格校验与可注入传输测试。
                if content_type and "text/event-stream" not in content_type:
                    raise StreamProtocolError(
                        f"服务返回的内容类型不是 SSE：{content_type}"
                    )
                async for event in iter_sse(response.aiter_lines()):
                    yield event
        except MyCodeError:
            raise
        except httpx.TimeoutException as exc:
            raise TransportError(f"请求超时：{exc}") from exc
        except httpx.RequestError as exc:
            raise TransportError(f"网络请求失败：{exc}") from exc

    async def aclose(self) -> None:
        # 关闭操作幂等，便于应用 finally 和测试清理路径安全地重复调用。
        if not self._closed:
            self._closed = True
            await self._client.aclose()
