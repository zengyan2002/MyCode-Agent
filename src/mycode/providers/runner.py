"""执行 Provider 流，并统一处理取消、关闭和完成事件协议。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from mycode.agent.cancellation import CancellationToken
from mycode.errors import StreamProtocolError
from mycode.models.provider import (
    ProviderCompleted,
    ProviderEvent,
    ProviderRequest,
)
from mycode.providers.base import Provider


class ProviderRequestCancelled(Exception):
    """本地取消信号先于 Provider 下一条流事件到达。"""


class ProviderRequestRunner:
    """负责发送模型请求，并处理模型返回的事件流

    普通对话通过 events 逐条获取事件，以便界面实时显示模型输出；上下文摘要
    通过 collect 等待完整响应，不显示生成过程。这个类还负责响应用户取消、
    关闭事件流，并检查模型响应必须以一个 ProviderCompleted 事件结束
    """

    def __init__(self, provider: Provider) -> None:
        # 当前会话固定使用的协议适配器；普通与摘要请求共用该对象和连接池。
        self._provider = provider

    async def _close_iterator(self, iterator: object) -> None:
        """关闭支持 ``aclose`` 的 Provider 迭代器。"""
        close = getattr(iterator, "aclose", None)
        if callable(close):
            await close()

    async def _next_event(
        self,
        iterator: AsyncIterator[ProviderEvent],
    ) -> ProviderEvent | BaseException:
        """读取下一条 Provider 流事件；遇到键盘中断或进程退出时返回异常对象，由调用方重新抛出"""
        try:
            return await anext(iterator)
        except (KeyboardInterrupt, SystemExit) as exc:
            return exc

    async def events(
        self,
        request: ProviderRequest,
        cancellation: CancellationToken,
    ) -> AsyncIterator[ProviderEvent]:
        """发送一次请求并逐条返回事件。

        Args:
            request: 已包含 Prompt、消息和当前工具视图的完整请求。
            cancellation: 当前 Agent 回合或手动压缩使用的取消信号。

        Yields:
            Provider 产生的思考、文本和唯一完成事件。
        """
        if cancellation.is_cancelled:
            raise ProviderRequestCancelled
        iterator = self._provider.stream(request)
        completed = False
        try:
            while True:
                next_event = asyncio.create_task(self._next_event(iterator))
                cancel_waiter = asyncio.create_task(cancellation.wait())
                try:
                    done, _ = await asyncio.wait(
                        {next_event, cancel_waiter},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if cancel_waiter in done:
                        if not next_event.done():
                            next_event.cancel()
                            await asyncio.gather(
                                next_event,
                                return_exceptions=True,
                            )
                        raise ProviderRequestCancelled
                    try:
                        event = await next_event
                    except StopAsyncIteration:
                        break
                    if isinstance(event, (KeyboardInterrupt, SystemExit)):
                        raise event
                finally:
                    if not next_event.done():
                        next_event.cancel()
                        await asyncio.gather(next_event, return_exceptions=True)
                    cancel_waiter.cancel()
                    await asyncio.gather(cancel_waiter, return_exceptions=True)

                if completed:
                    raise StreamProtocolError("Provider 在完成事件后仍产生数据")
                if isinstance(event, ProviderCompleted):
                    completed = True
                yield event
        finally:
            await self._close_iterator(iterator)
        if not completed:
            raise StreamProtocolError("Provider 流缺少完成事件")

    async def collect(
        self,
        request: ProviderRequest,
        cancellation: CancellationToken,
    ) -> ProviderCompleted:
        """读取模型返回的全部事件，只返回最后的完成结果

        处理中收到的增量事件不会发送给界面。事件流缺少完成事件时抛出协议错误；
        请求被取消时，继续向调用方抛出 ``ProviderRequestCancelled`
        """
        completed: ProviderCompleted | None = None
        async for event in self.events(request, cancellation):
            if isinstance(event, ProviderCompleted):
                completed = event
        if completed is None:
            raise StreamProtocolError("Provider 完成事件缺少响应")
        return completed
