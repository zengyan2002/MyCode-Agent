"""Agent 循环与工具调度器共用的取消控制原语。"""

from __future__ import annotations

import asyncio
from enum import Enum


class CancellationReason(str, Enum):
    EXTERNAL = "external"
    DEADLINE = "deadline"

#CancellationToken 就是多个异步组件共用的“任务已取消”令牌
class CancellationToken:
    def __init__(self) -> None:
        # asyncio.Event 同时支持已经取消后的立即观察和多个等待者唤醒，
        # 比在各层传递 Task.cancel() 更适合表达跨 Provider/工具的协作式取消。
        self._event = asyncio.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()


class CancellationController:
    """把调用方取消信号和可选截止时间合并为一个本地取消信号。"""

    def __init__(
        self,
        external: CancellationToken,
        timeout_seconds: float | None,
    ) -> None:
        # 对外暴露的是新的本地 token：下游只监听一个信号，不需要分别处理
        # 调用方取消和截止时间。reason 则保留首个触发源供错误分类使用。
        self.token = CancellationToken()
        self._external = external
        self._timeout_seconds = timeout_seconds
        self._reason: CancellationReason | None = None
        self._watchers: list[asyncio.Task[None]] = []

    @property
    def reason(self) -> CancellationReason | None:
        return self._reason

    async def __aenter__(self) -> "CancellationController":
        # 外部 token 可能在控制器创建前已经取消，必须先同步检查，避免启动
        # watcher 后产生一个短暂的“尚未取消”窗口。
        if self._external.is_cancelled:
            self.cancel(CancellationReason.EXTERNAL)
        else:
            self._watchers.append(asyncio.create_task(self._watch_external()))
        if self._timeout_seconds is not None:
            self._watchers.append(asyncio.create_task(self._watch_deadline()))
        return self

    async def __aexit__(self, *args: object) -> None:
        # watcher 只服务当前 Agent 回合。退出作用域时统一取消并等待回收，
        # 避免下一轮事件循环中残留等待任务或“Task was destroyed”警告。
        for watcher in self._watchers:
            watcher.cancel()
        if self._watchers:
            await asyncio.gather(*self._watchers, return_exceptions=True)
        self._watchers.clear()

    def cancel(self, reason: CancellationReason) -> None:
        # 多个信号可能几乎同时到达。只记录第一个原因，保证用户取消和超时
        # 的最终错误分类稳定，不因调度顺序在已经取消后反复改变。
        if self._reason is None:
            self._reason = reason
            self.token.cancel()

    async def _watch_external(self) -> None:
        await self._external.wait()
        self.cancel(CancellationReason.EXTERNAL)

    async def _watch_deadline(self) -> None:
        assert self._timeout_seconds is not None
        await asyncio.sleep(self._timeout_seconds)
        self.cancel(CancellationReason.DEADLINE)
