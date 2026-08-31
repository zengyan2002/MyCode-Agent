"""在 Lead 进程的 asyncio 事件循环中运行成员 Host。"""

from __future__ import annotations

import asyncio
import secrets

from mycode.models.teams import TeammateBackend
from mycode.teams.backends.base import (
    BackendHandle,
    BackendProbe,
    HostCoroutine,
    TeammateLaunch,
)


class InProcessBackend:
    """管理当前进程内的成员 task 和唤醒 Event。

    Attributes:
        host: 接收启动数据并运行 TeammateHost 的生产协程。
    """

    backend = TeammateBackend.IN_PROCESS

    def __init__(self, host: HostCoroutine) -> None:
        """保存实际运行成员事件循环的协程入口。

        Args:
            host: 接收成员启动数据和等待函数、持续处理团队事件的协程。

        Returns:
            新的同进程后端实例。
        """

        self.host = host
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._events: dict[str, asyncio.Event] = {}

    async def start(self, launch: TeammateLaunch) -> BackendHandle:
        """创建一个 Host task，并返回同进程运行 ID。

        Args:
            launch: 成员身份、工作目录、租约和 Host 初始配置。

        Returns:
            保存新 task 运行 ID 的后端句柄。

        Raises:
            RuntimeError: Host 在启动后的首轮调度中立即失败。
        """

        reference = f"inproc-{secrets.token_hex(6)}"
        self._events[reference] = asyncio.Event()
        event = self._events[reference]
        task = asyncio.create_task(
            self.host(launch, event.wait),
            name=f"team:{launch.agent_id}",
        )
        self._tasks[reference] = task
        await asyncio.sleep(0)
        if task.done() and task.exception() is not None:
            error = task.exception()
            self._tasks.pop(reference, None)
            self._events.pop(reference, None)
            raise RuntimeError(f"in-process 成员启动失败：{error}")
        return BackendHandle(self.backend, reference, None)

    async def wake(self, handle: BackendHandle) -> None:
        """设置成员运行 ID 对应的 asyncio Event。

        Args:
            handle: ``start`` 返回的同进程运行句柄。

        Returns:
            Event 设置完成后不返回数据。

        Raises:
            RuntimeError: 运行 ID 已不存在。
        """

        event = self._events.get(handle.reference)
        if event is None:
            raise RuntimeError("in-process 成员句柄不存在")
        event.set()

    async def stop(self, handle: BackendHandle, *, force: bool) -> None:
        """取消同进程 task；正常保存由 Host 在取消处理中完成。

        Args:
            handle: ``start`` 返回的同进程运行句柄。
            force: 后端接口统一提供的强制停止标志；asyncio 取消不区分此标志。

        Returns:
            task 已停止或原本不存在时不返回数据。
        """

        task = self._tasks.get(handle.reference)
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._tasks.pop(handle.reference, None)
            self._events.pop(handle.reference, None)

    async def probe(self, handle: BackendHandle) -> BackendProbe:
        """根据 asyncio task 是否结束返回实际存活状态。

        Args:
            handle: 要查询的同进程运行句柄。

        Returns:
            包含 task 是否仍在运行及固定说明文本的探测结果。
        """

        task = self._tasks.get(handle.reference)
        return BackendProbe(task is not None and not task.done(), "同进程 asyncio task")

    def wake_event(self, reference: str) -> asyncio.Event:
        """返回 Host 等待的 Event；句柄不存在时直接报错。

        Args:
            reference: start 返回的同进程运行 ID。

        Returns:
            Supervisor 和 Host 共用的 asyncio.Event。
        """

        try:
            return self._events[reference]
        except KeyError as exc:
            raise RuntimeError("in-process 成员句柄不存在") from exc
