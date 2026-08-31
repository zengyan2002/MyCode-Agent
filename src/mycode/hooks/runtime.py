"""保存一个主会话或 Skill fork 独享的 Hook 运行状态。"""

from __future__ import annotations

import asyncio

from mycode.agent.instructions import RuntimeInstructionManager
from mycode.constants import HOOK_SHUTDOWN_TIMEOUT_SECONDS
from mycode.models.hooks import HookSource


class HookRunScope:
    """保存一段 Agent 运行期间不能与其他 Agent 共用的 Hook 状态。

    主会话每次新建或恢复时创建一个实例；每次 Skill fork 也创建一个临时
    实例。`HookEngine` 用这些属性判断 once、保存后台任务，并把 prompt
    动作写入正确的运行时提示管理器。

    Attributes:
        instruction_manager: 当前 Agent 下一次模型请求使用的提示管理器。
        executed_once: 已经成功执行或明确拒绝过的 once Hook 来源。
        in_flight_once: 已经启动但尚未得出结果的 once Hook 来源。
        background_tasks: 当前 scope 仍在执行的异步 Hook 任务。
    """

    def __init__(
        self,
        instruction_manager: RuntimeInstructionManager,
    ) -> None:
        """创建一份初始为空的运行状态。

        Args:
            instruction_manager: 当前主会话或 fork 自己持有的提示管理器。

        Returns:
            None。调用方随后把该实例传入 Agent runner 和工具调度器。
        """

        self.instruction_manager = instruction_manager
        self.executed_once: set[HookSource] = set()
        self.in_flight_once: set[HookSource] = set()
        self.background_tasks: set[asyncio.Task[None]] = set()
        self._once_lock = asyncio.Lock()
        self._closed = False

    async def reserve_once(self, source: HookSource) -> bool:
        """在动作启动前原子地预留一条 once Hook。

        Args:
            source: 需要检查并预留的规则来源。

        Returns:
            本 scope 尚未执行或启动该规则时返回 True；否则返回 False。
        """

        async with self._once_lock:
            if (
                self._closed
                or source in self.executed_once
                or source in self.in_flight_once
            ):
                return False
            self.in_flight_once.add(source)
            return True

    async def finish_once(self, source: HookSource, *, completed: bool) -> None:
        """记录一次预留动作的最终状态。

        Args:
            source: 先前由 `reserve_once` 预留的规则来源。
            completed: True 表示以后跳过；False 表示失败后允许再次尝试。

        Returns:
            None。方法会从执行中集合移除来源，并按结果更新完成集合。
        """

        async with self._once_lock:
            self.in_flight_once.discard(source)
            if completed:
                self.executed_once.add(source)

    def add_background_task(self, task: asyncio.Task[None]) -> None:
        """持有一个异步 Hook 任务，直到任务完成或 scope 关闭。

        Args:
            task: 已经由当前事件循环启动的 Hook 任务。

        Returns:
            None。完成回调会读取异常并从集合移除任务。
        """

        if self._closed:
            task.cancel()
            return
        self.background_tasks.add(task)

        def remove(finished: asyncio.Task[None]) -> None:
            """读取后台任务异常并移除 scope 持有的强引用。

            Args:
                finished: 刚刚进入完成、失败或取消状态的后台 Hook 任务。

            Returns:
                无返回值；任务异常被读取后不会产生未获取异常警告。
            """

            self.background_tasks.discard(finished)
            if not finished.cancelled():
                finished.exception()

        task.add_done_callback(remove)

    async def close(self) -> None:
        """在有限时间内收尾当前 scope 的全部后台 Hook。

        Returns:
            None。超时后仍未结束的任务会被取消并等待退出；重复调用不做事。
        """

        if self._closed:
            return
        self._closed = True
        tasks = tuple(self.background_tasks)
        if not tasks:
            return
        done, pending = await asyncio.wait(
            tasks,
            timeout=HOOK_SHUTDOWN_TIMEOUT_SECONDS,
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            if not task.cancelled():
                task.exception()
