"""在应用启动后周期扫描过期临时 Worktree。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from mycode.models.config import WorktreeSettings
from mycode.models.worktrees import CleanupReport
from mycode.worktrees.manager import WorktreeManager


class WorktreeCleanupService:
    """运行一次启动扫描，并按配置间隔重复清理检查。

    Attributes:
        manager: 提供受管记录、租约和 fail-closed 变更检查的 Manager。
        settings: 过期小时数和扫描间隔配置。
        _task: 周期循环的 asyncio 任务；尚未启动或已经关闭时为 ``None``。
        _stop: ``close`` 设置的事件，让循环不需要等待完整间隔即可结束。
        _last_report: 最近一次扫描结果，供状态命令观察。
    """

    def __init__(
        self,
        manager: WorktreeManager,
        settings: WorktreeSettings,
    ) -> None:
        """创建一个尚未运行的后台清理服务。

        Args:
            manager: 已装配但可以尚未 ``start`` 的 Worktree Manager。
            settings: 已校验的 Worktree 清理配置。

        Returns:
            新的清理服务。

        Raises:
            ValueError: Manager 或配置类型无效。
        """

        if not isinstance(manager, WorktreeManager):
            raise ValueError("WorktreeCleanupService.manager 类型无效")
        if not isinstance(settings, WorktreeSettings):
            raise ValueError("WorktreeCleanupService.settings 类型无效")
        self.manager = manager
        self.settings = settings
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._last_report: CleanupReport | None = None

    @property
    def last_report(self) -> CleanupReport | None:
        """读取最近一次启动或周期扫描结果。

        Returns:
            尚未扫描时返回 ``None``，否则返回不可变 ``CleanupReport``。
        """

        return self._last_report

    async def start(self) -> CleanupReport:
        """立即执行一次扫描，然后启动周期循环。

        Returns:
            启动扫描的 ``CleanupReport``。

        Raises:
            RuntimeError: 同一个服务重复启动。
            WorktreeManagerError: Manager 未启动或状态不可信。
        """

        if self._task is not None:
            raise RuntimeError("WorktreeCleanupService 已经启动")
        self._stop.clear()
        self._last_report = await self.run_once()
        self._task = asyncio.create_task(
            self._run_loop(),
            name="mycode-worktree-cleanup",
        )
        return self._last_report

    async def run_once(self) -> CleanupReport:
        """用一个统一当前时间执行单次过期清理。

        Returns:
            本次 Manager 清理报告，同时保存到 ``last_report``。
        """

        report = await self.manager.cleanup_stale(
            datetime.now(UTC),
            stale_after_hours=self.settings.stale_after_hours,
        )
        self._last_report = report
        return report

    async def close(self) -> None:
        """停止周期清理任务并等待它退出。

        Returns:
            服务未启动时直接返回；循环结束且任务引用清空后不返回数据。
        """

        task = self._task
        if task is None:
            return
        self._stop.set()
        await task
        self._task = None

    async def _run_loop(self) -> None:
        """等待配置间隔并重复扫描，直到 ``close`` 发出停止事件。

        Returns:
            停止事件触发后返回。单次扫描异常不会结束循环，下次间隔仍会重试。
        """

        while True:
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self.settings.cleanup_interval_seconds,
                )
                return
            except TimeoutError:
                try:
                    await self.run_once()
                except Exception:
                    # Manager 和状态命令保留具体错误；后台循环不能因一次磁盘或
                    # Git 失败永久停止后续扫描。
                    continue
