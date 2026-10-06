"""通过标准输入管道管理独立成员进程，支持原生 Windows。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import secrets
import subprocess
import sys

from mycode.models.teams import TeammateBackend
from mycode.tools.processes import terminate_process_tree
from mycode.teams.backends.base import BackendHandle, BackendProbe, TeammateLaunch


class SubprocessBackend:
    """管理本次 Lead 启动的进程；旧句柄交由 Supervisor 重新启动。

    不按持久化 PID 操作进程，避免 PID 被复用时误杀其他程序。
    """

    backend = TeammateBackend.SUBPROCESS

    def __init__(self, *, stop_timeout: float = 13.0) -> None:
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._stop_timeout = stop_timeout

    @staticmethod
    def _host_args(launch: TeammateLaunch) -> list[str]:
        return [sys.executable, "-m", "mycode", "--team-host",
                launch.team_id, launch.agent_id, str(launch.generation)]

    async def start(self, launch: TeammateLaunch) -> BackendHandle:
        """直接启动当前 Python；租约只通过环境传递，握手由 Supervisor 校验。"""
        environment = dict(os.environ)
        environment.update(launch.environment)
        environment["MYCODE_TEAM_LEASE"] = launch.lease_token
        environment["MYCODE_TEAM_ROOT"] = str(launch.workspace_root)
        # 从源码运行时，切换到成员 worktree 后也能找到当前版本的包。
        package_root = str(Path(__file__).resolve().parents[3])
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
            package_root, environment.get("PYTHONPATH", ""),
        )))
        process = subprocess.Popen(
            self._host_args(launch), cwd=launch.worktree_path, env=environment,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if sys.platform == "win32" else 0,
            start_new_session=sys.platform != "win32",
        )
        reference = f"subprocess-{secrets.token_hex(12)}"
        self._processes[reference] = process
        return BackendHandle(self.backend, reference, process.pid)

    async def wake(self, handle: BackendHandle) -> None:
        """确认受管成员仍存活；实际唤醒来自持久化邮箱和任务通知。"""
        process = self._processes.get(handle.reference)
        if process is None or process.poll() is not None:
            raise RuntimeError("subprocess 成员已退出或不属于当前 Lead，请恢复成员")
        # 消息和认领通知已落盘；Host 定时读取，不再向未消费的 stdin 写入。

    async def stop(self, handle: BackendHandle, *, force: bool) -> None:
        """等待 Host 处理停止意图；超时或强制停止时清理受管进程树。"""
        process = self._processes.get(handle.reference)
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            if not force:
                deadline = asyncio.get_running_loop().time() + self._stop_timeout
                while process.poll() is None and asyncio.get_running_loop().time() < deadline:
                    await asyncio.sleep(0.05)
            if process.poll() is None:
                try:
                    await asyncio.to_thread(terminate_process_tree, process)
                except ProcessLookupError:
                    pass
            await asyncio.to_thread(process.wait)
        self._processes.pop(handle.reference, None)

    async def probe(self, handle: BackendHandle) -> BackendProbe:
        """只探测当前实例拥有的进程，不信任磁盘保存的 PID。"""
        process = self._processes.get(handle.reference)
        if process is None:
            return BackendProbe(False, "当前 Lead 未持有该进程，需要重新启动")
        code = process.poll()
        if code is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            self._processes.pop(handle.reference, None)
        return BackendProbe(code is None, "成员进程运行中" if code is None else f"成员进程已退出，退出码 {code}")
