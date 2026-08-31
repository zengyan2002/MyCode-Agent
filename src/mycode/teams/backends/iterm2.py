"""使用 it2 在 iTerm2 原生 split pane 中启动成员 CLI。"""

from __future__ import annotations

import asyncio
import os

from mycode.models.teams import TeammateBackend
from mycode.teams.backends.base import BackendHandle, BackendProbe, TeammateLaunch


class ITerm2Backend:
    """用 it2 命令创建和控制 iTerm2 成员 pane。

    Attributes:
        backend: 该实现对应的固定后端类型 ``iterm2``。
    """

    backend = TeammateBackend.ITERM2

    async def start(self, launch: TeammateLaunch) -> BackendHandle:
        """在新 split pane 中运行内部 Host，失败时不尝试其他后端。

        Args:
            launch: 成员身份、工作目录、租约和子进程环境。

        Returns:
            保存 iTerm2 session 标识的后端句柄。

        Raises:
            RuntimeError: it2 启动失败或没有返回 session 标识。
        """

        environment = dict(os.environ)
        environment.update(launch.environment)
        environment["MYCODE_TEAM_LEASE"] = launch.lease_token
        environment["MYCODE_TEAM_ROOT"] = str(launch.workspace_root)
        command = f'python -m mycode --team-host "{launch.team_id}" "{launch.agent_id}" {launch.generation}'
        code, stdout, stderr = await self._run(
            ["it2", "split", "--cwd", str(launch.worktree_path), "--", command], environment
        )
        if code != 0 or not stdout.strip():
            raise RuntimeError(f"iTerm2 成员启动失败：{stderr.strip() or stdout.strip()}")
        return BackendHandle(self.backend, stdout.strip())

    async def wake(self, handle: BackendHandle) -> None:
        """用 it2 向目标 session 发送空输入触发下一轮事件读取。

        Args:
            handle: ``start`` 返回的目标 session 句柄。

        Returns:
            唤醒命令成功后不返回数据。

        Raises:
            RuntimeError: it2 无法向目标 session 发送输入。
        """

        code, _, stderr = await self._run(["it2", "send-text", "--session", handle.reference, "\n"])
        if code != 0:
            raise RuntimeError(f"无法唤醒 iTerm2 成员：{stderr.strip()}")

    async def stop(self, handle: BackendHandle, *, force: bool) -> None:
        """关闭目标 iTerm2 session。

        Args:
            handle: ``start`` 返回的目标 session 句柄。
            force: 后端接口统一提供的强制停止标志；iTerm2 关闭命令不区分此标志。

        Returns:
            session 已关闭或原本不存在时不返回数据。

        Raises:
            RuntimeError: 目标仍存在但 it2 关闭失败。
        """

        code, _, stderr = await self._run(["it2", "close", "--session", handle.reference])
        if code != 0 and "not found" not in stderr.lower():
            raise RuntimeError(f"无法停止 iTerm2 成员：{stderr.strip()}")

    async def probe(self, handle: BackendHandle) -> BackendProbe:
        """查询目标 iTerm2 session 是否仍存在。

        Args:
            handle: 要查询的目标 session 句柄。

        Returns:
            包含存活判断和 it2 错误文本的探测结果。
        """

        code, stdout, stderr = await self._run(["it2", "list-sessions"])
        return BackendProbe(code == 0 and handle.reference in stdout.splitlines(), stderr.strip())

    @staticmethod
    async def _run(
        args: list[str], environment: dict[str, str] | None = None
    ) -> tuple[int, str, str]:
        """执行一条 ``it2`` 命令并传递成员 Host 所需环境。

        Args:
            args: ``it2`` 后面的参数列表。
            environment: 可选完整子进程环境；为空时继承当前环境。

        Returns:
            子进程退出码、标准输出和标准错误。
        """

        process = await asyncio.create_subprocess_exec(
            *args,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return process.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")
