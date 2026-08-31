"""在独立 tmux pane 或新 session 中启动完整成员 CLI。"""

from __future__ import annotations

import asyncio
import os
import secrets

from mycode.models.teams import TeammateBackend
from mycode.teams.backends.base import BackendHandle, BackendProbe, TeammateLaunch


class TmuxBackend:
    """用 tmux 命令创建、唤醒、探测和停止成员 pane。

    Attributes:
        backend: 该实现对应的固定后端类型 ``tmux``。
    """

    backend = TeammateBackend.TMUX

    async def start(self, launch: TeammateLaunch) -> BackendHandle:
        """创建 pane/session，并把租约放进子进程环境而不是命令参数。

        Args:
            launch: 成员身份、工作目录、租约和子进程环境。

        Returns:
            保存 tmux pane ID 或 session 名称的后端句柄。

        Raises:
            RuntimeError: tmux 无法创建 pane 或 session。
        """

        target = f"mycode-{launch.agent_id[-8:]}"
        command = self._host_command(launch)
        if os.environ.get("TMUX"):
            args = ["tmux", "split-window", "-d", "-P", "-F", "#{pane_id}", "-c", str(launch.worktree_path), command]
        else:
            args = ["tmux", "new-session", "-d", "-s", target, "-c", str(launch.worktree_path), command]
        code, stdout, stderr = await self._run(args, launch)
        if code != 0:
            raise RuntimeError(f"tmux 成员启动失败：{stderr.strip() or stdout.strip()}")
        reference = stdout.strip() if stdout.strip().startswith("%") else target
        return BackendHandle(self.backend, reference)

    async def wake(self, handle: BackendHandle) -> None:
        """向目标 pane 发送空输入，使其从文件事件等待中醒来。

        Args:
            handle: ``start`` 返回的 pane 或 session 句柄。

        Returns:
            输入发送成功后不返回数据。

        Raises:
            RuntimeError: tmux 找不到目标或发送输入失败。
        """

        code, _, stderr = await self._run_plain(["tmux", "send-keys", "-t", handle.reference, "", "Enter"])
        if code != 0:
            raise RuntimeError(f"无法唤醒 tmux 成员：{stderr.strip()}")

    async def stop(self, handle: BackendHandle, *, force: bool) -> None:
        """关闭目标 pane 或 session。

        Args:
            handle: ``start`` 返回的 pane 或 session 句柄。
            force: 后端接口统一提供的强制停止标志；tmux 的 kill 命令不区分此标志。

        Returns:
            目标已关闭或原本不存在时不返回数据。

        Raises:
            RuntimeError: 目标仍存在但 tmux 关闭失败。
        """

        action = "kill-pane" if handle.reference.startswith("%") else "kill-session"
        code, _, stderr = await self._run_plain(["tmux", action, "-t", handle.reference])
        if code != 0 and "can't find" not in stderr.lower():
            raise RuntimeError(f"无法停止 tmux 成员：{stderr.strip()}")

    async def probe(self, handle: BackendHandle) -> BackendProbe:
        """使用 tmux has-session/list-panes 判断引用是否仍存在。

        Args:
            handle: 要查询的 pane 或 session 句柄。

        Returns:
            包含存活判断和 tmux 错误文本的探测结果。
        """

        if handle.reference.startswith("%"):
            args = ["tmux", "list-panes", "-a", "-F", "#{pane_id}"]
            code, stdout, stderr = await self._run_plain(args)
            return BackendProbe(code == 0 and handle.reference in stdout.splitlines(), stderr.strip())
        code, _, stderr = await self._run_plain(["tmux", "has-session", "-t", handle.reference])
        return BackendProbe(code == 0, stderr.strip())

    @staticmethod
    def _host_command(launch: TeammateLaunch) -> str:
        """构造 tmux pane 内部启动成员 Host 的无 Shell 参数字符串。

        Args:
            launch: 已选后端收到的成员身份、generation 和租约。

        Returns:
            供 tmux ``send-keys`` 输入的 MyCode 内部 Host 命令。
        """

        return f'python -m mycode --team-host "{launch.team_id}" "{launch.agent_id}" {launch.generation}'

    async def _run(self, args: list[str], launch: TeammateLaunch) -> tuple[int, str, str]:
        """运行 tmux 子命令，并把成员租约放进子进程环境。

        Args:
            args: ``tmux`` 后面的参数列表。
            launch: 提供 team root 和租约的启动数据。

        Returns:
            子进程退出码、标准输出和标准错误。
        """

        environment = dict(os.environ)
        environment.update(launch.environment)
        environment["MYCODE_TEAM_LEASE"] = launch.lease_token
        environment["MYCODE_TEAM_ROOT"] = str(launch.workspace_root)
        return await self._run_plain(args, environment)

    @staticmethod
    async def _run_plain(args: list[str], environment: dict[str, str] | None = None) -> tuple[int, str, str]:
        """异步执行一条参数边界明确的外部命令。

        Args:
            args: 包含可执行程序名的完整参数数组。
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
