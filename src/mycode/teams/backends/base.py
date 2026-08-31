"""三种成员后端共同使用的启动参数和运行接口。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from mycode.models.teams import TeammateBackend


@dataclass(frozen=True, slots=True)
class TeammateLaunch:
    """保存启动一个成员 Host 所需的工作区和可信身份。

    Attributes:
        workspace_root: Lead 所在主仓库绝对路径。
        worktree_path: 成员工具实际使用的绝对工作目录。
        team_id: 成员所属团队 ID。
        agent_id: 成员不可变内部 ID。
        generation: 本次 Host 写状态使用的 generation。
        lease_token: 只通过进程环境或内存传递的租约原文。
        prompt: 成员首次运行时处理的具体工作说明。
        environment: 外部 Host 额外继承的非敏感环境字段。
    """

    workspace_root: Path
    worktree_path: Path
    team_id: str
    agent_id: str
    generation: int
    lease_token: str
    prompt: str
    environment: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BackendHandle:
    """保存 Supervisor 后续唤醒、探测和停止成员所需的后端引用。

    Attributes:
        backend: 实际创建该句柄的后端。
        reference: pane、session 或同进程 task 的稳定引用文字。
        process_id: 外部 Host 的进程 ID；无法取得时为空。
    """

    backend: TeammateBackend
    reference: str
    process_id: int | None = None


@dataclass(frozen=True, slots=True)
class BackendProbe:
    """说明一个后端引用当前是否仍有 Host 接收事件。

    Attributes:
        alive: 后端确认 Host 仍可接收唤醒或停止请求时为 True。
        detail: 供启动、恢复失败信息展示的探测说明。
    """

    alive: bool
    detail: str = ""


class TeammateBackendAdapter(Protocol):
    """由三个真实生产后端实现的成员进程控制接口。

    Attributes:
        backend: 当前实现负责的 tmux、iTerm2 或 in-process 后端类型。
    """

    backend: TeammateBackend

    async def start(self, launch: TeammateLaunch) -> BackendHandle:
        """使用已选定后端启动一个成员 Host。

        Args:
            launch: 团队、成员、generation、租约、工作区和首次提示。

        Returns:
            已完成启动握手、可供 wake/stop/probe 使用的后端句柄。
        """

    async def wake(self, handle: BackendHandle) -> None:
        """通知 idle 或 suspended Host 读取已经落盘的事件。

        Args:
            handle: start 返回或从成员记录恢复的后端句柄。

        Returns:
            唤醒信号成功送达后不返回数据。
        """

    async def stop(self, handle: BackendHandle, *, force: bool) -> None:
        """正常或强制停止句柄对应的成员 Host。

        Args:
            handle: 要停止的成员后端句柄。
            force: True 时允许后端跳过优雅退出并终止进程或 task。

        Returns:
            后端确认 Host 已停止后不返回数据。
        """

    async def probe(self, handle: BackendHandle) -> BackendProbe:
        """检查句柄对应的 Host 当前是否仍在运行。

        Args:
            handle: 要检查的成员后端句柄。

        Returns:
            包含是否存活和诊断说明的探测结果。
        """


WakeWaiter = Callable[[], Awaitable[None]]
HostCoroutine = Callable[[TeammateLaunch, WakeWaiter], Awaitable[None]]
