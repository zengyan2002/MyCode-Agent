"""在创建成员前一次性选择 tmux、iTerm2 或同进程后端。"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping

from mycode.models.teams import BackendPreference, TeammateBackend


class BackendDetectionError(RuntimeError):
    """表示用户显式指定的成员后端在当前环境不可用。"""


class BackendDetector:
    """根据创建前的终端环境和可执行程序选择一次后端。

    Attributes:
        environment: 检测时读取的环境变量快照。
    """

    def __init__(self, environment: Mapping[str, str] | None = None) -> None:
        """冻结检测使用的环境变量，避免选择过程中前后不一致。

        Args:
            environment: 测试或调用方提供的环境；未传时复制当前进程环境。

        Returns:
            不返回数据。
        """

        self.environment = dict(os.environ if environment is None else environment)

    def select(self, preference: BackendPreference) -> TeammateBackend:
        """按固定优先级选择后端，显式选择不可用时直接报错。

        Args:
            preference: auto 或用户明确指定的唯一后端。

        Returns:
            创建成员期间不再改变的实际后端。

        Raises:
            BackendDetectionError: 显式指定后端不可用。
        """

        if preference is BackendPreference.TMUX:
            if not self._tmux_available():
                raise BackendDetectionError("显式指定 tmux，但当前找不到可执行程序")
            return TeammateBackend.TMUX
        if preference is BackendPreference.ITERM2:
            if not self._iterm_available():
                raise BackendDetectionError("显式指定 iTerm2，但当前不在 iTerm2 或找不到 it2")
            return TeammateBackend.ITERM2
        if preference is BackendPreference.IN_PROCESS:
            return TeammateBackend.IN_PROCESS
        if self._inside_tmux():
            return TeammateBackend.TMUX
        if self._iterm_available():
            return TeammateBackend.ITERM2
        if self._tmux_available():
            return TeammateBackend.TMUX
        return TeammateBackend.IN_PROCESS

    def _inside_tmux(self) -> bool:
        """判断当前进程确实位于 tmux 且可执行程序仍可用。

        Returns:
            环境含 TMUX 并能解析 tmux 可执行程序时为 True。
        """

        return bool(self.environment.get("TMUX")) and self._tmux_available()

    def _iterm_available(self) -> bool:
        """判断当前终端为 iTerm2 且原生 ``it2`` 命令可用。

        Returns:
            两项条件同时满足时为 True。
        """

        return (
            self.environment.get("TERM_PROGRAM") == "iTerm.app"
            and shutil.which("it2", path=self.environment.get("PATH")) is not None
        )

    def _tmux_available(self) -> bool:
        """判断冻结的 PATH 中能否找到 tmux 可执行程序。

        Returns:
            能解析 tmux 路径时为 True。
        """

        return shutil.which("tmux", path=self.environment.get("PATH")) is not None
