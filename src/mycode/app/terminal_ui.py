"""根据终端能力创建全屏或普通输出界面。"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from typing import Any, Literal, Protocol

from prompt_toolkit.input import Input
from prompt_toolkit.output import Output
from rich.console import Console

from mycode.app.fullscreen_ui import FullscreenUI
from mycode.app.plain_ui import PlainUI
from mycode.app.ui_models import sanitize_terminal_text
from mycode.commands.completion import CommandCompleter
from mycode.models.config import SecretValue
from mycode.models.config import ProviderConfig
from mycode.models.events import AgentEvent
from mycode.models.permissions import (
    ApprovalChoice,
    PermissionMode,
    PermissionRequest,
)
from mycode.app.ui_models import UIAction
TerminalUIMode = Literal["fullscreen", "plain"]


class TerminalUI(Protocol):
    """声明应用、命令和权限拦截器共同依赖的终端行为。"""

    def configure(
        self,
        provider: ProviderConfig,
        permission_mode: PermissionMode,
    ) -> None:
        """设置界面首次运行需要的 Provider 和权限状态。

        Args:
            provider: 状态区显示的 Provider 名称和模型配置。
            permission_mode: 启动时生效的权限模式。

        Returns:
            None。
        """
        ...

    async def run_async(self) -> None:
        """启动界面并等待应用通知停止。

        Returns:
            None。
        """
        ...

    async def next_action(self) -> UIAction:
        """等待用户提交文字、取消或退出。

        Returns:
            应用循环下一步要处理的用户操作。
        """
        ...

    def stop(self) -> None:
        """通知界面结束运行并释放等待中的输入。

        Returns:
            None。
        """
        ...

    def set_plan_mode(self, enabled: bool) -> None:
        """更新输入提示和状态栏中的 Plan 模式。

        Args:
            enabled: True 表示显示 Plan 模式，False 表示执行模式。

        Returns:
            None。
        """
        ...

    def set_permission_mode(self, mode: PermissionMode) -> None:
        """更新界面中展示的当前权限模式。

        Args:
            mode: 权限控制器当前生效的模式。

        Returns:
            None。
        """
        ...

    def set_busy(self, busy: bool) -> None:
        """设置是否禁止用户同时提交另一条输入。

        Args:
            busy: True 表示前台任务正在运行。

        Returns:
            None。
        """
        ...

    def begin_turn(self, user_text: str) -> None:
        """在对话区显示一轮新的用户输入。

        Args:
            user_text: 界面应展示的用户原文或短命令。

        Returns:
            None。
        """
        ...

    def render_event(self, event: AgentEvent) -> None:
        """把 Agent 产生的一个结构化事件更新到界面。

        Args:
            event: 文字增量、工具状态、最终回复或错误事件。

        Returns:
            None。
        """
        ...

    def show_status(self, message: str) -> None:
        """在对话区显示一条可恢复的状态消息。

        Args:
            message: 需要向用户说明的状态文字。

        Returns:
            None。
        """
        ...

    def show_error(self, message: str) -> None:
        """在对话区显示一条未终止应用的错误。

        Args:
            message: 已可以安全展示给用户的错误文字。

        Returns:
            None。
        """
        ...

    def clear_transcript(self) -> None:
        """清空当前界面可见的对话记录。

        Returns:
            None。
        """
        ...

    def end_turn(self) -> None:
        """结束当前回合的流式显示并恢复普通输入。

        Returns:
            None。
        """
        ...

    async def request_permission(
        self,
        request: PermissionRequest,
    ) -> ApprovalChoice:
        """显示一次工具权限确认并等待用户选择。

        Args:
            request: 要确认的工具、操作摘要和候选选项。

        Returns:
            用户选中的拒绝或放行方式。
        """
        ...

    async def confirm(self, message: str) -> bool:
        """显示普通是否确认并等待用户回答。

        Args:
            message: 要向用户展示的确认问题。

        Returns:
            用户确认时返回 True，否则返回 False。
        """
        ...


def supports_fullscreen(
) -> bool:
    """判断当前输入输出是否适合由全屏应用接管。

    Returns:
        输入输出都是 TTY 且终端不是 dumb 时返回 True。
    """

    input_isatty = getattr(sys.stdin, "isatty", None)
    output_isatty = getattr(sys.stdout, "isatty", None)
    if not callable(input_isatty) or not callable(output_isatty):
        return False
    if not input_isatty() or not output_isatty():
        return False
    return os.environ.get("TERM") != "dumb"


def create_terminal_ui(
    *,
    console: Console | None = None,
    prompt_session: Any | None = None,
    input: Input | None = None,
    output: Output | None = None,
    mode: TerminalUIMode | None = None,
    secrets: Iterable[SecretValue] = (),
    command_completer: CommandCompleter | None = None,
) -> TerminalUI:
    """根据指定模式或当前终端能力创建实际使用的界面对象。

    Args:
        console: 普通行式 UI 使用的 Rich 输出对象。
        prompt_session: 外部提供的行式输入会话。
        input: 全屏 UI 使用的 prompt-toolkit 输入对象。
        output: 全屏 UI 使用的 prompt-toolkit 输出对象。
        mode: 强制选择 fullscreen 或 plain；None 表示自动检测。
        secrets: 所有用户可见错误中需要隐藏的密钥。
        command_completer: 由冻结命令注册表创建的斜杠命令补全器。

    Returns:
        已选定并配置好输入依赖的全屏或普通行式 UI。
    """

    if mode not in {None, "fullscreen", "plain"}:
        raise ValueError("终端 UI 模式只支持 fullscreen 或 plain")
    if mode is None:
        # 显式传入普通输出对象时直接使用 PlainUI；正常启动则检查终端能力。
        selected: TerminalUIMode = (
            "plain"
            if console is not None or prompt_session is not None
            else "fullscreen" if supports_fullscreen() else "plain"
        )
    else:
        selected = mode
    if selected == "fullscreen":
        return FullscreenUI(
            input=input,
            output=output,
            secrets=secrets,
            command_completer=command_completer,
        )
    return PlainUI(
        console=console,
        prompt_session=prompt_session,
        secrets=secrets,
        command_completer=command_completer,
    )
