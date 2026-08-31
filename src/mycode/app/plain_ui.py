"""非全屏终端兼容渲染器。"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterable
from typing import Any

from prompt_toolkit import PromptSession
from rich.console import Console

from mycode.app.tool_summary import (
    safe_summary_text,
    summarize_tool_result,
    summarize_tool_start,
)
from mycode.app.ui_models import (
    DisplayKind,
    TerminalStatus,
    TranscriptBuffer,
    UIAction,
    UIActionKind,
    sanitize_terminal_text,
)
from mycode.commands.completion import CommandCompleter
from mycode.models.config import ProviderConfig, SecretValue
from mycode.models.events import (
    AgentErrorEvent,
    AgentEvent,
    AgentWarningEvent,
    CompactionStatusEvent,
    FinalReplyEvent,
    ModelTextDeltaEvent,
    ThinkingDeltaEvent,
    ToolResultEvent,
    ToolStartedEvent,
    UserMessageEvent,
)
from mycode.models.permissions import (
    ApprovalChoice,
    PermissionMode,
    PermissionRequest,
)


class PlainUI:
    """在普通输出流中逐行呈现与全屏模式相同的语义。"""

    def __init__(
        self,
        *,
        console: Console | None = None,
        prompt_session: Any | None = None,
        secrets: Iterable[SecretValue] = (),
        command_completer: CommandCompleter | None = None,
    ) -> None:
        """创建逐行输出界面，并在交互式终端启用命令补全。

        Args:
            console: 接收 Rich 文本输出的控制台对象。
            prompt_session: 外部提供的 prompt-toolkit 输入会话。
            secrets: 展示错误或工具摘要时需要隐藏的密钥。
            command_completer: 从冻结命令注册表读取候选的补全器。

        Returns:
            None。
        """

        self._console = console or Console()
        # 外部输入会话优先；真实交互终端在有命令补全器时创建 PromptSession
        self._prompt_session = prompt_session
        if (
            self._prompt_session is None
            and command_completer is not None
            and getattr(sys.stdin, "isatty", lambda: False)()
            and self._console.is_terminal
        ):
            self._prompt_session = PromptSession(
                completer=command_completer,
                complete_while_typing=False,
            )
        # 外部 PromptSession 的每次 prompt 调用也会收到这一个补全器
        self._command_completer = command_completer
        self._secrets = tuple(secrets)
        self._permission_lock = asyncio.Lock()
        self._stopped = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._configured = False
        self._welcome_shown = False
        self._current_stream: tuple[DisplayKind, int] | None = None
        self.transcript = TranscriptBuffer()
        self.status: TerminalStatus | None = None

    def configure(
        self,
        provider: ProviderConfig,
        permission_mode: PermissionMode,
    ) -> None:
        self.status = TerminalStatus(
            provider.name,
            provider.model,
            False,
            permission_mode,
        )
        self._configured = True

    async def run_async(self) -> None:
        if not self._configured or self.status is None:
            raise RuntimeError("UI 必须先配置再运行")
        if not self._welcome_shown:
            self._show_welcome()
        await self._stopped.wait()

    def stop(self) -> None:
        self._finish_open_line()
        self._stopped.set()
        self._idle.set()

    def _show_welcome(self) -> None:
        assert self.status is not None
        title = (
            f"MyCode · {self.status.provider_name}/{self.status.model_name}"
        )
        hint = (
            "› 输入消息；输入 /help 查看斜杠命令"
        )
        self.transcript.append_welcome(title)
        self.transcript.append_welcome(hint)
        self._console.print(title, style="bold cyan", markup=False)
        self._console.print(hint, style="dim", markup=False)
        self._welcome_shown = True

    def _flush(self) -> None:
        flush = getattr(self._console.file, "flush", None)
        if callable(flush):
            flush()

    def _finish_open_line(self) -> None:
        if self._current_stream is not None:
            self._console.print()
            self._current_stream = None
            self.transcript.finish_stream()

    async def _prompt_async(self, prompt: str) -> str:
        """从 PromptSession 或系统 input 读取一行原始输入。

        Args:
            prompt: 显示在用户输入位置之前的模式提示。

        Returns:
            用户提交且尚未去除首尾空白的原始文字。
        """

        if self._prompt_session is None:
            return await asyncio.to_thread(input, prompt)
        method = getattr(self._prompt_session, "prompt_async", None)
        if callable(method):
            if self._command_completer is None:
                return await method(prompt)
            return await method(
                prompt,
                completer=self._command_completer,
                complete_while_typing=False,
            )
        if self._command_completer is None:
            return await asyncio.to_thread(self._prompt_session.prompt, prompt)
        return await asyncio.to_thread(
            self._prompt_session.prompt,
            prompt,
            completer=self._command_completer,
            complete_while_typing=False,
        )

    async def next_action(self) -> UIAction:
        if self.status is None:
            raise RuntimeError("UI 尚未配置")
        while True:
            await self._idle.wait()
            prompt = (
                "[PLAN] /do 切换到执行模式 › "
                if self.status.plan_only
                else "[DEFAULT] /help 查看命令 › "
            )
            try:
                raw = await self._prompt_async(prompt)
            except KeyboardInterrupt:
                self.show_status("已取消当前输入")
                continue
            except EOFError:
                return UIAction(UIActionKind.EXIT)
            text = raw.strip()
            if text:
                return UIAction(UIActionKind.SUBMIT, text)

    def set_plan_mode(self, enabled: bool) -> None:
        if self.status is None:
            raise RuntimeError("UI 尚未配置")
        self.status.plan_only = enabled

    def set_permission_mode(self, mode: PermissionMode) -> None:
        if self.status is None:
            raise RuntimeError("UI 尚未配置")
        self.status.permission_mode = mode

    def set_busy(self, busy: bool) -> None:
        if self.status is None:
            raise RuntimeError("UI 尚未配置")
        self.status.busy = busy
        if busy:
            self._idle.clear()
        else:
            self._idle.set()

    def begin_turn(self, user_text: str) -> None:
        self._finish_open_line()
        cleaned = sanitize_terminal_text(user_text)
        self.transcript.append_user(cleaned)
        self._console.print(
            f"› {cleaned}",
            style="bold cyan",
            markup=False,
            highlight=False,
        )

    def render_event(self, event: AgentEvent) -> None:
        if isinstance(event, UserMessageEvent):
            return
        if isinstance(event, FinalReplyEvent):
            self._finish_open_line()
            return
        if isinstance(event, AgentErrorEvent):
            self.show_error(event.message)
            return
        if isinstance(event, AgentWarningEvent):
            self._finish_open_line()
            cleaned = sanitize_terminal_text(event.message)
            self.transcript.append_warning(cleaned)
            self._console.print(
                f"⚠ {cleaned}",
                style="bold yellow",
                markup=False,
                highlight=False,
            )
            return
        if isinstance(event, CompactionStatusEvent):
            self.show_status(event.message)
            return
        if isinstance(event, ToolStartedEvent):
            self._finish_open_line()
            display = summarize_tool_start(event.invocation, self._secrets)
            self.transcript.start_tool(display)
            self._console.print(
                self._tool_text("·", display),
                style="dim",
                markup=False,
                highlight=False,
            )
            return
        if isinstance(event, ToolResultEvent):
            self._finish_open_line()
            display = summarize_tool_result(
                event.invocation,
                event.result,
                self._secrets,
            )
            try:
                self.transcript.finish_tool(display)
            except ValueError:
                # 兼容第三方事件源直接发送结果而没有开始事件。
                running = summarize_tool_start(event.invocation, self._secrets)
                self.transcript.start_tool(running)
                self.transcript.finish_tool(display)
            marker = "✓" if display.status.value == "success" else "×"
            self._console.print(
                self._tool_text(marker, display),
                style=(
                    "green"
                    if display.status.value == "success"
                    else "bold red"
                ),
                markup=False,
                highlight=False,
            )
            return

        if isinstance(event, ThinkingDeltaEvent):
            kind = DisplayKind.THINKING
            prefix = "● "
            style = "dim magenta"
        elif isinstance(event, ModelTextDeltaEvent):
            kind = DisplayKind.ANSWER
            prefix = "  "
            style = ""
        else:
            return

        stream = (kind, event.model_call_number)
        if stream != self._current_stream:
            self._finish_open_line()
            self._console.print(
                prefix,
                end="",
                style=style,
                markup=False,
                highlight=False,
            )
            self._current_stream = stream
        self.transcript.append_stream(kind, event.model_call_number, event.text)
        self._console.file.write(sanitize_terminal_text(event.text))
        self._flush()

    def _tool_text(self, marker: str, display: Any) -> str:
        text = f"{marker} {display.label}"
        if display.target:
            text += f": {display.target}"
        if display.detail:
            text += f" ({display.detail})"
        if display.duration_ms is not None:
            text += f" ({display.duration_ms / 1000:.1f}s)"
        return text

    def show_status(self, message: str) -> None:
        self._finish_open_line()
        cleaned = sanitize_terminal_text(message)
        self.transcript.append_status(cleaned)
        self._console.print(
            f"◆ {cleaned}",
            style="cyan",
            markup=False,
            highlight=False,
        )

    def show_error(self, message: str) -> None:
        self._finish_open_line()
        cleaned = sanitize_terminal_text(message)
        self.transcript.append_error(cleaned)
        self._console.print(
            f"! {cleaned}",
            style="bold red",
            markup=False,
            highlight=False,
        )

    def clear_transcript(self) -> None:
        self._finish_open_line()
        self.transcript.clear()

    def end_turn(self) -> None:
        self._finish_open_line()

    async def request_permission(
        self,
        request: PermissionRequest,
    ) -> ApprovalChoice:
        async with self._permission_lock:
            self._finish_open_line()
            operation_text = safe_summary_text(
                request.operation.display_value,
                self._secrets,
            )
            self._console.print(
                f"? 权限确认 · {request.operation.tool.value}: {operation_text}",
                style="bold yellow",
                markup=False,
                highlight=False,
            )
            self._console.print(
                "[1] 拒绝  [2] 本次允许  [3] 本会话允许  [4] 永久允许",
                markup=False,
                highlight=False,
            )
            choices = {
                "1": ApprovalChoice.DENY,
                "2": ApprovalChoice.ALLOW_ONCE,
                "3": ApprovalChoice.ALLOW_SESSION,
                "4": ApprovalChoice.ALLOW_PERMANENT,
            }
            while True:
                try:
                    raw = await self._prompt_async("请选择：")
                except (KeyboardInterrupt, EOFError):
                    self.show_status("权限确认已取消，本次调用已拒绝")
                    return ApprovalChoice.DENY
                except Exception:
                    self.show_status("权限确认不可用，本次调用已拒绝")
                    return ApprovalChoice.DENY
                choice = choices.get(raw.strip())
                if choice is not None:
                    return choice
                self.show_status("请输入 1、2、3 或 4")

    async def confirm(self, message: str) -> bool:
        """显示默认拒绝的确认提示；输入 y 才返回确认。"""

        async with self._permission_lock:
            self._finish_open_line()
            cleaned = sanitize_terminal_text(message)
            try:
                raw = await self._prompt_async(f"? {cleaned} [y/N] ")
            except (KeyboardInterrupt, EOFError, Exception):
                return False
            return raw.strip().casefold() in {"y", "yes"}
