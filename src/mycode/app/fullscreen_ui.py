"""基于 prompt-toolkit 的全屏终端交互界面。"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import (
    BufferControl,
    ConditionalContainer,
    DynamicContainer,
    FormattedTextControl,
    HSplit,
    Layout,
    ScrollablePane,
    VSplit,
    Window,
)
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style

from mycode.app.tool_summary import (
    safe_summary_text,
    summarize_tool_result,
    summarize_tool_start,
)
from mycode.app.ui_models import (
    ConfirmationPromptState,
    DisplayKind,
    PermissionPromptState,
    TerminalStatus,
    ToolDisplay,
    ToolDisplayStatus,
    TranscriptBuffer,
    TranscriptEntry,
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


_MIN_COLUMNS = 60
_MIN_ROWS = 12
_PERMISSION_CHOICES = (
    ("拒绝", ApprovalChoice.DENY),
    ("本次允许", ApprovalChoice.ALLOW_ONCE),
    ("本会话允许", ApprovalChoice.ALLOW_SESSION),
    ("永久允许", ApprovalChoice.ALLOW_PERMANENT),
)
_CONFIRMATION_CHOICES = (("取消", False), ("确认", True))
_STYLE = Style.from_dict(
    {
        "screen": "bg:#151515 #e6e6e6",
        "welcome": "fg:#888888",
        "welcome.title": "fg:#5fd7d7 bold",
        "user.marker": "fg:#5fd7d7 bold",
        "user": "fg:#f0f0f0 bold",
        "thinking.marker": "fg:#9d7cff",
        "thinking": "fg:#8f86aa italic",
        "answer": "fg:#e6e6e6",
        "tool.running": "fg:#858585",
        "tool.success": "fg:#56d38b",
        "tool.failure": "fg:#ff6b6b bold",
        "status.marker": "fg:#5fd7d7",
        "status": "fg:#8fa9b5",
        "warning.marker": "fg:#ffd75f bold",
        "warning": "fg:#ffd75f",
        "error.marker": "fg:#ff6b6b bold",
        "error": "fg:#ff8a8a",
        "separator": "fg:#343434 bg:#151515",
        "input": "fg:#f0f0f0 bg:#191919",
        "input.prompt": "fg:#5fd7d7 bg:#191919 bold",
        "input.prompt.plan": "fg:#9d7cff bg:#191919 bold",
        "input.busy": "fg:#777777 bg:#191919",
        "statusbar": "fg:#a5a5a5 bg:#242424",
        "statusbar.plan": "fg:#9d7cff bg:#242424 bold",
        "statusbar.permission": "fg:#5fd7d7 bg:#242424",
        "statusbar.busy": "fg:#d7af5f bg:#242424 bold",
        "permission": "fg:#e6e6e6 bg:#242424",
        "permission.title": "fg:#d7af5f bg:#242424 bold",
        "permission.selected": "fg:#151515 bg:#d7af5f bold",
        "permission.option": "fg:#c8c8c8 bg:#242424",
        "small.title": "fg:#d7af5f bold",
        "small": "fg:#a5a5a5",
    }
)


class _TranscriptControl(FormattedTextControl):
    """把转轮事件交给 transcript 自己的滚动状态。"""

    def __init__(
        self,
        *,
        text: Any,
        scroll_up: Any,
        scroll_down: Any,
    ) -> None:
        super().__init__(text=text, focusable=False, show_cursor=False)
        self._scroll_up_callback = scroll_up
        self._scroll_down_callback = scroll_down

    def mouse_handler(self, mouse_event: Any) -> Any:
        if mouse_event.event_type is MouseEventType.SCROLL_UP:
            self._scroll_up_callback()
            return None
        if mouse_event.event_type is MouseEventType.SCROLL_DOWN:
            self._scroll_down_callback()
            return None
        return super().mouse_handler(mouse_event)


class _BoundedScrollablePane(ScrollablePane):
    """记录真实可滚动范围，并在每次绘制前约束滚动位置。"""

    def __init__(self, content: Any) -> None:
        super().__init__(
            content,
            keep_cursor_visible=False,
            keep_focused_window_visible=False,
            show_scrollbar=False,
            display_arrows=False,
        )
        self.content_height = 0
        self.viewport_height = 0

    @property
    def max_scroll(self) -> int:
        return max(self.content_height - self.viewport_height, 0)

    def write_to_screen(
        self,
        screen: Any,
        mouse_handlers: Any,
        write_position: Any,
        parent_style: str,
        erase_bg: bool,
        z_index: int | None,
    ) -> None:
        virtual_width = write_position.width - (
            1 if self.show_scrollbar() else 0
        )
        preferred = self.content.preferred_height(
            max(virtual_width, 0),
            self.max_available_height,
        ).preferred
        self.content_height = min(
            max(preferred, write_position.height),
            self.max_available_height,
        )
        self.viewport_height = write_position.height
        self.vertical_scroll = min(
            max(self.vertical_scroll, 0),
            self.max_scroll,
        )
        super().write_to_screen(
            screen,
            mouse_handlers,
            write_position,
            parent_style,
            erase_bg,
            z_index,
        )


class FullscreenUI:
    """在 prompt-toolkit 全屏应用中显示对话、状态和输入框。"""

    def __init__(
        self,
        *,
        input: Input | None = None,
        output: Output | None = None,
        secrets: Iterable[SecretValue] = (),
        command_completer: CommandCompleter | None = None,
    ) -> None:
        """创建全屏 UI 使用的缓冲区、布局和按键绑定。

        Args:
            input: 测试或终端提供的 prompt-toolkit 输入对象。
            output: 测试或终端提供的 prompt-toolkit 输出对象。
            secrets: 展示错误或工具摘要时需要隐藏的密钥。
            command_completer: 从冻结命令注册表读取候选的补全器。

        Returns:
            None。
        """

        self._secrets = tuple(secrets)
        # Tab 按键与候选菜单共同使用的斜杠命令补全器
        self._command_completer = command_completer
        self._output = output
        self._actions: asyncio.Queue[UIAction] = asyncio.Queue()
        self._permission_lock = asyncio.Lock()
        self._permission_state: (
            PermissionPromptState | ConfirmationPromptState | None
        ) = None
        self._permission_future: asyncio.Future[object] | None = None
        self._configured = False
        self._running = False
        self._stopping = False
        self._entry_cache: dict[
            str, tuple[int, StyleAndTextTuples]
        ] = {}
        self.transcript = TranscriptBuffer()
        self.status: TerminalStatus | None = None

        self._input_buffer = Buffer(
            completer=command_completer,
            multiline=False,
            read_only=Condition(self._input_is_read_only),
        )
        self._transcript_control = _TranscriptControl(
            text=self._formatted_transcript,
            scroll_up=self._scroll_up,
            scroll_down=self._scroll_down,
        )
        self._transcript_window = Window(
            content=self._transcript_control,
            wrap_lines=True,
            always_hide_cursor=True,
            dont_extend_height=True,
            style="class:screen",
        )
        self._transcript_pane = _BoundedScrollablePane(
            self._transcript_window
        )
        self._input_control = BufferControl(buffer=self._input_buffer)
        self._input_window = Window(
            content=self._input_control,
            height=1,
            style="class:input",
        )
        self._completion_menu = CompletionsMenu(
            max_height=8,
            extra_filter=Condition(self._has_multiple_completions),
        )
        self._permission_control = FormattedTextControl(
            text=self._formatted_permission,
            focusable=True,
            show_cursor=False,
        )
        self._permission_window = Window(
            content=self._permission_control,
            height=3,
            wrap_lines=True,
            always_hide_cursor=True,
            style="class:permission",
        )
        self._status_control = FormattedTextControl(
            text=self._formatted_status,
            show_cursor=False,
        )
        self._status_window = Window(
            content=self._status_control,
            height=1,
            style="class:statusbar",
            always_hide_cursor=True,
        )

        self._input_row = self._build_input_row()
        self._main_container = self._build_main_container()
        self._small_container = self._build_small_container()
        self._root = DynamicContainer(self._active_container)
        self._key_bindings = self._build_key_bindings()
        self._application: Application[None] = Application(
            layout=Layout(self._root, focused_element=self._input_control),
            key_bindings=self._key_bindings,
            style=_STYLE,
            full_screen=True,
            mouse_support=True,
            min_redraw_interval=0.01,
            max_render_postpone_time=0.02,
            terminal_size_polling_interval=0.25,
            input=input,
            output=output,
        )

    def _build_main_container(self) -> HSplit:
        permission = ConditionalContainer(
            content=HSplit(
                [
                    Window(
                        height=1,
                        char="─",
                        style="class:separator",
                    ),
                    self._permission_window,
                ]
            ),
            filter=Condition(lambda: self._permission_state is not None),
        )
        return HSplit(
            [
                self._transcript_pane,
                permission,
                Window(height=1, char="─", style="class:separator"),
                self._input_row,
                self._completion_menu,
                self._status_window,
            ]
        )

    def _build_input_row(self) -> VSplit:
        return VSplit(
            [
                Window(
                    FormattedTextControl(self._formatted_input_prompt),
                    width=Dimension(min=2, max=9),
                    height=1,
                    style="class:input",
                ),
                self._input_window,
            ],
            height=1,
        )

    def _build_small_container(self) -> HSplit:
        message = FormattedTextControl(
            lambda: [
                ("class:small.title", "终端窗口太小\n"),
                (
                    "class:small",
                    f"请调整到至少 {_MIN_COLUMNS}×{_MIN_ROWS}；"
                    "Ctrl+C 取消，Ctrl+D 退出。",
                ),
            ],
            show_cursor=False,
        )
        return HSplit(
            [
                Window(
                    message,
                    wrap_lines=True,
                    style="class:screen",
                ),
                Window(height=1, char="─", style="class:separator"),
                self._input_row,
                self._completion_menu,
                self._status_window,
            ]
        )

    def _active_container(self) -> Any:
        return (
            self._small_container
            if self._is_too_small()
            else self._main_container
        )

    def _has_multiple_completions(self) -> bool:
        """判断当前补全状态是否包含多个候选。

        Returns:
            有两个及以上候选时为 True，否则为 False。
        """

        state = self._input_buffer.complete_state
        return state is not None and len(state.completions) > 1

    def _terminal_size(self) -> tuple[int, int]:
        application = getattr(self, "_application", None)
        output = (
            application.output
            if application is not None
            else self._output
        )
        if output is None:
            return 80, 24
        size = output.get_size()
        return size.columns, size.rows

    def _is_too_small(self) -> bool:
        columns, rows = self._terminal_size()
        return columns < _MIN_COLUMNS or rows < _MIN_ROWS

    def _input_is_read_only(self) -> bool:
        return (
            self.status is not None
            and self.status.busy
        ) or self._permission_state is not None

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
        self.transcript.append_welcome(
            f"MyCode · {provider.name}/{provider.model}"
        )
        self.transcript.append_welcome(
            "输入消息开始；输入 /help 查看斜杠命令"
        )
        self._configured = True
        self._invalidate()

    async def run_async(self) -> None:
        if not self._configured:
            raise RuntimeError("UI 必须先配置再运行")
        if self._running:
            raise RuntimeError("全屏 UI 不能重复运行")
        self._running = True
        try:
            await self._application.run_async()
        finally:
            self._running = False
            self._deny_pending_permission()

    async def next_action(self) -> UIAction:
        return await self._actions.get()

    def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        self._deny_pending_permission()
        if self._running and not self._application.is_done:
            self._application.exit()
        if self._actions.empty():
            self._actions.put_nowait(UIAction(UIActionKind.EXIT))

    def _deny_pending_permission(self) -> None:
        future = self._permission_future
        if future is not None and not future.done():
            future.set_result(
                False
                if isinstance(self._permission_state, ConfirmationPromptState)
                else ApprovalChoice.DENY
            )

    def _prompt_choices(self) -> tuple[tuple[str, object], ...]:
        if isinstance(self._permission_state, ConfirmationPromptState):
            return _CONFIRMATION_CHOICES
        return _PERMISSION_CHOICES

    def _invalidate(self) -> None:
        if self.transcript.follow_tail:
            self._scroll_to_tail()
        if self._running:
            self._application.invalidate()

    def _scroll_to_tail(self) -> None:
        # 下一次绘制会依据真实换行后的高度约束到末尾。
        self._transcript_pane.vertical_scroll = 10**9

    def _scroll_limit(self) -> int:
        return self._transcript_pane.max_scroll

    def _scroll_up(self) -> None:
        step = max(self._terminal_size()[1] // 2, 1)
        current = self._transcript_pane.vertical_scroll
        if current >= 10**8:
            current = self._scroll_limit()
        self._transcript_pane.vertical_scroll = max(current - step, 0)
        self.transcript.begin_history_view()
        self._application.invalidate()

    def _scroll_down(self) -> None:
        step = max(self._terminal_size()[1] // 2, 1)
        limit = self._scroll_limit()
        current = min(self._transcript_pane.vertical_scroll, limit)
        target = min(current + step, limit)
        self._transcript_pane.vertical_scroll = target
        if target >= limit:
            self.transcript.resume_follow_tail()
        self._application.invalidate()

    def _scroll_home(self) -> None:
        self._transcript_pane.vertical_scroll = 0
        self.transcript.begin_history_view()
        self._application.invalidate()

    def _scroll_end(self) -> None:
        self.transcript.resume_follow_tail()
        self._scroll_to_tail()
        self._application.invalidate()

    def _build_key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add(
            Keys.Enter,
            filter=Condition(lambda: self._permission_state is not None),
            eager=True,
        )
        def confirm_permission(event: Any) -> None:
            del event
            self._resolve_permission(
                self._prompt_choices()[
                    self._permission_state.selected_index  # type: ignore[union-attr]
                ][1]
            )

        @bindings.add(
            Keys.Enter,
            filter=Condition(lambda: self._permission_state is None),
        )
        def submit(event: Any) -> None:
            del event
            if self.status is None or self.status.busy:
                return
            text = self._input_buffer.text.strip()
            if not text:
                return
            self._input_buffer.reset()
            self._actions.put_nowait(
                UIAction(UIActionKind.SUBMIT, text)
            )

        @bindings.add(Keys.BackTab)
        def toggle_plan(event: Any) -> None:
            del event
            if self._permission_state is not None:
                return
            if self.status is not None and self.status.busy:
                self.show_status("当前轮次执行中，暂不能切换 Plan 模式")
                return
            self._actions.put_nowait(UIAction(UIActionKind.TOGGLE_PLAN))

        @bindings.add(Keys.Tab)
        def complete_command(event: Any) -> None:
            del event
            if self._permission_state is not None:
                return
            if self.status is not None and self.status.busy:
                return
            if self._command_completer is None:
                return
            if self._input_buffer.complete_state is not None:
                self._input_buffer.complete_next()
                return
            completions = list(
                self._command_completer.get_completions(
                    self._input_buffer.document,
                    CompleteEvent(completion_requested=True),
                )
            )
            if len(completions) == 1:
                self._input_buffer.apply_completion(completions[0])
            elif len(completions) > 1:
                self._input_buffer.start_completion(select_first=True)

        @bindings.add(Keys.ControlC)
        def cancel(event: Any) -> None:
            del event
            if self._permission_state is not None:
                self._deny_pending_permission()
                return
            if self.status is not None and self.status.busy:
                self._actions.put_nowait(UIAction(UIActionKind.CANCEL))
                return
            self._input_buffer.reset()
            self.show_status("已取消当前输入")

        @bindings.add(Keys.Escape)
        def adopt_background(event: Any) -> None:
            """把 ESC 转成前台子 Agent 移交请求，不代替 Ctrl+C 取消。

            Args:
                event: prompt-toolkit 传入的按键事件，本函数不读取其字段。

            Returns:
                不返回数据；忙碌时向应用队列加入 ADOPT_BACKGROUND，空闲时
                只显示当前没有可移交任务。
            """

            del event
            if self.status is not None and self.status.busy:
                self._actions.put_nowait(
                    UIAction(UIActionKind.ADOPT_BACKGROUND)
                )
                return
            if self._permission_state is not None:
                self.show_status("权限确认中，请先选择或取消当前确认")
                return
            self.show_status("当前没有可移交到后台的子 Agent")

        @bindings.add(Keys.ControlD)
        def exit_or_deny(event: Any) -> None:
            del event
            if self._permission_state is not None:
                self._deny_pending_permission()
                return
            self._actions.put_nowait(UIAction(UIActionKind.EXIT))

        for number in range(1, 5):
            @bindings.add(
                str(number),
                filter=Condition(
                    lambda: self._permission_state is not None
                ),
                eager=True,
            )
            def choose(
                event: Any,
                selected_index: int = number - 1,
            ) -> None:
                del event
                choices = self._prompt_choices()
                if selected_index < len(choices):
                    self._resolve_permission(choices[selected_index][1])

        @bindings.add(
            Keys.Left,
            filter=Condition(lambda: self._permission_state is not None),
            eager=True,
        )
        def previous_choice(event: Any) -> None:
            del event
            assert self._permission_state is not None
            self._permission_state.selected_index = (
                self._permission_state.selected_index - 1
            ) % len(self._prompt_choices())
            self._application.invalidate()

        @bindings.add(
            Keys.Right,
            filter=Condition(lambda: self._permission_state is not None),
            eager=True,
        )
        def next_choice(event: Any) -> None:
            del event
            assert self._permission_state is not None
            self._permission_state.selected_index = (
                self._permission_state.selected_index + 1
            ) % len(self._prompt_choices())
            self._application.invalidate()

        @bindings.add(Keys.PageUp)
        def page_up(event: Any) -> None:
            del event
            self._scroll_up()

        @bindings.add(Keys.PageDown)
        def page_down(event: Any) -> None:
            del event
            self._scroll_down()

        @bindings.add(Keys.ControlHome)
        def home(event: Any) -> None:
            del event
            self._scroll_home()

        @bindings.add(Keys.ControlEnd)
        def end(event: Any) -> None:
            del event
            self._scroll_end()

        return bindings

    def _resolve_permission(self, choice: object) -> None:
        future = self._permission_future
        if future is not None and not future.done():
            future.set_result(choice)

    def _formatted_input_prompt(self) -> StyleAndTextTuples:
        if self.status is not None and self.status.busy:
            return [("class:input.busy", "… ")]
        if self.status is not None and self.status.plan_only:
            return [("class:input.prompt.plan", "[PLAN] › ")]
        return [("class:input.prompt", "[DEFAULT] › ")]

    def _scroll_status(self) -> str:
        limit = self._transcript_pane.max_scroll
        if limit <= 0:
            return "all"
        position = min(
            max(self._transcript_pane.vertical_scroll, 0),
            limit,
        )
        if self.transcript.follow_tail or position >= limit:
            return "bottom"
        return f"{round(position * 100 / limit)}%"

    def _formatted_status(self) -> StyleAndTextTuples:
        if self.status is None:
            return [("class:statusbar", " MyCode")]
        fragments: StyleAndTextTuples = []
        if self.status.busy:
            fragments.append(("class:statusbar.busy", " Working "))
        mode_hint = (
            "[PLAN] /do 切换到执行模式"
            if self.status.plan_only
            else "[DEFAULT] /help 查看命令"
        )
        fragments.extend(
            [
                ("class:statusbar.plan", f" {mode_hint} "),
                (
                    "class:statusbar.permission",
                    f" Permission {self.status.permission_mode.value} ",
                ),
                (
                    "class:statusbar",
                    f" {self.status.provider_name}/{self.status.model_name} ",
                ),
                (
                    "class:statusbar",
                    f" Scroll {self._scroll_status()} · PgUp/PgDn · "
                    "Ctrl+C Cancel ",
                ),
            ]
        )
        return fragments

    def _formatted_permission(self) -> StyleAndTextTuples:
        state = self._permission_state
        if state is None:
            return []
        if isinstance(state, ConfirmationPromptState):
            title = " ? 请确认: "
            detail = safe_summary_text(state.message, self._secrets, limit=160)
        else:
            title = f" ? 权限确认 · {state.request.operation.tool.value}: "
            detail = safe_summary_text(
                state.request.operation.display_value,
                self._secrets,
                limit=160,
            )
        fragments: StyleAndTextTuples = [
            ("class:permission.title", title),
            ("class:permission", detail + "\n"),
        ]
        for index, (label, _) in enumerate(self._prompt_choices()):
            style = (
                "class:permission.selected"
                if index == state.selected_index
                else "class:permission.option"
            )
            fragments.append((style, f" [{index + 1}] {label} "))
        fragments.append(
            (
                "class:permission",
                "\n  数字键直接选择，或使用 ← → 后按 Enter",
            )
        )
        return fragments

    def _formatted_transcript(self) -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = []
        live_keys = set()
        for index, entry in enumerate(self.transcript.entries):
            live_keys.add(entry.key)
            cached = self._entry_cache.get(entry.key)
            if cached is None or cached[0] != entry.revision:
                rendered = self._format_entry(entry)
                self._entry_cache[entry.key] = (
                    entry.revision,
                    rendered,
                )
            else:
                rendered = cached[1]
            fragments.extend(rendered)
            if index != len(self.transcript.entries) - 1:
                fragments.append(("", "\n\n"))
        for key in tuple(self._entry_cache):
            if key not in live_keys:
                del self._entry_cache[key]
        return fragments

    def _format_entry(
        self,
        entry: TranscriptEntry,
    ) -> StyleAndTextTuples:
        text = sanitize_terminal_text(entry.text)
        if entry.kind is DisplayKind.WELCOME:
            style = (
                "class:welcome.title"
                if entry.key.endswith("-1")
                else "class:welcome"
            )
            return [(style, text)]
        if entry.kind is DisplayKind.USER:
            return [
                ("class:user.marker", "› "),
                ("class:user", text),
            ]
        if entry.kind is DisplayKind.THINKING:
            return [
                ("class:thinking.marker", "● "),
                ("class:thinking", text),
            ]
        if entry.kind is DisplayKind.ANSWER:
            return [("class:answer", "  " + text)]
        if entry.kind is DisplayKind.STATUS:
            return [
                ("class:status.marker", "◆ "),
                ("class:status", text),
            ]
        if entry.kind is DisplayKind.WARNING:
            return [
                ("class:warning.marker", "⚠ "),
                ("class:warning", text),
            ]
        if entry.kind is DisplayKind.ERROR:
            return [
                ("class:error.marker", "! "),
                ("class:error", text),
            ]
        if entry.kind is DisplayKind.TOOL and entry.tool is not None:
            return self._format_tool(entry.tool)
        return [("", text)]

    def _format_tool(self, tool: ToolDisplay) -> StyleAndTextTuples:
        if tool.status is ToolDisplayStatus.RUNNING:
            marker = "·"
            style = "class:tool.running"
        elif tool.status is ToolDisplayStatus.SUCCESS:
            marker = "✓"
            style = "class:tool.success"
        else:
            marker = "×"
            style = "class:tool.failure"
        text = f"{marker} {tool.label}"
        if tool.target:
            text += f": {tool.target}"
        if tool.detail:
            text += f" ({tool.detail})"
        if tool.duration_ms is not None:
            text += f" ({tool.duration_ms / 1000:.1f}s)"
        return [(style, text)]

    def set_plan_mode(self, enabled: bool) -> None:
        if self.status is None:
            raise RuntimeError("UI 尚未配置")
        self.status.plan_only = enabled
        self._invalidate()

    def set_permission_mode(self, mode: PermissionMode) -> None:
        if self.status is None:
            raise RuntimeError("UI 尚未配置")
        self.status.permission_mode = mode
        self._invalidate()

    def set_busy(self, busy: bool) -> None:
        if self.status is None:
            raise RuntimeError("UI 尚未配置")
        self.status.busy = busy
        if not busy and self._permission_state is None:
            try:
                self._application.layout.focus(self._input_control)
            except ValueError:
                pass
        self._invalidate()

    def begin_turn(self, user_text: str) -> None:
        self.transcript.append_user(user_text)
        self.transcript.resume_follow_tail()
        self._invalidate()

    def render_event(self, event: AgentEvent) -> None:
        if isinstance(event, UserMessageEvent):
            return
        if isinstance(event, FinalReplyEvent):
            self.transcript.finish_stream()
        elif isinstance(event, AgentErrorEvent):
            self.transcript.append_error(event.message)
        elif isinstance(event, AgentWarningEvent):
            self.transcript.append_warning(event.message)
        elif isinstance(event, CompactionStatusEvent):
            self.transcript.append_status(event.message)
        elif isinstance(event, ToolStartedEvent):
            self.transcript.start_tool(
                summarize_tool_start(event.invocation, self._secrets)
            )
        elif isinstance(event, ToolResultEvent):
            completed = summarize_tool_result(
                event.invocation,
                event.result,
                self._secrets,
            )
            try:
                self.transcript.finish_tool(completed)
            except ValueError:
                self.transcript.start_tool(
                    summarize_tool_start(event.invocation, self._secrets)
                )
                self.transcript.finish_tool(completed)
        elif isinstance(event, ThinkingDeltaEvent):
            self.transcript.append_stream(
                DisplayKind.THINKING,
                event.model_call_number,
                event.text,
            )
        elif isinstance(event, ModelTextDeltaEvent):
            self.transcript.append_stream(
                DisplayKind.ANSWER,
                event.model_call_number,
                event.text,
            )
        self._invalidate()

    def show_status(self, message: str) -> None:
        self.transcript.append_status(message)
        self._invalidate()

    def show_error(self, message: str) -> None:
        self.transcript.append_error(message)
        self._invalidate()

    def clear_transcript(self) -> None:
        self.transcript.clear()
        self._entry_cache.clear()
        self._transcript_pane.vertical_scroll = 0
        self._invalidate()

    def end_turn(self) -> None:
        self.transcript.finish_stream()
        self._invalidate()

    async def request_permission(
        self,
        request: PermissionRequest,
    ) -> ApprovalChoice:
        async with self._permission_lock:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self._permission_state = PermissionPromptState(request)
            self._permission_future = future
            try:
                self._application.layout.focus(self._permission_window)
            except ValueError:
                pass
            self._invalidate()
            try:
                result = await future
                assert isinstance(result, ApprovalChoice)
                return result
            except asyncio.CancelledError:
                return ApprovalChoice.DENY
            except Exception:
                return ApprovalChoice.DENY
            finally:
                self._permission_state = None
                self._permission_future = None
                try:
                    self._application.layout.focus(self._input_control)
                except ValueError:
                    pass
                self._invalidate()

    async def confirm(self, message: str) -> bool:
        """在全屏提示栏显示默认选中“取消”的通用确认。"""

        async with self._permission_lock:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self._permission_state = ConfirmationPromptState(message)
            self._permission_future = future
            try:
                self._application.layout.focus(self._permission_window)
            except ValueError:
                pass
            self._invalidate()
            try:
                result = await future
                return result is True
            except (asyncio.CancelledError, Exception):
                return False
            finally:
                self._permission_state = None
                self._permission_future = None
                try:
                    self._application.layout.focus(self._input_control)
                except ValueError:
                    pass
                self._invalidate()
