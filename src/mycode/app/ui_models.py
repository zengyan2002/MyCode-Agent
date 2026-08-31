"""终端 UI 的纯状态模型与不可信文本清理。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from mycode.models.permissions import PermissionMode, PermissionRequest


class DisplayKind(str, Enum):
    WELCOME = "welcome"
    USER = "user"
    THINKING = "thinking"
    ANSWER = "answer"
    TOOL = "tool"
    STATUS = "status"
    WARNING = "warning"
    ERROR = "error"


class ToolDisplayStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class UIActionKind(str, Enum):
    SUBMIT = "submit"
    TOGGLE_PLAN = "toggle_plan"
    CANCEL = "cancel"
    ADOPT_BACKGROUND = "adopt_background"
    EXIT = "exit"


def sanitize_terminal_text(text: str) -> str:
    """移除能控制终端的 C0/C1 字符，同时保留正常换行和制表符。"""

    return "".join(
        character
        for character in text
        if character in {"\n", "\t"}
        or not (ord(character) < 32 or 127 <= ord(character) <= 159)
    )


def single_line_display(text: str) -> str:
    """把不可信文本压成适合状态栏或工具摘要的一行。"""

    cleaned = sanitize_terminal_text(text)
    return " ".join(cleaned.split())


def truncate_display_text(text: str, limit: int) -> str:
    """按 Unicode 字符截断展示文本，并用省略号明确标记。"""

    if limit <= 0:
        raise ValueError("展示文本长度上限必须为正数")
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    return text[: limit - 1] + "…"


@dataclass
class ToolDisplay:
    call_id: str
    label: str
    target: str | None
    status: ToolDisplayStatus
    duration_ms: int | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.call_id or not self.label:
            raise ValueError("工具展示必须包含调用 ID 和名称")
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("工具展示耗时不能为负数")
        if self.status is ToolDisplayStatus.RUNNING and self.duration_ms is not None:
            raise ValueError("运行中的工具不能包含完成耗时")


@dataclass
class TranscriptEntry:
    key: str
    kind: DisplayKind
    text: str
    model_call_number: int | None = None
    tool: ToolDisplay | None = None
    revision: int = 0

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("消息记录必须包含稳定 key")
        if self.kind is DisplayKind.TOOL:
            if self.tool is None:
                raise ValueError("工具消息必须包含 ToolDisplay")
        elif self.tool is not None:
            raise ValueError("非工具消息不能包含 ToolDisplay")
        if self.kind in {DisplayKind.THINKING, DisplayKind.ANSWER}:
            if self.model_call_number is None or self.model_call_number <= 0:
                raise ValueError("流式消息必须包含有效模型调用序号")
        elif self.model_call_number is not None:
            raise ValueError("非流式消息不能包含模型调用序号")

    def append(self, text: str) -> None:
        self.text += text
        self.revision += 1

    def replace_tool(self, tool: ToolDisplay) -> None:
        if self.kind is not DisplayKind.TOOL:
            raise ValueError("只能更新工具消息")
        if self.tool is None or self.tool.call_id != tool.call_id:
            raise ValueError("工具调用 ID 与消息记录不匹配")
        self.tool = tool
        self.revision += 1


@dataclass
class TerminalStatus:
    provider_name: str
    model_name: str
    plan_only: bool
    permission_mode: PermissionMode
    busy: bool = False

    def __post_init__(self) -> None:
        if not self.provider_name or not self.model_name:
            raise ValueError("终端状态必须包含 Provider 和模型")


@dataclass(frozen=True)
class UIAction:
    kind: UIActionKind
    text: str | None = None

    def __post_init__(self) -> None:
        if self.kind is UIActionKind.SUBMIT:
            if self.text is None or not self.text.strip():
                raise ValueError("提交动作必须包含非空文本")
        elif self.text is not None:
            raise ValueError("只有提交动作可以携带文本")


@dataclass
class PermissionPromptState:
    request: PermissionRequest
    selected_index: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.selected_index < 4:
            raise ValueError("权限选项索引必须位于 0 到 3")


@dataclass
class ConfirmationPromptState:
    """保存通用确认提示及当前选中的“取消/确认”选项。"""

    message: str
    selected_index: int = 0

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("确认提示不能为空")
        if not 0 <= self.selected_index < 2:
            raise ValueError("确认选项索引必须位于 0 到 1")


@dataclass
class TranscriptBuffer:
    entries: list[TranscriptEntry] = field(default_factory=list)
    follow_tail: bool = True
    _sequence: int = 0
    _active_stream: tuple[DisplayKind, int, str] | None = None
    _tool_keys: dict[str, str] = field(default_factory=dict)

    def _key(self, prefix: str) -> str:
        self._sequence += 1
        return f"{prefix}-{self._sequence}"

    def _append_plain(self, kind: DisplayKind, text: str) -> TranscriptEntry:
        if kind in {
            DisplayKind.THINKING,
            DisplayKind.ANSWER,
            DisplayKind.TOOL,
        }:
            raise ValueError("该消息类型必须使用专用追加接口")
        self.finish_stream()
        entry = TranscriptEntry(
            self._key(kind.value),
            kind,
            sanitize_terminal_text(text),
        )
        self.entries.append(entry)
        return entry

    def append_welcome(self, text: str) -> TranscriptEntry:
        return self._append_plain(DisplayKind.WELCOME, text)

    def append_user(self, text: str) -> TranscriptEntry:
        return self._append_plain(DisplayKind.USER, text)

    def append_status(self, text: str) -> TranscriptEntry:
        return self._append_plain(DisplayKind.STATUS, text)

    def append_warning(self, text: str) -> TranscriptEntry:
        return self._append_plain(DisplayKind.WARNING, text)

    def append_error(self, text: str) -> TranscriptEntry:
        return self._append_plain(DisplayKind.ERROR, text)

    def append_stream(
        self,
        kind: Literal[DisplayKind.THINKING, DisplayKind.ANSWER],
        model_call_number: int,
        text: str,
    ) -> TranscriptEntry:
        if kind not in {DisplayKind.THINKING, DisplayKind.ANSWER}:
            raise ValueError("流式消息只支持思考或回答")
        if model_call_number <= 0:
            raise ValueError("模型调用序号必须为正数")
        cleaned = sanitize_terminal_text(text)
        if (
            self._active_stream is not None
            and self._active_stream[:2] == (kind, model_call_number)
        ):
            key = self._active_stream[2]
            entry = self._entry_by_key(key)
            entry.append(cleaned)
            return entry
        self.finish_stream()
        entry = TranscriptEntry(
            self._key(kind.value),
            kind,
            cleaned,
            model_call_number=model_call_number,
        )
        self.entries.append(entry)
        self._active_stream = (kind, model_call_number, entry.key)
        return entry

    def finish_stream(self) -> None:
        self._active_stream = None

    def start_tool(self, tool: ToolDisplay) -> TranscriptEntry:
        if tool.status is not ToolDisplayStatus.RUNNING:
            raise ValueError("工具开始记录必须处于运行状态")
        self.finish_stream()
        if tool.call_id in self._tool_keys:
            raise ValueError(f"工具调用已经存在：{tool.call_id}")
        entry = TranscriptEntry(
            self._key("tool"),
            DisplayKind.TOOL,
            "",
            tool=tool,
        )
        self.entries.append(entry)
        self._tool_keys[tool.call_id] = entry.key
        return entry

    def finish_tool(self, tool: ToolDisplay) -> TranscriptEntry:
        if tool.status is ToolDisplayStatus.RUNNING:
            raise ValueError("工具完成记录不能仍处于运行状态")
        key = self._tool_keys.get(tool.call_id)
        if key is None:
            raise ValueError(f"未知工具调用：{tool.call_id}")
        entry = self._entry_by_key(key)
        entry.replace_tool(tool)
        return entry

    def _entry_by_key(self, key: str) -> TranscriptEntry:
        for entry in reversed(self.entries):
            if entry.key == key:
                return entry
        raise ValueError(f"未知消息记录：{key}")

    def clear(self) -> None:
        self.entries.clear()
        self._active_stream = None
        self._tool_keys.clear()
        self.follow_tail = True

    def begin_history_view(self) -> None:
        self.follow_tail = False

    def resume_follow_tail(self) -> None:
        self.follow_tail = True
