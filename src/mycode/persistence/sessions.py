"""保存、列出、恢复和清理当前项目的 JSONL 会话；切换会话时同步内存消息与上下文摘要"""

from __future__ import annotations

import os
import re
import secrets
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TextIO

from mycode.agent.conversation import Conversation
from mycode.constants import SESSION_RETENTION_DAYS, SESSION_TITLE_MAX_CHARS
from mycode.context.manager import ContextManager
from mycode.errors import MyCodeError
from mycode.models.messages import (
    AssistantMessage,
    ChatMessage,
    ToolResultMessage,
    UserMessage,
)
from mycode.models.sessions import SessionRuntimeMetadata
from mycode.models.skills import SkillSessionState
from mycode.models.teams import TeamBinding
from mycode.persistence.session_codec import (
    SessionCodec,
    SessionDecodeError,
    SessionRecord,
    SessionRuntimeMetadataCodec,
    SessionRuntimeMetadataDecodeError,
    SkillSessionMetadataCodec,
    SkillSessionMetadataDecodeError,
)

# 匹配合法的session_id  8位日期-6位时间-4位十六进制随机字符
_SESSION_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")


class SessionError(MyCodeError):
    """会话文件无法创建、读取、切换或删除。"""


@dataclass(frozen=True)
class SessionInfo:
    """保存会话列表实际展示的文件信息。"""

    # 会话文件名中的唯一编号，用于恢复或删除指定会话
    session_id: str
    # 保存这段会话消息的 JSONL 文件路径
    path: Path
    # 从第一条用户消息中提取的会话列表标题
    title: str
    # 会话文件中最后一条有效消息的写入时间
    last_active: datetime


@dataclass(frozen=True)
class CurrentSessionSummary:
    """保存状态命令展示的当前会话概要。"""

    # 当前打开的会话编号，用户可用它恢复或定位会话文件
    session_id: str
    # 从当前会话第一条用户消息中提取的简短标题
    title: str
    # 当前会话最后一条有效消息的写入时间
    last_active: datetime
    # 当前会话在内存中保留的消息总数
    message_count: int


@dataclass(frozen=True)
class SessionCandidate:
    """保存从旧会话文件中读出的消息，等待检查通过后正式恢复"""

    # 要恢复的会话 ID
    session_id: str
    # 保存这段会话的 JSONL 文件路径
    path: Path
    # 成功解析并修整工具调用链后的消息
    messages: tuple[ChatMessage, ...]
    # 文件中最后一条成功解析记录的时间
    last_active: datetime
    # 因 JSON 或字段格式错误而跳过的行数
    skipped_lines: int
    # 是否删除过不完整或不匹配的工具调用链
    chain_truncated: bool


@dataclass(frozen=True)
class PreparedSession:
    """保存完成 Token 检查后可以正式切换的会话内容。"""

    # 最初从 JSONL 文件中读取并修整得到的候选会话
    candidate: SessionCandidate
    # 正式切换后放入当前 Conversation 的消息；内容过长时只保留近期消息
    messages: tuple[ChatMessage, ...]
    # 较早消息的摘要；恢复时没有压缩消息则为 None
    checkpoint: str | None
    # 为了满足模型上下文限制而执行的压缩次数
    compactions: int


@dataclass(frozen=True)
class SessionRestoreResult:
    """记录一次成功恢复旧会话时发生的修整和压缩，供界面显示"""

    # 成功恢复的会话 ID
    session_id: str
    # 因 JSON 或字段格式错误而跳过的记录行数
    skipped_lines: int
    # 是否删除过缺少对应结果的工具调用消息
    chain_truncated: bool
    # 为适应模型上下文限制而执行的压缩次数
    compactions: int
    # 是否因为会话超过 24 小时未活动而加入状态变化提醒
    time_gap_notice_added: bool
    # Skill 旁路文件损坏、定义缺失或模式变化产生的恢复提示。
    skill_warnings: tuple[str, ...] = ()
    # 会话原 Worktree 无效、状态损坏或项目指令加载异常产生的提示。
    worktree_warnings: tuple[str, ...] = ()


def _time_from_id(session_id: str) -> datetime:
    """从会话 ID 中读取会话创建时间

    函数取会话 ID 开头的日期和时间部分，并按照
    YYYYMMDD-HHMMSS 格式解析，最后补上当前系统的本地时区

    Args:
        session_id: 包含创建时间和随机后缀的会话 ID，例如``20260804-150913-a3f9``

    Returns:
        会话 ID 中记录的创建时间，并带有当前系统的本地时区
    """
    timestamp = session_id[:15]
    return datetime.strptime(timestamp, "%Y%m%d-%H%M%S").astimezone()


def _to_local_datetime(value: datetime) -> datetime:
    """把时间转换成带有当前系统本地时区的 datetime

    Args:
        value: 从会话记录中读取的时间，可以带时区，也可以不带时区

    Returns:
        使用当前系统本地时区，并且包含时区信息的时间
    """
    return value.astimezone()


def _title(message: UserMessage) -> str:
    """从用户消息的第一行生成会话列表标题

    Args:
        message: 用于生成标题的第一条用户消息

    Returns:
        整理并截断后的会话标题；第一行为空时返回“未命名会话”
    """
    first_line = message.content.splitlines()[0] if message.content.splitlines() else ""
    # 整理第一行中的空白：删除开头和结尾的空格，把连续多个空格或制表符压缩成一个普通空格
    normalized = " ".join(first_line.split())
    return normalized[:SESSION_TITLE_MAX_CHARS] or "未命名会话"


def _valid_chain(messages: Sequence[ChatMessage]) -> tuple[tuple[ChatMessage, ...], bool]:
    """检查并修整会话消息中的工具调用和工具结果。

        函数按照消息顺序检查每一批工具调用。助手发起的每个工具调用都必须有调用 ID 和工具名相匹配的工具结果。缺少结果、使用重复调用 ID，
        或被其他消息打断的整批工具调用会被删除；没有对应调用或工具名不匹配的工具结果会被跳过。传入的消息序列不会被修改。

    Args:
        messages: 从会话文件中成功解析出的消息，顺序与文件中的记录一致。

    Returns:
        一个二元组。第一项是删除残缺工具调用和无效工具结果后的消息元组；第二项表示修整过程中是否删除或跳过过任何消息。
     """
    # 保存当前这批还没有收到结果的工具调用  {"工具调用ID": "工具名称"}
    pending: dict[str, str] = {}
    # 保存当前这批工具调用在 valid 列表中的起始位置
    pending_start: int | None = None
    # 保存扫描过程中准备保留的消息
    valid: list[ChatMessage] = []
    # 标志检查过程中是否修改过消息序列
    repaired = False
    for message in messages:
        if isinstance(message, AssistantMessage) and message.tool_calls:
            if pending:
                # 表示这一批工具调用都来了，上一批还有工具调用没有结果
                # 由于pending非空所以断言pending_start非空
                assert pending_start is not None
                # 删除掉上一次模型工具调用的消息及其不完整的结果
                del valid[pending_start:]
                # 清空，准备处理下一次的
                pending.clear()
                # 修改过消息序列
                repaired = True
            # 保存当前这批工具调用在 valid 列表中的起始位置
            pending_start = len(valid)
            # 新的工具调用消息放进去
            valid.append(message)
            # 遍历助手工具调用工具请求
            for call in message.tool_calls:
                if call.id in pending:
                    # 假设模型使用重复的id调用了多个工具，则说明出问题，这个模型调用工具的请求及后续的工具调用结果都不要
                    del valid[pending_start:]
                    pending.clear()
                    pending_start = None
                    repaired = True
                    break
                # 将当前的工具调用放入pending
                pending[call.id] = call.name
            continue
        if isinstance(message, ToolResultMessage):
            expected_name = pending.get(message.tool_call_id)
            if expected_name is None or expected_name != message.tool_name:
                repaired = True
                continue
            valid.append(message)
            del pending[message.tool_call_id]
            if not pending:
                pending_start = None
            continue
        if pending:
            assert pending_start is not None
            del valid[pending_start:]
            pending.clear()
            pending_start = None
            repaired = True
        valid.append(message)
    if pending:
        assert pending_start is not None
        del valid[pending_start:]
        repaired = True
    return tuple(valid), repaired


class SessionManager:
    """把 Agent 已确认的消息先写入当前 JSONL，再更新内存历史。"""

    def __init__(
        self,
        workspace_root: Path,
        conversation: Conversation,
        context_manager: ContextManager,
        sessions_dir: Path | None = None,
    ) -> None:
        """绑定会话工作区、内存对话和可选生产会话目录。

        Args:
            workspace_root: 文件工具使用的主工作区绝对路径。
            conversation: 当前 Agent 在内存中的消息历史。
            context_manager: 负责恢复和重置压缩上下文的管理器。
            sessions_dir: 团队成员保存会话的绝对目录；未提供时使用工作区
                ``.mycode/sessions``。

        Returns:
            不返回数据。
        """
        # 当前工作目录
        self._workspace_root = workspace_root.resolve(strict=True)
        # 当前项目保存全部会话 JSONL 文件的目录
        if sessions_dir is not None and not sessions_dir.is_absolute():
            raise ValueError("会话目录必须是绝对路径")
        self._sessions_dir = (
            sessions_dir.resolve(strict=False)
            if sessions_dir is not None
            else self._workspace_root / ".mycode" / "sessions"
        )
        # 当前会话在进程内存中的消息记录
        self._conversation = conversation
        # 负责重置上下文摘要，以及采用恢复旧会话时生成的摘要
        self._context_manager = context_manager
        # 负责在 ChatMessage 对象和单行 JSON 字符串之间转换
        self._codec = SessionCodec()
        # 负责严格解析和生成同名 .meta.json 中的活动 Skill 名单。
        self._skill_codec = SkillSessionMetadataCodec()
        # 组合编解码器在不破坏旧 active_skills 字段的前提下保存 TeamBinding。
        self._runtime_codec = SessionRuntimeMetadataCodec()
        # 当前正在使用的会话 ID；尚未创建或恢复会话时为 None
        self._current_id: str | None = None
        # 当前已打开的会话 JSONL 文件；尚未打开或已经关闭时为 None
        self._file: TextIO | None = None

    @property
    def current_id(self) -> str:
        if self._current_id is None:
            raise SessionError("当前还没有创建会话")
        return self._current_id

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        return self._conversation.history

    def _path(self, session_id: str) -> Path:
        if _SESSION_ID.fullmatch(session_id) is None:
            raise SessionError("会话 ID 格式无效")
        return self._sessions_dir / f"{session_id}.jsonl"

    def _metadata_path(self, session_id: str) -> Path:
        """取得指定会话的 Skill 旁路文件路径。

        Args:
            session_id: 通过格式检查的会话 ID。

        Returns:
            与 JSONL 同目录、同名且以 ``.meta.json`` 结尾的路径。

        Raises:
            SessionError: 会话 ID 格式无效。
        """

        if _SESSION_ID.fullmatch(session_id) is None:
            raise SessionError("会话 ID 格式无效")
        return self._sessions_dir / f"{session_id}.meta.json"

    def save_skill_state(self, state: SkillSessionState) -> None:
        """原子保存当前会话的活动 inline Skill 名单。

        Args:
            state: 按激活顺序排列、只含 Skill 名的会话状态。

        Returns:
            None。数据完整写入同目录临时文件后才替换正式旁路文件。

        Raises:
            SessionError: 当前会话尚未创建，或旁路文件无法写入和替换。
        """

        if self._current_id is None:
            raise SessionError("当前还没有创建会话")
        target = self._metadata_path(self._current_id)
        current, _ = self.read_runtime_metadata(self._current_id)
        serialized = self._runtime_codec.encode(
            SessionRuntimeMetadata(skills=state, team=current.team)
        )
        temporary: Path | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, raw_path = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
            )
            temporary = Path(raw_path)
            with os.fdopen(
                descriptor,
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            temporary = None
        except (OSError, UnicodeError) as exc:
            raise SessionError(f"无法保存 Skill 会话状态：{exc}") from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def read_skill_state(
        self,
        session_id: str,
    ) -> tuple[SkillSessionState, str | None]:
        """读取一个会话保存的活动 Skill 名单。

        Args:
            session_id: 准备恢复或检查的会话 ID。

        Returns:
            二元组。第一项是状态；旧会话没有旁路文件时为空状态。第二项
            是坏文件的用户可见警告；正常读取或旧会话兼容时为 None。
        """

        metadata, warning = self.read_runtime_metadata(session_id)
        return metadata.skills, warning

    def save_runtime_metadata(self, state: SessionRuntimeMetadata) -> None:
        """原子保存当前会话的 Skill 状态和可选 TeamBinding。

        Args:
            state: 下一次恢复会话时需要重新装配的组合状态。

        Returns:
            数据完整写入并替换正式文件后不返回数据。

        Raises:
            SessionError: 当前没有会话或旁路文件无法写入。
        """

        if self._current_id is None:
            raise SessionError("当前还没有创建会话")
        target = self._metadata_path(self._current_id)
        serialized = self._runtime_codec.encode(state)
        temporary: Path | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, raw_path = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            temporary = Path(raw_path)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            temporary = None
        except (OSError, UnicodeError) as exc:
            raise SessionError(f"无法保存会话运行状态：{exc}") from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def read_runtime_metadata(
        self,
        session_id: str,
    ) -> tuple[SessionRuntimeMetadata, str | None]:
        """读取一个新旧会话保存的组合运行状态。

        Args:
            session_id: 准备恢复或检查的会话 ID。

        Returns:
            二元组。第一项是组合状态；没有旁路文件时为空状态。第二项是
            损坏文件的用户可见警告，正常读取时为 None。
        """

        path = self._metadata_path(session_id)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return SessionRuntimeMetadata(), None
        except (OSError, UnicodeError) as exc:
            return (
                SessionRuntimeMetadata(),
                f"无法读取会话运行状态，已只恢复消息历史：{exc}",
            )
        try:
            return self._runtime_codec.decode(text), None
        except SessionRuntimeMetadataDecodeError as exc:
            return (
                SessionRuntimeMetadata(),
                f"会话运行状态无效，已只恢复消息历史：{exc}",
            )

    def save_team_binding(self, binding: TeamBinding | None) -> None:
        """更新当前会话的团队绑定，同时保留活动 Skill 顺序。

        Args:
            binding: 当前团队和 Lead generation；团队删除成功时传 None。

        Returns:
            旁路文件原子替换后不返回数据。
        """

        metadata, _ = self.read_runtime_metadata(self.current_id)
        self.save_runtime_metadata(
            SessionRuntimeMetadata(skills=metadata.skills, team=binding)
        )

    def create_new(self) -> str:
        """创建空会话，成功后清除上一段对话的运行时上下文。"""

        try:
            # 创建目录
            self._sessions_dir.mkdir(parents=True, exist_ok=True)
            while True:
                # 生成session_id
                session_id = (
                    f"{datetime.now().astimezone():%Y%m%d-%H%M%S}-"
                    f"{secrets.token_hex(2)}"
                )
                # 得到当前会话的jsonl文件的存储路径
                path = self._path(session_id)
                try:
                    # 文件需要在整个会话期间保持打开，后续 append() 会继续使用，因此不使用 with；切换会话或程序退出时再显式关闭
                    # 只有目标文件目前不存在时才创建；如果已经存在，立即报错，绝不覆盖
                    new_file = path.open("x", encoding="utf-8", newline="\n")
                except FileExistsError:
                    continue
                break
        except OSError as exc:
            raise SessionError(f"无法创建会话文件：{exc}") from exc

        old_file = self._file
        self._file = new_file
        self._current_id = session_id
        self._conversation.clear()
        self._context_manager.reset()
        if old_file is not None:
            old_file.close()
        return session_id

    def append(self, messages: Sequence[ChatMessage]) -> None:
        """把一批消息保存到当前会话文件，并加入内存消息历史

        函数为每条消息记录当前时间，将其编码成单行 JSON，再按传入顺序追加到当前会话的 JSONL 文件。文件写入和刷新未报错后，才把同一批消息加入 Conversation。messages 为空时不执行任何操作。

        Args:
            messages: 需要保存的用户消息、助手消息或工具结果消息。
        """

        if not messages:
            return
        if self._file is None:
            raise SessionError("当前还没有打开会话文件")
        serialized = "".join(
            self._codec.encode(
                SessionRecord(datetime.now().astimezone(), message)
            )
            + "\n"
            for message in messages
        )
        # 严格保证先写文件再更新内存
        try:
            # 把编码后的 JSONL 文本写入 Python 文件缓冲区
            self._file.write(serialized)
            # 把 Python 文件缓冲区中的内容立即交给操作系统
            self._file.flush()
            # 避免每批消息都等待磁盘同步；会话追加后只刷新到操作系统缓存
            # os.fsync(self._file.fileno())
        except (OSError, UnicodeError) as exc:
            raise SessionError(f"无法追加会话消息：{exc}") from exc
        self._conversation.extend(messages)

    def _scan(self, path: Path) -> tuple[str, datetime]:
        """读取一个会话文件，取得会话列表需要的标题和最后活动时间

        函数逐行解析 JSONL。标题取第一条有效用户消息的第一行；最后活动时间取最后一条有效记录的时间。格式错误的记录会被跳过。如果文件中
        没有有效用户消息，标题使用“未命名会话”；如果没有任何有效记录，最后活动时间使用会话 ID 中的创建时间

        Args:
            path: 需要扫描的会话 JSONL 文件路径。

        Returns:
            一个二元组。第一项是会话标题，第二项是带本地时区的最后活动时间。
        """
        session_id = path.stem
        title = "未命名会话"
        # 默认使用当前会话的创建时间；读到有效消息后会更新为最后一条有效消息的时间
        last_active = _time_from_id(session_id)
        # 标志当前扫描的会话文件中，是否已经找到第一条有效的用户消息，并用它生成了标题
        found_user = False
        try:
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    try:
                        record = self._codec.decode(line)
                    except SessionDecodeError:
                        continue
                    last_active = _to_local_datetime(record.timestamp)
                    if not found_user and isinstance(record.message, UserMessage):
                        title = _title(record.message)
                        found_user = True
        except (OSError, UnicodeError) as exc:
            raise SessionError(f"无法读取会话文件 {path.name}：{exc}") from exc
        return title, last_active

    def list_sessions(self) -> tuple[SessionInfo, ...]:
        """扫描全部会话文件，返回最近使用的会话在前的会话列表"""

        if not self._sessions_dir.exists():
            return ()
        sessions: list[SessionInfo] = []
        for path in sorted(self._sessions_dir.glob("*.jsonl")):
            if _SESSION_ID.fullmatch(path.stem) is None:
                continue
            title, last_active = self._scan(path)
            sessions.append(
                SessionInfo(path.stem, path, title, last_active)
            )
        sessions.sort(key=lambda item: (item.last_active, item.session_id), reverse=True)
        return tuple(sessions)

    def current_summary(self) -> CurrentSessionSummary:
        """读取当前会话的标题、活动时间和内存消息数量。

        Returns:
            可直接供状态命令展示的当前会话概要。
        """

        session_id = self.current_id
        title, last_active = self._scan(self._path(session_id))
        return CurrentSessionSummary(
            session_id=session_id,
            title=title,
            last_active=last_active,
            message_count=len(self._conversation.history),
        )

    def read_candidate(self, session_id: str) -> SessionCandidate:
        """读取并修整指定旧会话，但不切换当前正在使用的会话。

        函数逐行解析目标 JSONL 文件。格式错误的记录会被跳过并计数，其余消息会按原顺序保存。读取完成后还会删除缺少对应结果的工具
        调用和无法配对的工具结果，最后把消息和读取情况组成候选会话。本函数不会修改当前 Conversation、上下文摘要或当前会话文件。

        Args:
            session_id: 需要读取的旧会话 ID。

        Returns:
            从目标文件中读取出的候选会话，包含修整后的消息、最后活动时间、跳过的损坏记录数量，以及是否删除过残缺工具调用。
        """

        path = self._path(session_id)
        if not path.is_file():
            raise SessionError("会话不存在")
        messages: list[ChatMessage] = []
        # 记录读取会话文件时，因为内容格式错误而跳过了多少行记录
        skipped = 0
        last_active = _time_from_id(session_id)
        try:
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    try:
                        record = self._codec.decode(line)
                    except SessionDecodeError:
                        skipped += 1
                        continue
                    messages.append(record.message)
                    last_active = _to_local_datetime(record.timestamp)
        except (OSError, UnicodeError) as exc:
            raise SessionError(f"无法读取会话文件：{exc}") from exc
        valid_messages, truncated = _valid_chain(messages)
        return SessionCandidate(
            session_id,
            path,
            valid_messages,
            last_active,
            skipped,
            truncated,
        )

    def activate(self, prepared: PreparedSession) -> None:
        """恢复一个已经检查过的旧会话，后续消息继续写入它原来的文件

         函数打开旧会话文件，替换内存消息并采用已生成的上下文摘要。切换失败时恢复原来的消息，并继续使用当前会话

        Args:
            prepared: 已完成消息修整和上下文长度检查的旧会话
        """

        candidate = prepared.candidate
        expected_path = self._path(candidate.session_id)
        if candidate.path != expected_path:
            raise SessionError("恢复候选的文件路径与会话 ID 不匹配")
        try:
            new_file = expected_path.open("a", encoding="utf-8", newline="\n")
        except OSError as exc:
            raise SessionError(f"无法打开待恢复会话：{exc}") from exc

        old_history = self._conversation.history
        try:
            self._conversation.replace(prepared.messages)
            self._context_manager.adopt_restored_context(
                prepared.checkpoint
            )
        except Exception:
            self._conversation.replace(old_history)
            new_file.close()
            raise

        old_file = self._file
        self._file = new_file
        self._current_id = candidate.session_id
        if old_file is not None:
            old_file.close()

    def delete(self, session_id: str) -> None:
        """永久删除一个未使用会话的消息文件和 Skill 旁路文件。

        Args:
            session_id: 需要删除的非当前会话 ID。

        Returns:
            None。旁路文件不存在时仍视为兼容的旧会话并正常完成。

        Raises:
            SessionError: 目标是当前会话、消息文件不存在或删除失败。
        """

        if self._current_id == session_id:
            raise SessionError("不能删除当前正在使用的会话")
        path = self._path(session_id)
        try:
            path.unlink()
        except FileNotFoundError as exc:
            raise SessionError("会话不存在") from exc
        except OSError as exc:
            raise SessionError(f"无法删除会话：{exc}") from exc
        try:
            self._metadata_path(session_id).unlink(missing_ok=True)
        except OSError as exc:
            raise SessionError(f"无法删除 Skill 会话状态：{exc}") from exc

    def cleanup_expired(self, now: datetime) -> int:
        """删除超过保留天数的非当前会话，并返回实际删除数量。"""

        cutoff = _to_local_datetime(now) - timedelta(days=SESSION_RETENTION_DAYS)
        deleted = 0
        for info in self.list_sessions():
            if info.session_id == self._current_id or info.last_active >= cutoff:
                continue
            self.delete(info.session_id)
            deleted += 1
        return deleted

    def close(self) -> None:
        """关闭当前 JSONL 句柄，不删除任何会话文件。"""

        if self._file is not None:
            self._file.close()
            self._file = None
