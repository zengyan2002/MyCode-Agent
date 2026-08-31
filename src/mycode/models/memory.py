"""长期记忆相关模块共同使用的数据模型。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from mycode.models.messages import ChatMessage

# 匹配以字母或数字开头、只包含字母数字下划线和连字符的 .md 文件名
_NOTE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*\.md$")


class MemoryType(str, Enum):
    """自动记忆，四类记忆"""
    # 用户偏好   存到用户级目录（ ~/.mewcode/memory/ ）
    USER = "user"
    # 纠正反馈  存到用户级目录 ~/.mewcode/memory/
    FEEDBACK = "feedback"
    # 项目知识  存在项目级目录 <项目根>/.mewcode/memory/ 下
    PROJECT = "project"
    # 参考信息  存到项目级目录 <项目根>/.mewcode/memory/ 下
    REFERENCE = "reference"


class MemoryAction(str, Enum):
    """模型可以对四类记忆的现有笔记集合执行的三种修改。"""
    # 创建新笔记
    CREATE = "create"
    # 更新已有笔记
    UPDATE = "update"
    # 删除过时笔记
    DELETE = "delete"


def validate_note_filename(filename: str) -> None:
    """检查长期记忆笔记的文件名是否合法

    文件名只能包含英文字母、数字、下划线和连字符，必须以字母或数字开头并以 ``.md`` 结尾。不能包含目录，也不能使用为记忆索引保留的 ``memory.md``

    Args:
        filename: 需要检查的笔记文件名
    """
    if (
        not isinstance(filename, str)
        or _NOTE_FILENAME.fullmatch(filename) is None
        or filename.casefold() == "memory.md"
    ):
        raise ValueError("记忆文件名必须是普通的 .md 文件名")


@dataclass(frozen=True)
class MemoryNote:
    """表示 memory 目录中的一份独立 Markdown 笔记。"""

    # 笔记在 memory 目录中的实际文件名，例如 preferences.md
    filename: str
    # 笔记标题，会显示在 memory.md 索引中
    name: str
    # 笔记内容的简短说明，帮助模型判断是否需要读取正文
    description: str
    # 笔记类别，决定保存目录和索引分类
    type: MemoryType
    # 笔记的完整 Markdown 正文
    body: str

    def __post_init__(self) -> None:
        validate_note_filename(self.filename)
        if not isinstance(self.type, MemoryType):
            raise ValueError("记忆类型无效")
        for label, value in (
            ("名称", self.name),
            ("说明", self.description),
            ("正文", self.body),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"记忆{label}不能为空")


@dataclass(frozen=True)
class MemoryOperation:
    """描述一次创建、更新或删除笔记的请求。"""

    # 要执行的操作：创建、更新或删除
    action: MemoryAction
    # 目标笔记的类别，同时决定去用户记忆目录还是项目记忆目录操作
    type: MemoryType
    # 要创建、更新或删除的实际文件名，例如 preferences.md
    target: str
    # 创建或更新后要保存的完整笔记；删除操作不需要，必须为 None
    note: MemoryNote | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, MemoryAction):
            raise ValueError("记忆操作类型无效")
        if not isinstance(self.type, MemoryType):
            raise ValueError("记忆类型无效")
        validate_note_filename(self.target)
        if self.action in (MemoryAction.CREATE, MemoryAction.UPDATE):
            if self.note is None:
                raise ValueError("创建或更新记忆时必须提供完整笔记")
            if self.note.filename != self.target:
                raise ValueError("记忆目标和笔记文件名不一致")
            if self.note.type is not self.type:
                raise ValueError("记忆操作和笔记类型不一致")
        elif self.note is not None:
            raise ValueError("删除记忆时不能携带新笔记")


@dataclass(frozen=True)
class MemoryUpdate:
    """表示后台模型分析一轮对话后提出的全部记忆修改
    这些操作用于创建、更新或删除用户级和项目级记忆笔记。
    没有值得记录的内容时，operations 为空。
    """

    operations: tuple[MemoryOperation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.operations, tuple) or not all(
            isinstance(item, MemoryOperation) for item in self.operations
        ):
            raise ValueError("记忆更新必须包含操作元组")
        targets: set[tuple[str, str]] = set()
        for operation in self.operations:
            scope = (
                "user"
                if operation.type in (MemoryType.USER, MemoryType.FEEDBACK)
                else "project"
            )
            key = (scope, operation.target.casefold())
            if key in targets:
                raise ValueError("同一次更新不能重复操作同一记忆文件")
            targets.add(key)


@dataclass(frozen=True)
class MemorySnapshot:
    """保存提取模型需要看到的两份索引和全部现有笔记。"""
    # 用户级 ~/.mycode/memory/MEMORY.md 的完整文本，包含用户偏好和纠正反馈的目录
    user_index: str
    # 项目级 <项目>/.mycode/memory/MEMORY.md 的完整文本，包含项目知识和参考资料的目录
    project_index: str
    # 两级目录下所有普通记忆笔记的完整内容，不包含 MEMORY.md 索引
    notes: tuple[MemoryNote, ...]


@dataclass(frozen=True)
class CompletedTurn:
    """保存刚刚正常结束的一轮对话，供后台提取长期记忆

    一轮对话从用户发送消息开始，到 Agent 给出最终回复结束
    这里只保存本轮新产生的用户消息、工具调用、工具结果和最终回复不包含更早的会话历史
    """

    session_id: str
    messages: tuple[ChatMessage, ...]

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("完成回合必须包含会话 ID")
        if not self.messages:
            raise ValueError("完成回合必须包含消息")


class MemoryWorkerStatusKind(str, Enum):
    """后台记忆任务处理完一轮对话后，可能产生的三种结果状态"""

    # 成功创建、更新或删除了至少一份记忆笔记
    SUCCEEDED = "succeeded"
    # 分析正常完成，但本轮没有值得长期保存的内容
    NO_ACTION = "no_action"
    # 分析对话或写入记忆文件时发生错误
    FAILED = "failed"


@dataclass(frozen=True)
class MemoryWorkerStatus:
    """记录一个完成回合的后台笔记处理结果。"""

    # 处理结果：已更新记忆、无需更新或处理失败
    kind: MemoryWorkerStatusKind
    # 产生这轮对话的会话 ID
    session_id: str
    # 向用户说明处理结果或失败原因的文本
    message: str
