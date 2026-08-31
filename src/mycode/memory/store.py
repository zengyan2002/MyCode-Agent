"""读取和更新用户级、项目级 Markdown 笔记。"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml

from mycode.constants import MEMORY_INDEX_MAX_BYTES, MEMORY_INDEX_MAX_LINES
from mycode.errors import MyCodeError, redact_secrets
from mycode.models.memory import (
    MemoryAction,
    MemoryNote,
    MemorySnapshot,
    MemoryType,
    MemoryUpdate,
)
from mycode.models.config import SecretValue
from mycode.models.prompts import RuntimeInstruction, RuntimeInstructionKind

# 保存到用户目录 ~/.mycode/memory/ 的记忆类型
_USER_TYPES = (MemoryType.USER, MemoryType.FEEDBACK)

# 保存到当前项目 .mycode/memory/ 的记忆类型
_PROJECT_TYPES = (MemoryType.PROJECT, MemoryType.REFERENCE)

# 生成 MEMORY.md 索引时，各类笔记的排列顺序；数字越小越靠前
_TYPE_ORDER = {
    MemoryType.USER: 0,
    MemoryType.FEEDBACK: 1,
    MemoryType.PROJECT: 2,
    MemoryType.REFERENCE: 3,
}

# 各类记忆在 MEMORY.md 索引中显示的中文名称
_TYPE_LABEL = {
    MemoryType.USER: "用户偏好",
    MemoryType.FEEDBACK: "纠正反馈",
    MemoryType.PROJECT: "项目知识",
    MemoryType.REFERENCE: "参考资料",
}


class MemoryStoreError(MyCodeError):
    """笔记或索引无法读取、校验或写入。"""


class MemoryStore:
    """维护当前用户和当前项目的两组独立笔记及索引。"""

    def __init__(
        self,
        project_root: Path,
        user_home: Path,
        *,
        secrets: Iterable[SecretValue] = (),
    ) -> None:
        try:
            # 项目根目录
            self._project_root = project_root.resolve(strict=True)
            # 用户家目录
            self._user_home = user_home.resolve(strict=True)
        except OSError as exc:
            raise MemoryStoreError("无法解析记忆根目录") from exc
        if not self._project_root.is_dir() or not self._user_home.is_dir():
            raise MemoryStoreError("记忆根目录必须是目录")
        # 项目级记忆存储目录  项目知识 参考信息
        self._project_memory = self._project_root / ".mycode" / "memory"
        # 用户级记忆存储目录  用户偏好 纠正反馈
        self._user_memory = self._user_home / ".mycode" / "memory"
        # 需要进行脱密处理的内容  例如api_key
        self._secrets = tuple(secrets)

    @property
    def project_memory_root(self) -> Path:
        return self._project_memory

    @property
    def user_memory_root(self) -> Path:
        return self._user_memory

    @staticmethod
    def _parse_note(path: Path) -> MemoryNote:
        """读取一份 Markdown 记忆文件，并转换成 MemoryNote

        函数使用 UTF-8 读取文件，将开头两条 ``---`` 之间的内容解析为YAML 元数据，并把结束标记之后的内容作为笔记正文。文件名直接取自
        path，名称、说明和记忆类型从 frontmatter 中读取

        Args:
            path: 要读取的普通 Markdown 记忆文件路径

        Returns:
            包含文件名、名称、说明、类型和正文的 MemoryNote
        """
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise MemoryStoreError(f"无法读取记忆文件 {path.name}：{exc}") from exc
        # 按行切割
        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            raise MemoryStoreError(f"记忆文件 {path.name} 缺少 YAML 前置元数据")
        try:
            # 找到YAML前置元数据的终止标志"---"
            end = next(
                index
                for index, line in enumerate(lines[1:], start=1)
                if line.strip() == "---"
            )
        except StopIteration as exc:
            raise MemoryStoreError(f"记忆文件 {path.name} 的 frontmatter 未闭合") from exc
        try:
            # 获得Yaml元数据
            metadata = yaml.safe_load("\n".join(lines[1:end]))
        except yaml.YAMLError as exc:
            raise MemoryStoreError(f"记忆文件 {path.name} 的 YAML 无效") from exc
        if not isinstance(metadata, Mapping):
            raise MemoryStoreError(f"记忆文件 {path.name} 的 frontmatter 必须是对象")
        try:
            memory_type = MemoryType(metadata["type"])
            return MemoryNote(
                filename=path.name,
                name=metadata["name"],
                description=metadata["description"],
                type=memory_type,
                body="\n".join(lines[end + 1 :]).strip(),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MemoryStoreError(f"记忆文件 {path.name} 的字段无效：{exc}") from exc

    @staticmethod
    def _inside(path: Path, root: Path) -> bool:
        """判断已存在的路径在解析符号链接后，是否位于指定根目录内（包含根目录本身）"""
        try:
            path.resolve(strict=True).relative_to(root.resolve(strict=True))
            return True
        except (OSError, ValueError):
            return False

    def _load_notes(
        self,
        root: Path,
        allowed: tuple[MemoryType, ...],
    ) -> tuple[MemoryNote, ...]:
        """读取指定目录中的全部普通记忆笔记

        函数按文件名排序读取目录下的 Markdown 文件，跳过 MEMORY.md 索引，
        并检查每个文件都位于目标目录内、内容格式正确且类型属于 allowed。
        目录不存在时返回空元组

        Args:
            root: 要扫描的用户级或项目级记忆目录。
            allowed: 当前目录允许保存的记忆类型。

        Returns:
            按文件名排序的记忆笔记元组。
        """
        if not root.exists():
            return ()
        try:
            paths = sorted(root.glob("*.md"), key=lambda item: item.name.casefold())
        except OSError as exc:
            raise MemoryStoreError(f"无法扫描记忆目录 {root}：{exc}") from exc
        notes: list[MemoryNote] = []
        for path in paths:
            if path.name.casefold() == "memory.md":
                # 不加载MEMORY.md 索引文件
                continue
            if not path.is_file() or not self._inside(path, root):
                raise MemoryStoreError(f"记忆文件超出目录范围：{path.name}")
            note = self._parse_note(path)
            if note.type not in allowed:
                raise MemoryStoreError(
                    f"记忆文件 {path.name} 的类型不属于当前目录"
                )
            notes.append(note)
        return tuple(notes)

    @staticmethod
    def _check_index(content: str, label: str) -> None:
        """检查一份记忆索引是否超过行数或文件大小限制

        Args:
            content: 准备写入 MEMORY.md 的完整索引文本。
            label: 用于错误提示的索引范围，例如“用户级”或“项目级”。
        """
        lines = len(content.splitlines())
        size = len(content.encode("utf-8"))
        if lines > MEMORY_INDEX_MAX_LINES or size > MEMORY_INDEX_MAX_BYTES:
            raise MemoryStoreError(
                f"{label}记忆索引超过 {MEMORY_INDEX_MAX_LINES} 行或 25KB"
            )

    def _read_index(self, root: Path, label: str) -> str:
        """读取并检查指定记忆目录中的 MEMORY.md 索引

        目录中没有 MEMORY.md 时返回空字符串；文件存在时，检查它位于目标
        目录内且没有超过索引大小限制，然后使用 UTF-8 读取完整内容

        Args:
            root: 用户级或项目级记忆目录。
            label: 用于错误提示的索引范围，例如“用户级”或“项目级”。

        Returns:
            MEMORY.md 的完整文本；索引不存在时返回空字符串
        """
        path = root / "MEMORY.md"
        if not path.exists():
            return ""
        if not path.is_file() or not self._inside(path, root):
            raise MemoryStoreError(f"{label}记忆索引路径无效")
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise MemoryStoreError(f"无法读取{label}记忆索引：{exc}") from exc
        self._check_index(content, label)
        return content

    def load_snapshot(self) -> MemorySnapshot:
        """读取当前用户和项目的全部长期记忆

        函数分别读取用户级、项目级 MEMORY.md，以及两个目录中的普通笔记，检查文件路径、笔记类型和索引大小后组成一份快照。目录或索引不存在时，
        对应内容为空。本函数只读取文件，不会修改现有记忆。

        Returns:
            包含用户级索引、项目级索引和四类现有笔记的 MemorySnapshot
        """

        user_notes = self._load_notes(self._user_memory, _USER_TYPES)
        project_notes = self._load_notes(self._project_memory, _PROJECT_TYPES)
        return MemorySnapshot(
            user_index=self._read_index(self._user_memory, "用户级"),
            project_index=self._read_index(self._project_memory, "项目级"),
            notes=(*user_notes, *project_notes),
        )

    def load_runtime_indexes(self) -> tuple[RuntimeInstruction, ...]:
        """把非空的长期记忆索引转换成模型请求使用的运行时指令

        函数读取用户级和项目级记忆快照，只把存在内容的 MEMORY.md 索引加入结果。用户级索引排在项目级索引之前。指令中只包含笔记目录，
        不包含笔记正文；模型需要详细内容时再调用文件读取工具

        Returns:
            由零到两条长期记忆指令组成的元组，顺序为用户级、项目级
        """

        snapshot = self.load_snapshot()
        instructions: list[RuntimeInstruction] = []
        if snapshot.user_index.strip():
            instructions.append(
                RuntimeInstruction(
                    RuntimeInstructionKind.LONG_TERM_MEMORY,
                    "用户级长期记忆索引（需要正文时读取 "
                    f"~/.mycode/memory/<文件名>）：\n{snapshot.user_index}",
                )
            )
        if snapshot.project_index.strip():
            instructions.append(
                RuntimeInstruction(
                    RuntimeInstructionKind.LONG_TERM_MEMORY,
                    "项目级长期记忆索引（需要正文时读取 "
                    f".mycode/memory/<文件名>）：\n{snapshot.project_index}",
                )
            )
        return tuple(instructions)

    @staticmethod
    def _render_note(note: MemoryNote) -> str:
        """把一份记忆笔记转换成可以写入 Markdown 文件的完整文本

        函数将名称、说明和类型写入 YAML frontmatter，再把笔记正文放在frontmatter 后面。这里只生成文本，不负责创建或写入文件。

        Args:
            note: 包含名称、说明、类型和正文的记忆笔记
        Returns:

            包含 YAML frontmatter 和正文，并以换行结尾的 Markdown 文本
        """
        metadata = yaml.safe_dump(
            {
                "name": note.name,
                "description": note.description,
                "type": note.type.value,
            },
            allow_unicode=True,
            sort_keys=False,
        ).strip()
        return f"---\n{metadata}\n---\n\n{note.body.strip()}\n"

    @staticmethod
    def _render_index(notes: Iterable[MemoryNote]) -> str:
        """根据现有笔记生成 MEMORY.md 的索引文本

        函数先按照记忆类型和文件名排序，再为每份笔记生成一行 Markdown链接，内容包括类型名称、笔记标题、文件名和简短说明。没有笔记时
        返回空字符串。本函数只生成文本，不负责写入索引文件

        Args:
            notes: 需要写入索引的一组记忆笔记

        Returns:
            排序后的 Markdown 索引文本；没有笔记时返回空字符串
        """

        ordered = sorted(
            notes,
            key=lambda note: (_TYPE_ORDER[note.type], note.filename.casefold()),
        )
        if not ordered:
            return ""
        return "".join(
            f"- [{_TYPE_LABEL[note.type]}] "
            f"[{note.name}]({note.filename}) — {note.description}\n"
            for note in ordered
        )

    def _memory_root_for(self, memory_type: MemoryType) -> Path:
        """返回指定记忆类型应该保存的用户级或项目级目录"""
        return (
            self._user_memory
            if memory_type in _USER_TYPES
            else self._project_memory
        )

    def _safe_note(self, note: MemoryNote) -> MemoryNote:
        """替换笔记标题、说明和正文中的已知密钥，并返回新的笔记对象"""
        return MemoryNote(
            filename=note.filename,
            name=redact_secrets(note.name, self._secrets),
            description=redact_secrets(note.description, self._secrets),
            type=note.type,
            body=redact_secrets(note.body, self._secrets),
        )

    def _prepare_update(
        self,
        update: MemoryUpdate,
    ) -> tuple[dict[tuple[Path, str], MemoryNote], set[tuple[Path, str]]]:
        """检查并在内存中计算一整批记忆修改的最终结果

        函数读取现有笔记，依次模拟创建、更新和删除操作。创建和更新的笔记会先替换已知密钥，但此阶段不会修改磁盘文件。全部操作检查通过后
        返回更新后的笔记集合和需要删除的文件

        Args:
            update: 模型提出的一整批创建、更新或删除笔记的操作

        Returns:
        一个二元组。第一项是执行全部操作后的笔记字典，键由存储目录和忽略大小写的文件名组成；第二项是稍后需要
        删除的文件集合，每项包含文件所在目录和忽略大小写后的文件名
        """

        snapshot = self.load_snapshot()
        # 把现有笔记按“所属目录和忽略大小写的文件名”建立索引，供后续创建、更新和删除操作快速查找
        notes = {
            (self._memory_root_for(note.type), note.filename.casefold()): note
            for note in snapshot.notes
        }
        # 记录本次更新最终需要删除哪些笔记文件
        deleted: set[tuple[Path, str]] = set()
        # 遍历本轮模型提出的记忆修改
        for operation in update.operations:
            # 拿到当前操作涉及的根目录
            root = self._memory_root_for(operation.type)
            key = (root, operation.target.casefold())
            existing = notes.get(key)
            if operation.action is MemoryAction.CREATE:
                if existing is not None or (root / operation.target).exists():
                    raise MemoryStoreError(f"记忆文件已存在：{operation.target}")
                assert operation.note is not None
                notes[key] = self._safe_note(operation.note)
            elif operation.action is MemoryAction.UPDATE:
                if existing is None:
                    raise MemoryStoreError(f"待更新记忆不存在：{operation.target}")
                if existing.type is not operation.type:
                    raise MemoryStoreError(f"待更新记忆类型不一致：{operation.target}")
                assert operation.note is not None
                notes[key] = self._safe_note(operation.note)
            else:
                if existing is None:
                    raise MemoryStoreError(f"待删除记忆不存在：{operation.target}")
                if existing.type is not operation.type:
                    raise MemoryStoreError(f"待删除记忆类型不一致：{operation.target}")
                del notes[key]
                deleted.add(key)
        return notes, deleted

    @staticmethod
    def _write_temporary(path: Path, content: str) -> Path:
        """把内容写入目标文件旁边的唯一临时文件

        函数根据正式文件路径生成临时文件名，以独占模式创建文件，并等待内容刷新到磁盘。写入失败时删除已经产生的临时文件；成功时只返回临时文件
        路径，不会替换正式文件

        Args:
            path: 最终准备替换的正式文件路径。
            content: 要写入临时文件的完整文本

        Returns:
            已经写入完成的临时文件路径
        """
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise MemoryStoreError(f"无法暂存记忆文件 {path.name}：{exc}") from exc
        return temporary

    def apply(self, update: MemoryUpdate) -> None:
        """把一批记忆修改写入笔记文件并重新生成两份索引

        函数先在内存中检查创建、更新和删除操作，生成更新后的用户级与项目级 索引，并确认索引没有超过大小限制。检查通过后，先把新笔记和索引写入
        临时文件，再用临时文件替换正式文件，最后删除不再保留的笔记。处理失败时会清理尚未提交的临时文件。没有任何操作时直接返回。

        Args:
            update: 本次需要创建、更新或删除的全部记忆操作
        """

        if not update.operations:
            return
        notes, deleted = self._prepare_update(update)
        # 取出用户级的记忆笔记
        user_notes = tuple(
            note for (root, _), note in notes.items() if root == self._user_memory
        )
        # 取出项目级的记忆笔记
        project_notes = tuple(
            note for (root, _), note in notes.items() if root == self._project_memory
        )
        # 用户级记忆索引文本
        user_index = self._render_index(user_notes)
        # 项目级记忆索引文本
        project_index = self._render_index(project_notes)
        # 检测两个记忆索引文本
        self._check_index(user_index, "用户级")
        self._check_index(project_index, "项目级")
        # 已经写好、等待替换正式文件 （临时文件路径，最终正式文件路径）
        staged: list[tuple[Path, Path]] = []
        try:
            # 创建两个级别的记忆笔记存储目录
            self._user_memory.mkdir(parents=True, exist_ok=True)
            self._project_memory.mkdir(parents=True, exist_ok=True)
            # 记录需要创建或重新写入的笔记键；完整笔记稍后通过 notes[key] 取得
            changed = {
                (self._memory_root_for(operation.type), operation.target.casefold())
                for operation in update.operations
                if operation.action is not MemoryAction.DELETE
            }
            # 遍历所有创建或更新的笔记，把它们写入临时文件，并记录每个临时文件对应的正式文件路径，等待后面统一替换
            for key in changed:
                note = notes[key]
                target = key[0] / note.filename
                staged.append((self._write_temporary(target, self._render_note(note)), target))
            # 把用户级和项目级的新索引分别写入临时文件，并加入待提交清单
            for root, content in (
                (self._user_memory, user_index),
                (self._project_memory, project_index),
            ):
                target = root / "MEMORY.md"
                staged.append((self._write_temporary(target, content), target))

            # 逐一用修改完的临时文件去替换正式文件
            for temporary, target in staged:
                temporary.replace(target)
            staged.clear()
            # 删除记忆文件
            for root, folded_name in deleted:
                matching = None

                for path in root.glob("*.md"):
                    if path.name.casefold() == folded_name:
                        matching = path
                        break
                if matching is not None:
                    matching.unlink()
        except MemoryStoreError:
            raise
        except OSError as exc:
            raise MemoryStoreError(f"无法应用记忆更新：{exc}") from exc
        finally:
            for temporary, _ in staged:
                temporary.unlink(missing_ok=True)
