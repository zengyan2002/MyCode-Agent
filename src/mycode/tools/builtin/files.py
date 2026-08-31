"""提供读取、创建和编辑 UTF-8 文本文件的内置工具
创建和编辑只允许操作工作区内的文件；读取还支持用户记忆目录中的单个文件
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from mycode.constants import READ_FILE_CHUNK_LIMIT_BYTES
from mycode.models.json_types import JsonValue
from mycode.models.tools import (
    ToolAccess,
    ToolDefinition,
    ToolErrorCode,
)
from mycode.tools.base import ToolContext, ToolFailure, ToolOutput
from mycode.tools.builtin.paths import WorkspacePaths


_READ_FILE = ToolDefinition(
    name="read_file",
    description=(
        "读取工作区边界内、~/.mycode/memory/ 或当前活动 Skill 目录内的 UTF-8 文本文件。大文件可用 offset_bytes 和 "
        "limit_bytes 分段读取，并按返回的 next_offset 继续。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "offset_bytes": {"type": "integer", "minimum": 0},
            "limit_bytes": {
                "type": "integer",
                "minimum": 1,
                "maximum": READ_FILE_CHUNK_LIMIT_BYTES,
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    access=ToolAccess.READ,
)
_WRITE_FILE = ToolDefinition(
    name="write_file",
    description=(
        "使用工作区相对路径在工作区边界内创建 UTF-8 文本文件；目标已存在时绝不覆盖。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    access=ToolAccess.WRITE,
)
_EDIT_FILE = ToolDefinition(
    name="edit_file",
    description=(
        "编辑前必须先用 read_file 读取目标；仅在工作区边界内、原文恰好出现一次时替换。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "old_text": {"type": "string", "minLength": 1},
            "new_text": {"type": "string"},
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    },
    access=ToolAccess.WRITE,
)

def _read_utf8_range(
    path: Path,
    offset_bytes: int,
    limit_bytes: int | None,
) -> tuple[str, int, int, int | None]:
    """读取 UTF-8 文件中指定字节范围的文本

    函数先验证整个文件是否为有效的 UTF-8 文本。读取结束位置如果落在多字节字符中间，会向前调整到上一个完整字符边界
    Args:
        path: 需要读取的文件路径。
        offset_bytes: 开始读取的字节位置，必须位于 UTF-8 字符边界。
        limit_bytes: 最多读取的字节数；为 None 时读取到文件末尾
    Returns:
        本次读取的文本、文件总字节数、实际返回字节数和下一段起始位置。
        文件已经读完时，下一段起始位置为 None
    """
    try:
        data = path.read_bytes()
        # 完整解码一次，避免只读取某一段时漏掉文件其他位置的非法编码。
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolFailure(
            ToolErrorCode.INVALID_ENCODING,
            "文件不是有效的 UTF-8 文本",
        ) from exc
    except OSError as exc:
        raise ToolFailure(ToolErrorCode.IO_ERROR, "文件无法读取") from exc

    total_bytes = len(data)
    if offset_bytes > total_bytes:
        raise ToolFailure(
            ToolErrorCode.INVALID_ARGUMENTS,
            f"offset_bytes 超出文件大小（文件共 {total_bytes} 字节）",
        )
    try:
        data[:offset_bytes].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolFailure(
            ToolErrorCode.INVALID_ARGUMENTS,
            "offset_bytes 必须指向 UTF-8 字符边界",
        ) from exc

    end = (
        total_bytes
        if limit_bytes is None
        else min(total_bytes, offset_bytes + limit_bytes)
    )
    # UTF-8 字符最多占四字节，所以最多回退三次即可找到完整字符边界。
    while end > offset_bytes:
        try:
            content = data[offset_bytes:end].decode("utf-8")
            break
        except UnicodeDecodeError as exc:
            if exc.end != end - offset_bytes:
                raise ToolFailure(
                    ToolErrorCode.INVALID_ENCODING,
                    "文件不是有效的 UTF-8 文本",
                ) from exc
            end -= 1
    else:
        content = ""

    next_offset = end if end < total_bytes else None
    return content, total_bytes, end - offset_bytes, next_offset


def _slice_utf8_text(
    content: str,
    offset_bytes: int,
    limit_bytes: int | None,
) -> tuple[str, int, int, int | None]:
    """从已缓存的合法 UTF-8 正文中截取一个完整字符范围。

    Args:
        content: AgentFileCache 返回的完整 Unicode 正文。
        offset_bytes: 开始读取的 UTF-8 字节位置。
        limit_bytes: 最多返回的字节数；``None`` 表示读取到末尾。

    Returns:
        本次文本、文件总字节数、返回字节数和下一段偏移；读完时最后一项
        为 ``None``。

    Raises:
        ToolFailure: 偏移超过文件大小或落在多字节字符中间。
    """

    data = content.encode("utf-8")
    total_bytes = len(data)
    if offset_bytes > total_bytes:
        raise ToolFailure(
            ToolErrorCode.INVALID_ARGUMENTS,
            f"offset_bytes 超出文件大小（文件共 {total_bytes} 字节）",
        )
    try:
        data[:offset_bytes].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolFailure(
            ToolErrorCode.INVALID_ARGUMENTS,
            "offset_bytes 必须指向 UTF-8 字符边界",
        ) from exc
    end = (
        total_bytes
        if limit_bytes is None
        else min(total_bytes, offset_bytes + limit_bytes)
    )
    while end > offset_bytes:
        try:
            selected = data[offset_bytes:end].decode("utf-8")
            break
        except UnicodeDecodeError:
            end -= 1
    else:
        selected = ""
    return (
        selected,
        total_bytes,
        end - offset_bytes,
        end if end < total_bytes else None,
    )

def _write_exclusive(path: Path, content: str) -> None:
    """
    以 UTF-8 编码创建并写入一个新文件

    使用独占创建模式，目标已经存在时不会覆盖。文件创建后如果写入或同步到磁盘失败，会尽量删除已经产生的残缺文件

    Args:
        path: 需要创建的文件路径。
        content: 需要写入文件的完整文本
    """
    # open("x") 把“不覆盖已有文件”交给操作系统原子保证，避免先 exists()，再 open("w") 之间的竞态窗口。fsync 则确保报告成功前数据已经交给系统。
    #用来记录文件是否已经创建
    created = False
    try:
        #以独占模式打开文件
        with path.open("x", encoding="utf-8", newline="") as handle:
            created = True
            # 把文本写入缓冲区
            handle.write(content)
            # 把Python缓冲区里的数据交给操作系统
            handle.flush()
            # 取得这个文件在操作系统中的文件描述符，并且要求操作系统把该文件尚未落盘的数据同步到存储设备
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ToolFailure(
            ToolErrorCode.ALREADY_EXISTS,
            "目标文件已存在，请使用 edit_file 修改",
        ) from exc
    except OSError as exc:
        # 文件可能已经创建但写入或 fsync 失败；尽量清理掉这个残缺文件，避免后续调用把残缺文件误认为一次成功创建的结果。
        if created:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise ToolFailure(
            ToolErrorCode.IO_ERROR,
            "新文件无法写入",
        ) from exc

#在一个 UTF-8 文本文件中查找唯一出现的一段旧文本，将它替换成新文本，并通过同目录临时文件原子替换原文件。
def _edit_unique(path: Path, old_text: str, new_text: str) -> int:
    """替换 UTF-8 文件中唯一出现的一段文本

    旧文本必须在文件中恰好出现一次。替换后的完整内容会先写入同目录临时
    文件，写入成功后再替换原文件；检查或写入失败时保留原文件

    Args:
        path: 需要编辑的 UTF-8 文本文件路径。
        old_text: 需要查找并替换的原文。
        new_text: 用来替换原文的新文本。

    Returns:
        替换后完整文件的 UTF-8 字节数
    """
    try:
        #以二进制模式读取文件，返回字节
        data = path.read_bytes()
        #解析成utf-8字符串
        original = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolFailure(
            ToolErrorCode.INVALID_ENCODING,
            "文件不是有效的 UTF-8 文本",
        ) from exc
    except OSError as exc:
        raise ToolFailure(ToolErrorCode.IO_ERROR, "文件无法读取") from exc

    #统计旧文本出现次数
    occurrences = original.count(old_text)
    if occurrences == 0:
        raise ToolFailure(
            ToolErrorCode.NO_MATCH,
            "未找到待替换的原文",
        )
    if occurrences > 1:
        raise ToolFailure(
            ToolErrorCode.MULTIPLE_MATCHES,
            f"待替换原文出现了 {occurrences} 次",
        )

    #以新的文本内容替换旧文本内容，并转换成二进制bytes文本
    replacement = original.replace(old_text, new_text, 1).encode("utf-8")

    #临时文件路径
    temp_path: Path | None = None
    try:
        #在当前编辑的文件目录下安全创建临时文件，并且打开此文件
        descriptor, raw_temp = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temp_path = Path(raw_temp)

        #通过底层文件描述符 fd 创建一个文件对象,并返回这个文件对象
        with os.fdopen(descriptor, "wb") as handle:
            #将替换后的二进制bytes文本写入文件
            handle.write(replacement)
            #刷新 Python 缓冲区
            handle.flush()
            #handle.fileno() 取得底层文件描述符，os.fsync要求操作系统完成临时文件数据的同步。
            os.fsync(handle.fileno())
        #原子替换
        os.replace(temp_path, path)
        #临时文件路径置为None
        temp_path = None
    except OSError as exc:
        raise ToolFailure(
            ToolErrorCode.IO_ERROR,
            "编辑后的文件无法完成原子替换",
        ) from exc
    finally:
        # os.replace 成功后 temp_path 已置空；失败时删除同目录临时文件，
        # 清理失败不覆盖最初的编辑错误。
        if temp_path is not None:
            try:
                #删除临时文件
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
    return len(replacement)

class ReadFileTool:
    """读取工作区文本文件，供模型恢复存盘内容或检查普通源码。

    不传 ``limit_bytes`` 时返回从指定偏移到文件末尾的全部内容；传入上限
    时最多返回 48 KiB，并通过元数据告诉调用方下一段从哪里继续。
    """

    @property
    def definition(self) -> ToolDefinition:
        """返回模型可见的 read_file 参数说明。"""

        return _READ_FILE

    async def execute(
        self,
        arguments: Mapping[str, JsonValue],
        context: ToolContext,
    ) -> ToolOutput:
        """读取已通过 Schema 校验的文件路径和可选字节范围。"""

        paths = WorkspacePaths(context.workspace_root)
        try:
            path, display_path = paths.readable_file(
                str(arguments["path"]),
                context.user_memory_root,
                context.skill_resources,
            )
            offset_bytes = int(arguments.get("offset_bytes", 0))
            raw_limit = arguments.get("limit_bytes")
            limit_bytes = int(raw_limit) if raw_limit is not None else None
            try:
                cached_content = await asyncio.to_thread(
                    context.file_cache.read_text,
                    path,
                )
            except UnicodeDecodeError as exc:
                raise ToolFailure(
                    ToolErrorCode.INVALID_ENCODING,
                    "文件不是有效的 UTF-8 文本",
                ) from exc
            except OSError as exc:
                raise ToolFailure(
                    ToolErrorCode.IO_ERROR,
                    "文件无法读取",
                ) from exc
            content, total_bytes, returned_bytes, next_offset = _slice_utf8_text(
                cached_content,
                offset_bytes,
                limit_bytes,
            )
            metadata = {
                "path": display_path,
                "total_bytes": total_bytes,
                "offset_bytes": offset_bytes,
                "returned_bytes": returned_bytes,
                "next_offset": next_offset,
            }
            return ToolOutput.ok(
                content,
                metadata=metadata,
                original_size_bytes=returned_bytes,
            )
        except ToolFailure as exc:
            return ToolOutput.fail(exc.code, str(exc))

#写文件的工具类
class WriteFileTool:
    @property
    def definition(self) -> ToolDefinition:
        return _WRITE_FILE

    async def execute(
        self,
        arguments: Mapping[str, JsonValue],
        context: ToolContext,
    ) -> ToolOutput:
        # new_file 只验证并返回安全候选路径；真正的“目标不存在”保证仍由
        # _write_exclusive 在打开瞬间完成，抵御检查后的并发创建。
        paths = WorkspacePaths(context.workspace_root)
        try:
            path = paths.new_file(str(arguments["path"]))
            content = str(arguments["content"])
            await asyncio.to_thread(_write_exclusive, path, content)
            context.file_cache.invalidate(path)
            return ToolOutput.ok(
                f"已创建 {paths.relative_path(path)}",
                metadata={
                    "path": paths.relative_path(path),
                    "bytes_written": len(content.encode("utf-8")),
                },
            )
        except ToolFailure as exc:
            return ToolOutput.fail(exc.code, str(exc))

#编辑文件的工具类
class EditFileTool:
    @property
    def definition(self) -> ToolDefinition:
        return _EDIT_FILE

    async def execute(
        self,
        arguments: Mapping[str, JsonValue],
        context: ToolContext,
    ) -> ToolOutput:
        # 编辑返回最终 UTF-8 字节数作为元数据，而正文只给出简短确认，
        # 避免把整份修改后文件再次占用模型上下文。
        paths = WorkspacePaths(context.workspace_root)
        try:
            path = paths.existing_file(str(arguments["path"]))
            size = await asyncio.to_thread(
                _edit_unique,
                path,
                str(arguments["old_text"]),
                str(arguments["new_text"]),
            )
            context.file_cache.invalidate(path)
            return ToolOutput.ok(
                f"已编辑 {paths.relative_path(path)}",
                metadata={
                    "path": paths.relative_path(path),
                    "bytes_written": size,
                    "replacements": 1,
                },
            )
        except ToolFailure as exc:
            return ToolOutput.fail(exc.code, str(exc))
