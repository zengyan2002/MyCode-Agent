"""
稳定的工作区文件查找与 UTF-8 正则内容搜索。
满足Glob模式：glob模式是一种专门用来匹配文件路径和文件名的通配符规则‌，它比正则表达式更简单直观，主要用在命令行和脚本里批量找文件
"""


from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from pathlib import Path

from mycode.models.json_types import JsonValue
from mycode.models.tools import ToolAccess, ToolDefinition, ToolErrorCode
from mycode.tools.base import ToolContext, ToolFailure, ToolOutput
from mycode.tools.builtin.paths import WorkspacePaths


_FIND_FILES = ToolDefinition(
    name="find_files",
    description="用工作区相对 Glob 查找工作区边界内的文件。",
    #工具输入参数的格式
    input_schema={
        "type": "object",
        #定义对象中允许出现的字段
        "properties": {"pattern": {"type": "string", "minLength": 1}},
        "required": ["pattern"],
        "additionalProperties": False,
    },
    access=ToolAccess.READ,
)
_SEARCH_CODE = ToolDefinition(
    name="search_code",
    description=(
        "使用正则表达式和工作区相对 Glob 搜索工作区边界内的 UTF-8 文件。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            #在文件中查找的内容
            "pattern": {"type": "string", "minLength": 1},
            #要查找的文件内容
            "file_pattern": {"type": "string", "minLength": 1},
        },
        "required": ["pattern", "file_pattern"],
        "additionalProperties": False,
    },
    access=ToolAccess.READ,
)

def _search_file(
    path: Path,
    expression: re.Pattern[str],
    relative: str,
) -> tuple[list[str], str | None]:
    """返回一个 UTF-8 文件中的全部匹配行。

    路径和正则由调用方提前验证。本函数只读取一个文件；无法作为文本读取
    时返回跳过原因，调用方据此统计，不会把部分结果误当成完整结果。
    """

    matches: list[str] = []
    try:
        # newline="" 可避免通用换行转换；格式化 path:line:text 时，
        # 只移除文件中真实存在的行结束符。
        with open(path, "r", encoding="utf-8", errors="strict", newline="") as handle:
            for number, line in enumerate(handle, start=1):
                #如果发现 NUL，立即放弃整个文件，因为一旦出现NUL就不是普通的utf-8文件了
                if "\x00" in line:
                    return [], "nul"
                if expression.search(line):
                    matches.append(
                        f"{relative}:{number}:{line.rstrip('\n\r')}"
                    )
    except UnicodeDecodeError:
        return [], "encoding"
    except OSError:
        return [], "unreadable"
    return matches, None

class FindFilesTool:
    """返回符合工作区 Glob 的全部文件路径。"""

    @property
    def definition(self) -> ToolDefinition:
        return _FIND_FILES

    async def execute(
        self,
        arguments: Mapping[str, JsonValue],
        context: ToolContext,
    ) -> ToolOutput:
        """查找文件并把安全绝对路径转换成工作区相对路径。"""

        paths = WorkspacePaths(context.workspace_root)
        try:
            files = await asyncio.to_thread(
                paths.matching_files,
                str(arguments["pattern"]),
            )
            relative = [paths.relative_path(path) for path in files]
            content = "\n".join(relative)
            return ToolOutput.ok(
                content,
                metadata={"match_count": len(relative)},
            )
        except ToolFailure as exc:
            return ToolOutput.fail(exc.code, str(exc))

class SearchCodeTool:
    """在工作区 UTF-8 文件中返回全部正则匹配行。"""

    @property
    def definition(self) -> ToolDefinition:
        return _SEARCH_CODE

    async def execute(
        self,
        arguments: Mapping[str, JsonValue],
        context: ToolContext,
    ) -> ToolOutput:
        """搜索候选文件，跳过二进制或无法读取的文件并返回统计。"""

        # 正则编译必须先于目录遍历，错误模式应快速失败，而不是扫描完工作区
        # 后才返回。文件 glob 与正则分别由不同参数表达，避免语义混用。
        paths = WorkspacePaths(context.workspace_root)
        try:
            try:
                expression = re.compile(str(arguments["pattern"]))
            except re.error as exc:
                raise ToolFailure(
                    ToolErrorCode.INVALID_PATTERN,
                    f"正则表达式无效：{exc}",
                ) from exc

            #查找满足file_pattern的候选文件
            candidates = await asyncio.to_thread(
                paths.matching_files,
                str(arguments["file_pattern"]),
            )
            output_lines: list[str] = []
            match_count = 0
            skipped = {"encoding": 0, "nul": 0, "unreadable": 0}

            for path in candidates:
                relative = paths.relative_path(path)
                lines, reason = await asyncio.to_thread(
                    _search_file,
                    path,
                    expression,
                    relative,
                )
                if reason is not None:
                    skipped[reason] += 1
                    continue
                output_lines.extend(lines)
                match_count += len(lines)

            content = "\n".join(output_lines)
            return ToolOutput.ok(
                content,
                metadata={
                    "match_count": match_count,
                    "files_considered": len(candidates),
                    "skipped_invalid_encoding": skipped["encoding"],
                    "skipped_nul": skipped["nul"],
                    "skipped_unreadable": skipped["unreadable"],
                },
            )
        except ToolFailure as exc:
            return ToolOutput.fail(exc.code, str(exc))
