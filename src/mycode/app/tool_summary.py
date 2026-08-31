"""把工具调用与结果压缩成不会展示正文的终端安全摘要。"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from mycode.app.ui_models import (
    ToolDisplay,
    ToolDisplayStatus,
    single_line_display,
    truncate_display_text,
)
from mycode.errors import redact_secrets
from mycode.models.config import SecretValue
from mycode.models.tools import (
    ToolErrorCode,
    ToolExecutionResult,
    ToolInvocation,
)


_TOOL_LABELS = {
    "Agent": "Agent",
    "execute_command": "Shell",
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "find_files": "Find",
    "search_code": "Search",
}
_ASSIGNMENT_SECRET = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD))"
    r"\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s]+)"
)
_FLAG_SECRET = re.compile(
    r"(?i)(--?(?:api[-_]?key|token|secret|password|passwd))"
    r"(\s+|=)(\"[^\"]*\"|'[^']*'|[^\s]+)"
)
_BEARER_SECRET = re.compile(r"(?i)\bBearer\s+[^\s]+")
_ERROR_LABELS = {
    ToolErrorCode.INVALID_ARGUMENTS: "参数无效",
    ToolErrorCode.UNKNOWN_TOOL: "未知工具",
    ToolErrorCode.PATH_OUTSIDE_WORKSPACE: "路径越界",
    ToolErrorCode.NOT_FOUND: "目标不存在",
    ToolErrorCode.NOT_A_FILE: "目标不是文件",
    ToolErrorCode.ALREADY_EXISTS: "目标已存在",
    ToolErrorCode.INVALID_ENCODING: "编码无效",
    ToolErrorCode.INVALID_PATTERN: "模式无效",
    ToolErrorCode.NO_MATCH: "没有匹配",
    ToolErrorCode.MULTIPLE_MATCHES: "匹配不唯一",
    ToolErrorCode.COMMAND_FAILED: "命令失败",
    ToolErrorCode.IO_ERROR: "读写失败",
    ToolErrorCode.TIMEOUT: "执行超时",
    ToolErrorCode.BLOCKED: "权限拒绝",
    ToolErrorCode.CANCELLED: "已取消",
    ToolErrorCode.INTERNAL_ERROR: "内部错误",
}


def safe_summary_text(
    value: object,
    secrets: Iterable[SecretValue],
    *,
    limit: int = 160,
) -> str:
    text = single_line_display(str(value))
    text = redact_secrets(text, secrets)
    text = _ASSIGNMENT_SECRET.sub(r"\1=***", text)
    text = _FLAG_SECRET.sub(r"\1\2***", text)
    text = _BEARER_SECRET.sub("Bearer ***", text)
    return truncate_display_text(text, limit)


def _string_argument(arguments: Mapping[str, Any], name: str) -> str | None:
    value = arguments.get(name)
    return value if isinstance(value, str) and value else None


def _tool_target(
    invocation: ToolInvocation,
    secrets: tuple[SecretValue, ...],
) -> str | None:
    arguments = invocation.call.arguments
    name = invocation.call.name
    if name == "execute_command":
        raw = _string_argument(arguments, "command")
        return safe_summary_text(raw, secrets) if raw is not None else None
    if name in {"read_file", "write_file", "edit_file"}:
        raw = _string_argument(arguments, "path")
        return safe_summary_text(raw, secrets) if raw is not None else None
    if name == "find_files":
        raw = _string_argument(arguments, "pattern")
        return safe_summary_text(raw, secrets) if raw is not None else None
    if name == "search_code":
        pattern = _string_argument(arguments, "pattern")
        file_pattern = _string_argument(arguments, "file_pattern")
        if pattern is None and file_pattern is None:
            return None
        rendered = (
            f"{pattern or '?'} in {file_pattern or '?'}"
        )
        return safe_summary_text(rendered, secrets)
    return None


def _positive_int(metadata: Mapping[str, Any], name: str) -> int | None:
    value = metadata.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _result_detail(
    invocation: ToolInvocation,
    result: ToolExecutionResult,
) -> str | None:
    metadata = result.metadata
    name = invocation.call.name
    if not result.success:
        if result.error_code is None:
            return "执行失败"
        return _ERROR_LABELS.get(result.error_code, "执行失败")
    if name == "execute_command":
        exit_code = _positive_int(metadata, "exit_code")
        return f"exit {exit_code}" if exit_code is not None else None
    if name == "write_file":
        content = invocation.call.arguments.get("content")
        if isinstance(content, str):
            lines = content.count("\n") + 1
            return f"{lines} line" + ("" if lines == 1 else "s")
        size = _positive_int(metadata, "bytes_written")
        return f"{size} bytes" if size is not None else None
    if name == "edit_file":
        replacements = _positive_int(metadata, "replacements")
        if replacements is not None:
            return f"{replacements} replacement"
        return None
    if name in {"find_files", "search_code"}:
        matches = _positive_int(metadata, "match_count")
        if matches is not None:
            return f"{matches} match" + ("" if matches == 1 else "es")
    if name == "read_file":
        size = result.original_size_bytes
        return f"{size} bytes" if size > 0 else None
    return None


def summarize_tool_start(
    invocation: ToolInvocation,
    secrets: Iterable[SecretValue] = (),
) -> ToolDisplay:
    secret_values = tuple(secrets)
    return ToolDisplay(
        call_id=invocation.call.id,
        label=_TOOL_LABELS.get(invocation.call.name, "Tool"),
        target=_tool_target(invocation, secret_values),
        status=ToolDisplayStatus.RUNNING,
    )


def summarize_tool_result(
    invocation: ToolInvocation,
    result: ToolExecutionResult,
    secrets: Iterable[SecretValue] = (),
) -> ToolDisplay:
    if (
        invocation.call.id != result.tool_call_id
        or invocation.call.name != result.tool_name
    ):
        raise ValueError("工具结果与调用不匹配")
    secret_values = tuple(secrets)
    return ToolDisplay(
        call_id=invocation.call.id,
        label=_TOOL_LABELS.get(invocation.call.name, "Tool"),
        target=_tool_target(invocation, secret_values),
        status=(
            ToolDisplayStatus.SUCCESS
            if result.success
            else ToolDisplayStatus.FAILURE
        ),
        duration_ms=result.duration_ms,
        detail=_result_detail(invocation, result),
    )
