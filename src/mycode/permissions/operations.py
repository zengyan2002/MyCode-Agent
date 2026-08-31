"""把内部工具调用转换为稳定的待授权操作。"""

from __future__ import annotations

import json

from mycode.models.messages import ToolCall
from mycode.models.permissions import PermissionOperation, PermissionTool
from mycode.models.tools import (
    ToolAccess,
    ToolErrorCode,
    ToolExecutionPolicy,
    ToolSource,
)
from mycode.tools.base import ToolFailure

"""
工具调用到权限操作的映射表
字典值：包含三个元素的元组 1. 权限工具类型2. 需要提取的参数名3. 该参数是不是路径
涉及到文件路径的ToolCall，例如：
    call = ToolCall(
    "call-1",
    "read_file",
    {
        "path": r"src\app.py",
    },
)
涉及到命令的ToolCall，例如：
    call = ToolCall(
    "call-2",
    "execute_command",
    {
        "command": "git status",
    },
)
"""
_OPERATION_FIELDS: dict[str, tuple[PermissionTool, str, bool]] = {
    "execute_command": (PermissionTool.SHELL, "command", False),
    "read_file": (PermissionTool.READ_FILE, "path", True),
    "write_file": (PermissionTool.WRITE_FILE, "path", True),
    "edit_file": (PermissionTool.WRITE_FILE, "path", True),
    "find_files": (PermissionTool.FIND_FILES, "pattern", True),
    "search_code": (PermissionTool.SEARCH_CODE, "file_pattern", True),
}


def permission_operation_for_call(call: ToolCall) -> PermissionOperation | None:
    """提取当前工具用于权限匹配的命令、路径或文件模式。"""

    definition = _OPERATION_FIELDS.get(call.name)
    if definition is None:
        return None
    tool, field, is_path = definition
    raw = call.arguments.get(field)
    if not isinstance(raw, str) or not raw:
        raise ToolFailure(
            ToolErrorCode.INVALID_ARGUMENTS,
            f"工具 {call.name} 的权限参数 {field} 必须是非空字符串",
        )
    value = raw.replace("\\", "/") if is_path else raw
    return PermissionOperation(
        tool=tool,
        match_value=value,
        display_value=value,
        path_value=value if is_path else None,
    )


def mcp_permission_operation_for_call(
    call: ToolCall,
) -> PermissionOperation:
    """把 MCP 调用转换为包含完整规范化参数的待授权操作。

    Args:
        call: 使用本地命名空间名称的 MCP 工具调用。

    Returns:
        不带路径值的 MCP 权限操作。
    """
    arguments_json = json.dumps(
        call.arguments,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    value = f"{call.name} {arguments_json}"
    return PermissionOperation(
        tool=PermissionTool.MCP,
        match_value=value,
        display_value=value,
        path_value=None,
    )


def skill_permission_operation_for_call(
    call: ToolCall,
    policy: ToolExecutionPolicy,
    access: ToolAccess,
) -> PermissionOperation:
    """把 Skill 专属工具调用转换成可配置的权限操作。

    Args:
        call: 模型发出的专属工具名称和参数。
        policy: ToolRegistry 为该工具保存的所属 Skill 和来源信息。
        access: 注册表中工具定义声明的 read 或 write 分类。

    Returns:
        包含 Skill 名、工具名、读写分类和规范化参数的权限操作。环境变量、
        SOP、stderr 和实际启动命令不会写入该记录。

    Raises:
        ToolFailure: policy 不是完整的 Skill 工具执行策略。
    """

    if policy.source is not ToolSource.SKILL or not policy.skill_name:
        raise ToolFailure(
            ToolErrorCode.INVALID_ARGUMENTS,
            f"工具 {call.name} 缺少 Skill 权限信息",
        )
    arguments_json = json.dumps(
        call.arguments,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    identity = f"{policy.skill_name}/{call.name}"
    match_value = f"{identity} {arguments_json}"
    return PermissionOperation(
        tool=PermissionTool.SKILL,
        match_value=match_value,
        display_value=f"{identity} ({access.value}) {arguments_json}",
        path_value=None,
    )
