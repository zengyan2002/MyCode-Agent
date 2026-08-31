"""组装六个内置工具。"""

from __future__ import annotations

from mycode.tools.builtin.command import ExecuteCommandTool
from mycode.tools.builtin.files import EditFileTool, ReadFileTool, WriteFileTool
from mycode.tools.builtin.search import FindFilesTool, SearchCodeTool
from mycode.tools.builtin.tool_search import ToolSearchTool
from mycode.models.tools import ToolSource
from mycode.tools.registry import ToolRegistry


def create_builtin_registry() -> ToolRegistry:
    """
    返回参数的类型：ToolRegistry
    registry._tools  工具
    registry.__validators 参数校验器
    """
    registry = ToolRegistry()
    # 注册顺序会直接展示给模型，因此必须保持稳定；稳定顺序也有利于Provider 对相同请求前缀进行缓存。新增工具还必须明确 access 分类，
    # 因为调度屏障和 Plan 模式都依赖该元数据，而不是工具名称。
    for tool in (
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        ExecuteCommandTool(),
        FindFilesTool(),
        SearchCodeTool(),
    ):
        registry.register(tool)
    registry.register(
        ToolSearchTool(registry),
        source=ToolSource.SYSTEM,
    )
    return registry
