"""在本地工具注册表中查找并激活尚未显示的 MCP 工具。"""

from __future__ import annotations

from collections.abc import Mapping

from mycode.models.json_types import JsonValue
from mycode.models.tools import ToolAccess, ToolDefinition
from mycode.tools.base import ToolContext, ToolOutput
from mycode.tools.registry import ToolRegistry


_TOOL_SEARCH = ToolDefinition(
    name="tool_search",
    description="按名称或描述查找并激活当前未显示的 MCP 工具。",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "pattern": r"\S",
                "description": "要查找的工具名称或用途关键词。",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    access=ToolAccess.READ,
)


class ToolSearchTool:
    """搜索注册表中的 MCP 工具，并只激活到调用它的 Agent。"""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    @property
    def definition(self) -> ToolDefinition:
        """返回模型调用工具搜索时使用的名称和参数格式。"""
        return _TOOL_SEARCH

    async def execute(
        self,
        arguments: Mapping[str, JsonValue],
        context: ToolContext,
    ) -> ToolOutput:
        """搜索并激活 MCP 工具，同时返回名称和说明。

        Args:
            arguments: 已通过 Schema 校验、包含 ``query`` 的模型参数。
            context: 当前 Agent 独享的工具上下文；激活结果写入其中的
                ``tool_activation``，不会影响父 Agent 或其他后台任务。

        Returns:
            找到结果时返回已激活工具的名称和说明；没有匹配时返回普通成功
            消息，不暴露工具 JSON Schema。
        """

        matches = self._registry.search_mcp(
            str(arguments["query"]),
            context.tool_activation.active_mcp_names,
        )
        if not matches:
            return ToolOutput.ok("没有找到匹配的未激活 MCP 工具。")

        context.tool_activation.activate(
            frozenset(definition.name for definition in matches)
        )

        lines = ["已激活以下 MCP 工具；下一次模型请求会提供完整定义："]
        lines.extend(
            f"- {definition.name}: {definition.description.strip()}"
            for definition in matches
        )
        return ToolOutput.ok("\n".join(lines))
