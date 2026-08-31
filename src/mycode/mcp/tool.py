"""把远端 MCP 工具定义和调用结果适配为 MyCode 工具契约。"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from mcp.types import (
    AudioContent,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    Tool as SdkTool,
)

from mycode.errors import redact_secrets
from mycode.models.config import SecretValue
from mycode.models.json_types import JsonObject, JsonValue
from mycode.models.tools import ToolAccess, ToolDefinition, ToolErrorCode
from mycode.tools.base import ToolContext, ToolOutput

# 工具名必须以英文字母开头，只能包含字母、数字、下划线和短横线，总长度不超过 64
_TOOL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


class McpToolCaller(Protocol):
    """规定 McpToolAdapter 调用 MCP 工具时需要的方法。

    只要一个对象提供相同签名的 call_tool() 方法，就可以交给
    McpToolAdapter 使用。当前生产环境实际传入的是 McpConnection。
    """
    async def call_tool(
        self,
        remote_name: str,
        arguments: Mapping[str, JsonValue],
    ) -> CallToolResult:
        """调用指定远端工具。

        Args:
            remote_name: Server 原始工具名。
            arguments: 已校验的 JSON 参数。

        Returns:
            MCP 工具调用结果。
        """


def _compact_json(payload: JsonObject) -> str:
    """把结构化结果序列化为稳定的紧凑 JSON。

    Args:
        payload: MCP 返回的结构化 JSON 对象。

    Returns:
        保留 UTF-8 字符且键排序的 JSON 字符串。
    """
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _placeholder_for_block(
    block: ImageContent | AudioContent | ResourceLink | EmbeddedResource,
) -> str:
    """把图片、音频和资源转换成文字说明，不保留实际内容

    Args:
        block: 图片、音频、资源链接或嵌资源内容块。

    Returns:
        仅包含类型和小型标识信息的占位文本。
    """
    if isinstance(block, ImageContent):
        return f"[MCP 图片：{block.mimeType}，二进制正文已省略]"
    if isinstance(block, AudioContent):
        return f"[MCP 音频：{block.mimeType}，二进制正文已省略]"
    if isinstance(block, ResourceLink):
        details = [f"name={block.name}", f"uri={block.uri}"]
        if block.mimeType:
            details.append(f"type={block.mimeType}")
        return f"[MCP 资源链接：{', '.join(details)}]"

    # 不是前面三种的话就说明是嵌入式资源
    resource = block.resource
    mime_type = getattr(resource, "mimeType", None) or "unknown"
    uri = getattr(resource, "uri", "unknown")
    return (
        f"[MCP 嵌入资源：uri={uri}, type={mime_type}，"
        "嵌入正文已省略]"
    )


def mcp_result_to_tool_output(
    result: CallToolResult,
    *,
    secrets: Iterable[SecretValue] = (),
) -> ToolOutput:
    """把 MCP Server 返回的内容转换成 ToolOutput，并在交给模型前隐藏其中的已知密钥。

    Args:
        result: MCP Server 返回的工具执行结果。
        secrets: 配置中记录的 API Key、请求头和环境变量敏感值。

    Returns:
        MyCode 使用的工具结果；远端执行失败时包含 REMOTE_ERROR。
    """
    secret_values = tuple(secrets)
    parts: list[str] = []
    non_text_blocks = 0
    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
            continue
        if isinstance(
            block,
            (ImageContent, AudioContent, ResourceLink, EmbeddedResource),
        ):
            non_text_blocks += 1
            parts.append(_placeholder_for_block(block))

    if result.structuredContent is not None:
        parts.append(_compact_json(result.structuredContent))

    content = redact_secrets("\n".join(parts), secret_values)
    metadata: JsonObject = {
        "content_blocks": len(result.content),
        "non_text_blocks": non_text_blocks,
        "has_structured_content": result.structuredContent is not None,
    }
    if result.isError:
        return ToolOutput.fail(
            ToolErrorCode.REMOTE_ERROR,
            "MCP Server 报告工具调用失败",
            content=content,
            metadata=metadata,
        )
    return ToolOutput.ok(content, metadata=metadata)


@dataclass(frozen=True)
class McpToolAdapter:
    """将一个远端 MCP 工具包装成 MyCode 可注册、可执行的工具。是对接口Tool的实现"""

    # Server 在本地配置中的名称
    server_name: str
    # 不含本地命名空间的远端原始工具名
    remote_name: str
    # 提供给 Agent 和注册表的本地工具定义
    definition: ToolDefinition
    # 实际执行远端调用的窄连接接口
    caller: McpToolCaller
    # 需要从远端结果中统一移除的敏感值。
    secrets: tuple[SecretValue, ...] = ()

    @classmethod
    def from_remote_tool(
        cls,
        server_name: str,
        remote_tool: SdkTool,
        caller: McpToolCaller,
        *,
        secrets: Iterable[SecretValue] = (),
    ) -> "McpToolAdapter":
        """把 MCP Server 返回的一个工具转换成 MyCode 可以注册和调用的工具。

        生成的本地工具名格式为“Server名__远端工具名”。同时保留远端原名，供真正调用 MCP Server 时使用。

        Args:
            server_name: 当前 MCP Server 在配置中的名称。
            remote_tool: MCP Server 返回的工具名、说明和参数格式。
            caller: 真正向 MCP Server 发送工具调用的连接对象。
            secrets: 需要从工具返回内容中隐藏的密钥。

        Returns:
            可以注册到 MyCode 工具表中的 MCP 工具适配器。
        """
        # 设置远程工具在本地的工具名
        registered_name = f"{server_name}__{remote_tool.name}"
        if _TOOL_NAME.fullmatch(registered_name) is None:
            raise ValueError(
                f"MCP 工具组合名称无效或超过 64 个字符：{registered_name}"
            )
        description = (
            remote_tool.description.strip()
            if remote_tool.description and remote_tool.description.strip()
            else f"由 MCP Server {server_name} 提供的工具。"
        )
        # 拿到MCP Server 给工具附带的“补充说明”
        annotations = remote_tool.annotations
        # 如果说明中说其是只读工具，就定义其为只读，否则都是写工具
        access = (
            ToolAccess.READ
            if annotations is not None
            and annotations.readOnlyHint is True
            else ToolAccess.WRITE
        )
        definition = ToolDefinition(
            name=registered_name,
            description=description,
            input_schema=remote_tool.inputSchema,
            access=access,
        )
        return cls(
            server_name=server_name,
            remote_name=remote_tool.name,
            definition=definition,
            caller=caller,
            secrets=tuple(secrets),
        )

    async def execute(
        self,
        arguments: Mapping[str, JsonValue],
        context: ToolContext,
    ) -> ToolOutput:
        """调用 MCP Server 中的远端工具，并转换成 MyCode 的工具结果。

        Args:
            arguments: 模型传给工具的参数，进入这里前已经通过 Schema 检查。
            context: 所有工具统一接收的本地运行信息；MCP 工具不使用它。

        Returns:
            MCP Server 返回的工具内容；连接或通信失败时返回 IO_ERROR
        """
        del context
        try:
            result = await self.caller.call_tool(
                self.remote_name,
                arguments,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return ToolOutput.fail(
                ToolErrorCode.IO_ERROR,
                f"MCP Server {self.server_name} 通信失败",
            )
        return mcp_result_to_tool_output(result, secrets=self.secrets)
