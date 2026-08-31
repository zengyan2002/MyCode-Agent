"""提供使用可信本地 Actor 发送团队消息的 SendMessage 工具。"""

from __future__ import annotations

import json
from collections.abc import Mapping

from mycode.models.json_types import JsonObject, JsonValue
from mycode.models.teams import SendMessageRequest, TeamMessageKind
from mycode.models.tools import ToolAccess, ToolDefinition, ToolErrorCode
from mycode.teams.mailbox import TeamMailbox
from mycode.tools.base import ToolContext, ToolOutput


class SendMessageTool:
    """把普通通知或结构化协议消息持久化到目标团队邮箱。

    Attributes:
        mailbox: 负责名称解析、JSONL 追加、cursor 和按需唤醒的邮箱服务。
    """

    def __init__(self, mailbox: TeamMailbox) -> None:
        """保存当前工作区唯一的团队邮箱服务。

        Args:
            mailbox: 已连接 Supervisor wake handler 的 TeamMailbox。

        Returns:
            不返回数据。
        """
        self.mailbox = mailbox

    @property
    def definition(self) -> ToolDefinition:
        """返回 SendMessage 的收件人、类型、摘要和 payload 格式。

        Returns:
            注册表使用的消息写工具定义。
        """
        return ToolDefinition(
            "SendMessage", "给团队成员、Lead 或全体成员发送持久化消息；普通消息默认不唤醒。",
            {"type": "object", "properties": {
                "to": {"type": "string", "minLength": 1},
                "kind": {"type": "string", "enum": ["text", "plan_request", "plan_response", "shutdown_request", "shutdown_response"]},
                "summary": {"type": "string"}, "message": {"type": "string"},
                "wake": {"type": "boolean"}, "payload": {"type": "object"},
            }, "required": ["to"], "additionalProperties": False}, ToolAccess.WRITE,
        )

    async def execute(self, arguments: Mapping[str, JsonValue], context: ToolContext) -> ToolOutput:
        """使用 ToolContext 中的真实发送者投递一条消息。

        Args:
            arguments: to、消息类型、摘要、正文、wake 和结构化 payload。
            context: 保存真实 TeamActorContext 的本地工具上下文。

        Returns:
            每个收件人的 delivered 和 error；没有团队身份时返回 BLOCKED。
        """
        if context.team_actor is None:
            return ToolOutput.fail(ToolErrorCode.BLOCKED, "当前会话没有 Agent Team 身份")
        try:
            payload = arguments.get("payload", {})
            report = await self.mailbox.send(context.team_actor, SendMessageRequest(
                to=str(arguments["to"]), kind=TeamMessageKind(str(arguments.get("kind", "text"))),
                summary=str(arguments.get("summary", "")), message=str(arguments.get("message", "")),
                wake=bool(arguments.get("wake", False)), payload=dict(payload) if isinstance(payload, Mapping) else {},  # type: ignore[arg-type]
            ))
            content = [{"recipient_id": item.recipient_id, "delivered": item.delivered, "error": item.error} for item in report.deliveries]
            return ToolOutput.ok(json.dumps(content, ensure_ascii=False, indent=2))
        except Exception as exc:
            return ToolOutput.fail(ToolErrorCode.INVALID_ARGUMENTS, str(exc))
