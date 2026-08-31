"""团队成员名称寻址、JSONL 邮箱和协议消息校验。"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from mycode.models.teams import (
    DeliveryItem,
    DeliveryReport,
    MailboxCursor,
    MailboxMessage,
    MemberPlanApproval,
    PlanDecision,
    SendMessageRequest,
    TeamActorContext,
    TeamMessageKind,
)
from mycode.models.messages import UserMessage
from mycode.teams.locks import ExclusiveFileLock
from mycode.teams.store import TeamStateStore, _atomic_json, _read_json


class TeamMailboxError(RuntimeError):
    """表示收件人无效、消息协议越权或邮箱文件无法安全更新。"""


WakeHandler = Callable[[str, str, str], Awaitable[None]]


class TeamMailbox:
    """把团队消息先写入每个收件人的独立邮箱，再按需唤醒成员。

    Attributes:
        store: 提供团队成员名称和邮箱目录的持久化入口。
        wake_handler: 消息落盘后调用的 Supervisor 唤醒函数。
    """

    def __init__(
        self,
        store: TeamStateStore,
        wake_handler: WakeHandler | None = None,
    ) -> None:
        """保存 Store 和可在装配后补充的唤醒入口。

        Args:
            store: 当前工作区唯一的 TeamStateStore。
            wake_handler: 接收 team ID、member ID 和原因的异步唤醒函数。

        Returns:
            不返回数据。
        """

        self.store = store
        self.wake_handler = wake_handler

    def set_wake_handler(self, handler: WakeHandler) -> None:
        """在 Supervisor 创建完成后设置真实唤醒入口。

        Args:
            handler: 接收目标成员记录并唤醒其已选后端的异步函数。

        Returns:
            唤醒入口替换完成后不返回数据。
        """

        self.wake_handler = handler

    async def send(
        self,
        actor: TeamActorContext,
        request: SendMessageRequest,
    ) -> DeliveryReport:
        """校验并投递一条点对点或广播消息。

        Args:
            actor: 本地运行时确认的真实发送者。
            request: 收件人、消息类型、正文、摘要、唤醒和协议 payload。

        Returns:
            每个收件人各自的持久化或唤醒错误。
        """

        self.store.require_actor(actor)
        self._validate_protocol(actor, request)
        recipients = self._resolve_recipients(actor, request.to)
        deliveries: list[DeliveryItem] = []
        for recipient in recipients:
            try:
                message = self._append(actor, recipient, request)
                if request.wake and recipient != "lead" and self.wake_handler is not None:
                    try:
                        await self.wake_handler(actor.team_id, recipient, f"message:{message.message_id}")
                    except Exception as exc:
                        deliveries.append(DeliveryItem(recipient, True, f"消息已保存，但唤醒失败：{exc}"))
                        continue
                deliveries.append(DeliveryItem(recipient, True))
            except Exception as exc:
                deliveries.append(DeliveryItem(recipient, False, str(exc)))
        return DeliveryReport(tuple(deliveries))

    def read_unread(self, actor: TeamActorContext) -> tuple[MailboxMessage, ...]:
        """从当前 cursor 读取调用者收件箱中的完整未读消息。

        Args:
            actor: 当前 Lead 或成员身份，决定实际收件箱路径。

        Returns:
            按文件顺序排列且 message ID 去重的未读消息。
        """

        self.store.require_actor(actor)
        recipient = "lead" if actor.actor_kind == "lead" else actor.actor_id
        directory = self.store.team_dir(actor.team_id)
        cursor = self._read_cursor(directory, recipient)
        path = directory / "mailboxes" / f"{recipient}.jsonl"
        try:
            size = path.stat().st_size
            if cursor.byte_offset > size:
                raise TeamMailboxError("邮箱 cursor 超出当前文件大小")
            messages: list[MailboxMessage] = []
            seen: set[str] = set()
            with path.open("rb") as handle:
                handle.seek(cursor.byte_offset)
                for raw_line in handle:
                    if not raw_line.endswith(b"\n"):
                        break
                    message = self._decode_message(raw_line.decode("utf-8"))
                    if message.message_id not in seen and message.message_id != cursor.last_message_id:
                        seen.add(message.message_id)
                        messages.append(message)
            return tuple(messages)
        except (OSError, UnicodeError) as exc:
            raise TeamMailboxError(f"无法读取收件箱：{exc}") from exc

    def acknowledge(
        self,
        actor: TeamActorContext,
        messages: Sequence[MailboxMessage],
    ) -> MailboxCursor:
        """在消息处理完成后把 cursor 推进到当前邮箱末尾。

        Args:
            actor: 当前收件人身份。
            messages: 已经成功注入并处理的消息；为空时不推进。

        Returns:
            已原子保存的新 cursor。
        """

        if not messages:
            recipient = "lead" if actor.actor_kind == "lead" else actor.actor_id
            return self._read_cursor(self.store.team_dir(actor.team_id), recipient)
        self.store.require_actor(actor)
        recipient = "lead" if actor.actor_kind == "lead" else actor.actor_id
        directory = self.store.team_dir(actor.team_id)
        path = directory / "mailboxes" / f"{recipient}.jsonl"
        try:
            offset = path.stat().st_size
        except OSError as exc:
            raise TeamMailboxError(f"无法读取邮箱大小：{exc}") from exc
        cursor = MailboxCursor(offset, messages[-1].message_id)
        _atomic_json(
            directory / "cursors" / f"{recipient}.json",
            {"byte_offset": cursor.byte_offset, "last_message_id": cursor.last_message_id},
        )
        return cursor

    def drain_for_agent(
        self,
        actor: TeamActorContext,
    ) -> tuple[UserMessage, ...]:
        """把当前收件人的未读消息转换成可注入模型的用户消息。

        Args:
            actor: 当前 Lead 或成员的可信本地身份。

        Returns:
            每封未读邮件对应一条带 ``team-message`` 标签的
            ``UserMessage``；没有新消息时返回空元组。返回前 cursor 已推进，
            因为消息已经成功转换成 AgentTurnRunner 可接收的对象。
        """

        messages = self.read_unread(actor)
        converted = tuple(
            UserMessage(
                "\n".join(
                    (
                        "<team-message>",
                        f"message_id: {message.message_id}",
                        f"from: {message.sender_id}",
                        f"kind: {message.kind.value}",
                        f"summary: {message.summary}",
                        "body:",
                        message.body,
                        "</team-message>",
                    )
                )
            )
            for message in messages
        )
        if messages:
            self.acknowledge(actor, messages)
        return converted

    def latest_plan_approval(
        self,
        team_id: str,
        member_id: str,
    ) -> MemberPlanApproval | None:
        """读取成员邮箱中最新的计划审批回复。

        Args:
            team_id: 成员所属团队 ID。
            member_id: 需要判断写权限的成员 ID。

        Returns:
            按文件顺序最后一条有效 ``plan_response``；尚未回复
            时返回 ``None``。本方法不推进 cursor，因为审批还要由
            TeammateHost 正常注入成员对话。

        Raises:
            TeamMailboxError: 邮箱包含损坏的完整 JSONL 记录。
        """

        path = self.store.team_dir(team_id) / "mailboxes" / f"{member_id}.jsonl"
        latest: MemberPlanApproval | None = None
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    message = self._decode_message(line)
                    if message.kind is not TeamMessageKind.PLAN_RESPONSE:
                        continue
                    payload = message.payload
                    latest = MemberPlanApproval(
                        member_id=member_id,
                        task_id=str(payload["task_id"]),
                        attempt_number=int(payload["attempt_number"]),
                        plan_revision=int(payload["plan_revision"]),
                        plan_text=message.body,
                        decision=PlanDecision(str(payload["decision"])),
                        feedback=(
                            str(payload["feedback"])
                            if payload.get("feedback") is not None
                            else None
                        ),
                        decided_by_generation=int(payload["lead_generation"]),
                        updated_at=message.created_at,
                    )
        except (OSError, UnicodeError, KeyError, TypeError, ValueError) as exc:
            raise TeamMailboxError(f"无法读取计划审批：{exc}") from exc
        return latest

    def _resolve_recipients(self, actor: TeamActorContext, address: str) -> tuple[str, ...]:
        """把成员名称、Agent ID、Lead 或广播地址解析成实际收件人。

        Args:
            actor: 已由 Store 验证的发送方身份。
            address: SendMessage 的 ``to`` 字段。

        Returns:
            不包含广播发送方自己的收件人 ID 元组。

        Raises:
            TeamMailboxError: 点对点地址无法在当前团队花名册中解析。
        """

        snapshot = self.store.load_team(actor.team_id)
        by_id = {item.agent_id: item.agent_id for item in snapshot.members}
        by_name = {item.name: item.agent_id for item in snapshot.members}
        sender = "lead" if actor.actor_kind == "lead" else actor.actor_id
        if address == "*":
            recipients = ["lead", *by_id]
            return tuple(item for item in recipients if item != sender)
        if address == "lead":
            return ("lead",)
        resolved = by_id.get(address) or by_name.get(address)
        if resolved is None:
            raise TeamMailboxError(f"团队收件人不存在：{address}")
        return (resolved,)

    def _append(
        self,
        actor: TeamActorContext,
        recipient: str,
        request: SendMessageRequest,
    ) -> MailboxMessage:
        """在收件人专属锁内追加一行消息并同步到磁盘。

        Args:
            actor: 已验证的消息发送方身份。
            recipient: 名称解析后的单个收件人 ID。
            request: 已通过协议校验的消息输入。

        Returns:
            已写入邮箱的完整消息记录。

        Raises:
            TeamMailboxError: 文件无法追加或同步。
        """

        directory = self.store.team_dir(actor.team_id)
        path = directory / "mailboxes" / f"{recipient}.jsonl"
        payload = dict(request.payload)
        if request.kind is TeamMessageKind.PLAN_RESPONSE:
            payload["lead_generation"] = actor.generation
        message = MailboxMessage(
            message_id=f"msg-{secrets.token_hex(8)}",
            team_id=actor.team_id,
            sender_id="lead" if actor.actor_kind == "lead" else actor.actor_id,
            recipient_id=recipient,
            kind=request.kind,
            summary=request.summary.strip() or self._protocol_summary(request.kind),
            body=request.message,
            wake=request.wake,
            payload=payload,
            created_at=datetime.now().astimezone(),
        )
        line = json.dumps(
            {
                "message_id": message.message_id,
                "team_id": message.team_id,
                "sender_id": message.sender_id,
                "recipient_id": message.recipient_id,
                "kind": message.kind.value,
                "summary": message.summary,
                "body": message.body,
                "wake": message.wake,
                "payload": message.payload,
                "created_at": message.created_at.isoformat(),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        lock = directory / "locks" / f"mailbox-{recipient}.lock"
        with ExclusiveFileLock(lock, message.sender_id):
            try:
                with path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                raise TeamMailboxError(f"无法写入收件箱：{exc}") from exc
        return message

    @staticmethod
    def _validate_protocol(actor: TeamActorContext, request: SendMessageRequest) -> None:
        """检查消息类型所需字段和发送方权限。

        Args:
            actor: 运行时确认的 Lead 或成员身份。
            request: 准备发送的文本或结构化协议消息。

        Returns:
            消息可以安全路由时不返回数据。

        Raises:
            TeamMailboxError: 摘要、协议字段、决定值或发送方身份不合法。
        """

        if request.kind is TeamMessageKind.TEXT and not request.summary.strip():
            raise TeamMailboxError("普通文本消息必须包含可浏览摘要")
        if request.kind is TeamMessageKind.PLAN_RESPONSE and actor.actor_kind != "lead":
            raise TeamMailboxError("只有 Lead 能发送计划审批回复")
        if request.kind is TeamMessageKind.SHUTDOWN_RESPONSE and actor.actor_kind != "member":
            raise TeamMailboxError("只有成员能发送退出回复")
        if request.kind in {TeamMessageKind.PLAN_REQUEST, TeamMessageKind.PLAN_RESPONSE}:
            required = {"task_id", "attempt_number", "plan_revision"}
            if not required.issubset(request.payload):
                raise TeamMailboxError("计划消息必须包含任务、执行次数和计划版本")
        if request.kind is TeamMessageKind.PLAN_RESPONSE:
            if request.payload.get("decision") not in {"approved", "rejected"}:
                raise TeamMailboxError("计划回复 decision 只能是 approved 或 rejected")
        if request.kind is TeamMessageKind.SHUTDOWN_RESPONSE:
            if request.to != "lead":
                raise TeamMailboxError("退出回复只能发给 Lead")

    @staticmethod
    def _protocol_summary(kind: TeamMessageKind) -> str:
        """返回结构化协议消息在列表中显示的固定摘要。

        Args:
            kind: 已确认的团队消息类型。

        Returns:
            协议消息的中文摘要；普通文本返回空串并要求调用方提供摘要。
        """

        return {
            TeamMessageKind.PLAN_REQUEST: "成员提交计划",
            TeamMessageKind.PLAN_RESPONSE: "Lead 回复计划",
            TeamMessageKind.SHUTDOWN_REQUEST: "Lead 请求成员退出",
            TeamMessageKind.SHUTDOWN_RESPONSE: "成员回复退出请求",
            TeamMessageKind.TEXT: "",
        }[kind]

    @staticmethod
    def _decode_message(line: str) -> MailboxMessage:
        """把邮箱 JSONL 中的一行还原为消息记录。

        Args:
            line: 单条 UTF-8 JSON 文本，不含换行符也可解析。

        Returns:
            完成枚举和时间转换的 MailboxMessage。

        Raises:
            TeamMailboxError: 行内容缺字段、类型错误或不是合法 JSON。
        """

        try:
            raw: dict[str, Any] = json.loads(line)
            return MailboxMessage(
                message_id=str(raw["message_id"]),
                team_id=str(raw["team_id"]),
                sender_id=str(raw["sender_id"]),
                recipient_id=str(raw["recipient_id"]),
                kind=TeamMessageKind(str(raw["kind"])),
                summary=str(raw["summary"]),
                body=str(raw["body"]),
                wake=bool(raw["wake"]),
                payload=dict(raw.get("payload", {})),
                created_at=datetime.fromisoformat(str(raw["created_at"])),
            )
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise TeamMailboxError(f"收件箱包含无效消息：{exc}") from exc

    @staticmethod
    def _read_cursor(directory: Path, recipient: str) -> MailboxCursor:
        """读取一个收件人已经确认消费到的邮箱位置。

        Args:
            directory: 当前团队运行目录。
            recipient: Lead 或成员 Agent ID。

        Returns:
            持久化的字节偏移和最近消息 ID。
        """

        raw = _read_json(directory / "cursors" / f"{recipient}.json")
        return MailboxCursor(int(raw.get("byte_offset", 0)), raw.get("last_message_id"))
