"""验证名称寻址、默认不唤醒和 cursor 推进。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from mycode.models.teams import SendMessageRequest, TeamMessageKind
from mycode.teams.mailbox import TeamMailbox, TeamMailboxError
from tests.unit.teams.support import add_member, create_team


@pytest.mark.asyncio
async def test_text_message_persists_without_default_wake(tmp_path) -> None:
    store, lead = create_team(tmp_path)
    member = add_member(store, lead)
    wakes: list[tuple[str, str, str]] = []

    async def wake(team_id: str, member_id: str, reason: str) -> None:
        wakes.append((team_id, member_id, reason))

    mailbox = TeamMailbox(store, wake)
    report = await mailbox.send(
        lead,
        SendMessageRequest(
            to="alice",
            summary="接口变化",
            message="Authenticate 新增 ctx 参数",
        ),
    )

    assert report.deliveries[0].delivered is True
    assert wakes == []
    messages = mailbox.drain_for_agent(member)
    assert len(messages) == 1
    assert "Authenticate 新增 ctx 参数" in messages[0].content
    assert mailbox.read_unread(member) == ()


@pytest.mark.asyncio
async def test_plain_text_requires_summary(tmp_path) -> None:
    store, lead = create_team(tmp_path)
    add_member(store, lead)
    mailbox = TeamMailbox(store)

    with pytest.raises(TeamMailboxError, match="摘要"):
        await mailbox.send(
            lead,
            SendMessageRequest(to="alice", message="没有摘要"),
        )


@pytest.mark.asyncio
async def test_member_cannot_send_plan_response(tmp_path) -> None:
    store, lead = create_team(tmp_path)
    member = add_member(store, lead)
    mailbox = TeamMailbox(store)

    with pytest.raises(TeamMailboxError, match="只有 Lead"):
        await mailbox.send(
            member,
            SendMessageRequest(
                to="lead",
                kind=TeamMessageKind.PLAN_RESPONSE,
                message="批准",
                payload={
                    "task_id": "task-1",
                    "attempt_number": 1,
                    "plan_revision": 1,
                    "decision": "approved",
                },
            ),
        )


@pytest.mark.asyncio
async def test_acknowledge_keeps_messages_appended_after_read(tmp_path) -> None:
    """确认中文消息后，处理期间到达的下一条消息仍未读。"""
    store, lead = create_team(tmp_path)
    member = add_member(store, lead)
    mailbox = TeamMailbox(store)
    empty_cursor = mailbox.acknowledge(member, ())
    assert empty_cursor.byte_offset == 0
    await mailbox.send(lead, SendMessageRequest(to="alice", summary="第一批", message="中文正文🙂"))
    batch = mailbox.read_unread(member)
    path = store.team_dir(lead.team_id) / "mailboxes" / "agent-alice.jsonl"
    expected_offset = path.stat().st_size
    await mailbox.send(lead, SendMessageRequest(to="alice", summary="第二批", message="处理期间到达"))

    cursor = mailbox.acknowledge(member, batch)

    assert cursor.byte_offset == expected_offset
    assert cursor.last_message_id == batch[-1].message_id
    assert mailbox.acknowledge(member, ()) == cursor
    remaining = mailbox.read_unread(member)
    assert [message.body for message in remaining] == ["处理期间到达"]
    mailbox.acknowledge(member, remaining)
    assert mailbox.read_unread(member) == ()


@pytest.mark.asyncio
async def test_acknowledge_unknown_message_does_not_advance_cursor(tmp_path) -> None:
    store, lead = create_team(tmp_path)
    member = add_member(store, lead)
    mailbox = TeamMailbox(store)
    await mailbox.send(lead, SendMessageRequest(to="alice", summary="消息", message="正文"))
    batch = mailbox.read_unread(member)
    with pytest.raises(TeamMailboxError, match="确认"):
        mailbox.acknowledge(member, (replace(batch[0], message_id="missing"),))
    assert mailbox.read_unread(member) == batch
