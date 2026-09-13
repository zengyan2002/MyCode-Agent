"""通过轮次屏障验证成员收件、等待和消费确认。"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mycode.models.teams import SendMessageRequest, TeamMessageKind, TeammateState
from mycode.teams.backends.base import TeammateLaunch
from mycode.teams.host import TeammateHost
from mycode.teams.mailbox import TeamMailbox
from tests.unit.teams.support import add_member, create_team


class ControlledRuntime:
    """把每轮输入和放行 Future 交给测试，控制真实 Host 的执行边界。"""

    def __init__(self):
        self.turns = asyncio.Queue()
        self.active = 0
        self.max_active = 0
        self.closed = False

    async def run(self, prompt):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        done = asyncio.get_running_loop().create_future()
        self.turns.put_nowait((prompt, done))
        try:
            await done
            return "完成"
        finally:
            self.active -= 1

    def close(self):
        self.closed = True


class ControlledWaiter:
    """记录后端等待的创建与结束，不依赖固定 sleep 推测状态。"""

    def __init__(self):
        self.event = asyncio.Event()
        self.started = asyncio.Queue()
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def __call__(self):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.put_nowait(self.calls)
        try:
            await self.event.wait()
            self.event.clear()
        finally:
            self.active -= 1


def make_host(tmp_path, prompt="首轮工作"):
    """装配真实邮箱、成员租约和 Host，仅模型执行由测试控制。"""
    store, lead = create_team(tmp_path)
    member = add_member(store, lead)
    lease = "host-test-lease"
    store.update_member(lead, member.actor_id, lambda current: replace(
        current, lease_token_hash=hashlib.sha256(lease.encode()).hexdigest(),
    ))
    store.save_runtime_prompt(lead.team_id, member.actor_id, prompt)
    mailbox = TeamMailbox(store)
    runtime = ControlledRuntime()

    async def load_runtime(team_id, member_id):
        return runtime

    host = TeammateHost(store, mailbox, load_runtime)
    launch = TeammateLaunch(tmp_path, tmp_path, lead.team_id, member.actor_id, 1, lease, prompt)
    return SimpleNamespace(store=store, lead=lead, member=member, mailbox=mailbox,
                           runtime=runtime, host=host, launch=launch, waiter=ControlledWaiter())


async def next_turn(env):
    return await asyncio.wait_for(env.runtime.turns.get(), 3)


async def send(env, body, *, wake=True, kind=TeamMessageKind.TEXT):
    report = await env.mailbox.send(env.lead, SendMessageRequest(
        to=env.member.actor_id, summary=body, message=body, wake=wake, kind=kind,
    ))
    assert report.deliveries[0].delivered


async def stop_host(task):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_busy_host_drains_new_messages_after_each_turn(tmp_path):
    env = make_host(tmp_path)
    task = asyncio.create_task(env.host(env.launch, env.waiter))
    try:
        first, release_first = await next_turn(env)
        assert first == "首轮工作"
        await send(env, "第二轮消息")
        assert env.runtime.turns.empty()
        release_first.set_result(None)
        second, release_second = await next_turn(env)
        assert "第二轮消息" in second
        await send(env, "第三轮消息")
        release_second.set_result(None)
        third, release_third = await next_turn(env)
        assert "第三轮消息" in third
        assert "第二轮消息" not in third
        release_third.set_result(None)
        await asyncio.wait_for(env.waiter.started.get(), 3)
        assert env.mailbox.read_unread(env.member) == ()
        assert env.runtime.max_active == 1
    finally:
        await stop_host(task)
    assert env.runtime.closed


@pytest.mark.asyncio
async def test_host_acknowledges_batch_before_shutdown(tmp_path):
    env = make_host(tmp_path)
    await send(env, "退出", kind=TeamMessageKind.SHUTDOWN_REQUEST)
    task = asyncio.create_task(env.host(env.launch, env.waiter))
    try:
        _, release = await next_turn(env)
        release.set_result(None)
        await asyncio.wait_for(task, 3)
        assert env.mailbox.read_unread(env.member) == ()
        assert env.runtime.closed
    finally:
        await stop_host(task)


@pytest.mark.asyncio
async def test_failed_turn_keeps_messages_unread(tmp_path):
    env = make_host(tmp_path)
    await send(env, "尚未完成")
    task = asyncio.create_task(env.host(env.launch, env.waiter))
    try:
        _, release = await next_turn(env)
        release.set_exception(RuntimeError("模型失败"))
        with pytest.raises(RuntimeError, match="模型失败"):
            await asyncio.wait_for(task, 3)
        assert [m.body for m in env.mailbox.read_unread(env.member)] == ["尚未完成"]
        assert env.store.load_team(env.lead.team_id).members[0].state is TeammateState.FAILED
        assert env.runtime.closed
    finally:
        await stop_host(task)


@pytest.mark.asyncio
async def test_idle_host_polls_only_wake_messages_and_reuses_waiter(tmp_path, monkeypatch):
    env = make_host(tmp_path, prompt="")
    task = asyncio.create_task(env.host(env.launch, env.waiter))
    try:
        await asyncio.wait_for(env.waiter.started.get(), 3)
        reads = asyncio.Queue()
        original_read = env.mailbox.read_unread

        def observed_read(actor):
            messages = original_read(actor)
            reads.put_nowait(messages)
            return messages

        monkeypatch.setattr(env.mailbox, "read_unread", observed_read)
        await send(env, "普通消息", wake=False)
        for _ in range(2):
            batch = await asyncio.wait_for(reads.get(), 3)
            assert [m.body for m in batch] == ["普通消息"]
            assert env.runtime.turns.empty()
        await send(env, "请处理", wake=True)
        prompt, release = await next_turn(env)
        assert "普通消息" in prompt and "请处理" in prompt
        assert env.waiter.calls == 1
        release.set_result(None)
        # 观察实际空邮箱检查，确认处理结束后继续等待而非反复启动模型。
        for _ in range(2):
            while await asyncio.wait_for(reads.get(), 3):
                pass
        assert env.waiter.calls == 1
        assert env.waiter.max_active == 1
        assert env.runtime.turns.empty()
    finally:
        await stop_host(task)
    assert env.waiter.active == 0
    assert env.runtime.closed


@pytest.mark.asyncio
async def test_message_between_empty_check_and_wait_is_not_lost(tmp_path, monkeypatch):
    env = make_host(tmp_path, prompt="")
    original_read = env.mailbox.read_unread
    delivery = None

    def deliver_after_empty_read(actor):
        nonlocal delivery
        messages = original_read(actor)
        if delivery is None:
            assert not messages
            delivery = asyncio.create_task(send(env, "检查之后到达"))
        return messages

    monkeypatch.setattr(env.mailbox, "read_unread", deliver_after_empty_read)
    task = asyncio.create_task(env.host(env.launch, env.waiter))
    try:
        prompt, release = await next_turn(env)
        assert "检查之后到达" in prompt
        release.set_result(None)
        assert delivery is not None
        await delivery
    finally:
        await stop_host(task)
        if delivery is not None:
            await delivery


@pytest.mark.asyncio
async def test_backend_wait_error_is_reported_and_runtime_closed(tmp_path):
    env = make_host(tmp_path, prompt="")

    async def failed_wait():
        raise RuntimeError("终端已关闭")

    with pytest.raises(RuntimeError, match="终端已关闭"):
        await asyncio.wait_for(env.host(env.launch, failed_wait), 3)
    assert env.runtime.closed
    assert env.store.load_team(env.lead.team_id).members[0].state is TeammateState.FAILED


@pytest.mark.asyncio
async def test_cancel_idle_host_reclaims_waiter(tmp_path):
    env = make_host(tmp_path, prompt="")
    task = asyncio.create_task(env.host(env.launch, env.waiter))
    await asyncio.wait_for(env.waiter.started.get(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert env.waiter.active == 0
    assert env.runtime.closed
