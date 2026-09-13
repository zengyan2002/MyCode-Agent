"""运行真实团队通信组件，模型执行仅用 Future 控制完成时机。"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mycode.models.teams import TeammateBackend, TeammateState
from mycode.teams.backends.in_process import InProcessBackend
from mycode.teams.mailbox import TeamMailbox
from mycode.teams.message_tool import SendMessageTool
from mycode.teams.supervisor import TeammateSupervisor
from mycode.teams.tasks import TeamTaskBoard
from mycode.tools.base import ToolContext
from tests.unit.teams.support import add_member
from tests.unit.teams.test_host import make_host, next_turn


async def start_member(env, tmp_path):
    """使用真实同进程后端，同时观察 Host 的等待和退出。"""
    waiting = asyncio.Queue()
    finished = asyncio.Event()

    async def run_host(launch, wait_for_wake):
        async def observed_wait():
            waiting.put_nowait(None)
            await wait_for_wake()

        try:
            await env.host(launch, observed_wait)
        finally:
            finished.set()

    backend = InProcessBackend(run_host)
    handle = await backend.start(env.launch)
    env.store.update_member(env.lead, env.member.actor_id, lambda member: replace(
        member, backend_ref=handle.reference,
    ))
    supervisor = TeammateSupervisor(
        workspace_root=tmp_path, store=env.store, tasks=TeamTaskBoard(env.store),
        worktrees=MagicMock(), detector=MagicMock(),
        adapters={TeammateBackend.IN_PROCESS: backend}, session_creator=MagicMock(),
    )
    return backend, handle, supervisor, waiting, finished


async def deliver(tool, context, *, body, kind="text", wake=True):
    output = await tool.execute({
        "to": "alice", "kind": kind, "summary": body, "message": body, "wake": wake,
    }, context)
    assert output.success, output.error_message
    assert json.loads(output.content)[0]["delivered"]


@pytest.mark.asyncio
@pytest.mark.parametrize("with_callback", [True, False])
async def test_member_messages_are_drained_after_busy_turn(tmp_path, with_callback):
    env = make_host(tmp_path)
    sender = add_member(env.store, env.lead, agent_id="agent-bob", name="bob")
    backend, handle, supervisor, waiting, finished = await start_member(env, tmp_path)
    wake_results = []

    async def wake(team_id, member_id, reason):
        wake_results.append(await supervisor.wake(team_id, member_id))

    tool = SendMessageTool(TeamMailbox(env.store, wake if with_callback else None))
    context = ToolContext(tmp_path, team_actor=sender)
    try:
        _, first = await next_turn(env)
        await deliver(tool, context, body="成员消息一")
        assert env.runtime.turns.empty()
        if with_callback:
            assert wake_results == [False]  # 正在运行；通知由持久化消息保留。
        first.set_result(None)
        prompt, second = await next_turn(env)
        assert "成员消息一" in prompt
        await deliver(tool, context, body="成员消息二")
        second.set_result(None)
        prompt, third = await next_turn(env)
        assert "成员消息二" in prompt and "成员消息一" not in prompt
        third.set_result(None)
        await asyncio.wait_for(waiting.get(), 3)
        assert env.mailbox.read_unread(env.member) == ()
        assert env.runtime.max_active == 1
        await deliver(tool, context, body="正常退出", kind="shutdown_request")
        _, shutdown = await next_turn(env)
        shutdown.set_result(None)
        await asyncio.wait_for(finished.wait(), 3)
        assert env.mailbox.read_unread(env.member) == ()
        assert env.runtime.closed
    finally:
        await backend.stop(handle, force=True)


@pytest.mark.asyncio
async def test_subprocess_sender_wakes_idle_member_without_callback(tmp_path):
    env = make_host(tmp_path, prompt="")
    sender = add_member(env.store, env.lead, agent_id="agent-bob", name="bob")
    backend, handle, _, waiting, finished = await start_member(env, tmp_path)
    process = None
    try:
        await asyncio.wait_for(waiting.get(), 3)
        # 通过参数传路径和身份，子进程真实调用工具；不使用 shell 拼接。
        script = """
import asyncio, json, sys
from pathlib import Path
from mycode.models.teams import TeamActorContext
from mycode.teams.store import TeamStateStore
from mycode.teams.mailbox import TeamMailbox
from mycode.teams.message_tool import SendMessageTool
from mycode.tools.base import ToolContext
root = Path(sys.argv[1])
actor = TeamActorContext(sys.argv[2], sys.argv[3], 'member', 1)
tool = SendMessageTool(TeamMailbox(TeamStateStore(root)))
output = asyncio.run(tool.execute(
    {'to': 'alice', 'summary': 'from process', 'message': 'cross-process message', 'wake': True},
    ToolContext(root, team_actor=actor)))
assert output.success, output.error_message
assert json.loads(output.content)[0]['delivered']
"""
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script, str(tmp_path), env.lead.team_id, sender.actor_id,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(process.communicate(), 10)
        assert process.returncode == 0, stderr.decode(errors="replace")
        prompt, release = await next_turn(env)
        assert "cross-process message" in prompt
        release.set_result(None)
        tool = SendMessageTool(TeamMailbox(env.store))
        await deliver(tool, ToolContext(tmp_path, team_actor=sender), body="退出", kind="shutdown_request")
        _, shutdown = await next_turn(env)
        shutdown.set_result(None)
        await asyncio.wait_for(finished.wait(), 3)
        assert env.mailbox.read_unread(env.member) == ()
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        await backend.stop(handle, force=True)


@pytest.mark.asyncio
async def test_message_to_terminated_member_does_not_restart_it(tmp_path):
    env = make_host(tmp_path, prompt="")
    backend, handle, supervisor, waiting, _ = await start_member(env, tmp_path)
    await asyncio.wait_for(waiting.get(), 3)
    await backend.stop(handle, force=True)
    env.store.update_member(env.lead, env.member.actor_id,
                           lambda member: replace(member, state=TeammateState.TERMINATED))
    wake_results = []

    async def wake(team_id, member_id, reason):
        wake_results.append(await supervisor.wake(team_id, member_id))

    await deliver(SendMessageTool(TeamMailbox(env.store, wake)),
                  ToolContext(tmp_path, team_actor=env.lead), body="结束后送达")
    assert wake_results == [False]
    assert not (await backend.probe(handle)).alive
    assert env.runtime.turns.empty()
    assert env.store.load_team(env.lead.team_id).members[0].state is TeammateState.TERMINATED


@pytest.mark.asyncio
async def test_initialized_idle_host_completes_supervisor_handshake(tmp_path, monkeypatch):
    """恢复后没有初始提示的真实 Host 已经就绪，不应因空闲而握手超时。"""
    monkeypatch.setattr("mycode.teams.supervisor._HOST_HANDSHAKE_TIMEOUT_SECONDS", 0.3)
    env = make_host(tmp_path, prompt="")
    backend, handle, supervisor, waiting, _ = await start_member(env, tmp_path)
    try:
        await asyncio.wait_for(waiting.get(), 3)
        member = await supervisor._await_host_handshake(
            env.lead.team_id, env.member.actor_id, backend, handle,
        )
        assert member.state is TeammateState.IDLE
        assert env.runtime.turns.empty()
    finally:
        await backend.stop(handle, force=True)
