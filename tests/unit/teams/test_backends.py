"""验证后端自动选择顺序和显式选择失败语义。"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

from mycode.models.teams import BackendPreference, TeammateBackend
from mycode.teams.backends.detection import BackendDetectionError, BackendDetector
from mycode.teams.backends.base import BackendHandle, TeammateLaunch
from mycode.teams.backends.iterm2 import ITerm2Backend
from mycode.teams.backends.tmux import TmuxBackend
from mycode.teams.backends.subprocess import SubprocessBackend


def test_auto_uses_subprocess_when_no_pane_backend_exists(monkeypatch) -> None:
    monkeypatch.setattr("mycode.teams.backends.detection.shutil.which", lambda *args, **kwargs: None)

    selected = BackendDetector({"PATH": ""}).select(BackendPreference.AUTO)

    assert selected is TeammateBackend.SUBPROCESS


def test_explicit_tmux_unavailable_reports_error_without_fallback(monkeypatch) -> None:
    monkeypatch.setattr("mycode.teams.backends.detection.shutil.which", lambda *args, **kwargs: None)

    with pytest.raises(BackendDetectionError, match="显式指定 tmux"):
        BackendDetector({"PATH": ""}).select(BackendPreference.TMUX)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend,method,expected", [
    (TmuxBackend(), "_run_plain", ["tmux", "send-keys", "-t", "target", "", "Enter"]),
    (ITerm2Backend(), "_run", ["it2", "send-text", "--session", "target", "\n"]),
])
async def test_terminal_wake_targets_recorded_handle(backend, method, expected, monkeypatch):
    calls = []

    async def run(args, *rest):
        calls.append(args)
        return 0, "", ""

    monkeypatch.setattr(backend, method, run)
    await backend.wake(BackendHandle(backend.backend, "target"))
    assert calls == [expected]


async def wait_for_file(path):
    async def wait():
        while not path.exists():
            await asyncio.sleep(0.02)
    await asyncio.wait_for(wait(), 5)


def test_explicit_subprocess_requires_no_terminal():
    assert BackendDetector({"PATH": ""}).select(BackendPreference.SUBPROCESS) is TeammateBackend.SUBPROCESS


def test_agent_tool_accepts_subprocess_team_backend():
    from jsonschema import validate
    from mycode.agents.agent_tool import _AGENT_TOOL
    from mycode.models.agents import AgentToolRequest

    arguments = dict(prompt="检查代码", description="检查", name="reviewer",
                     subagent_type="reviewer", team_name="team", backend="subprocess")
    validate(arguments, _AGENT_TOOL.input_schema)
    assert AgentToolRequest(**arguments).backend == "subprocess"


@pytest.mark.asyncio
async def test_real_child_environment_wake_and_graceful_stop(tmp_path, monkeypatch):
    work = tmp_path / "成员 worktree"
    work.mkdir()
    backend = SubprocessBackend()
    launch = TeammateLaunch(tmp_path, work, "team", "member", 2, "secret-lease", "")
    args = backend._host_args(launch)
    assert args == [sys.executable, "-m", "mycode", "--team-host", "team", "member", "2"]
    script = (
        "import os,sys,json; from pathlib import Path; "
        "Path('ready.json').write_text(json.dumps([os.getpid(),os.getcwd(),"
        "os.environ['MYCODE_TEAM_ROOT'],os.environ['MYCODE_TEAM_LEASE']])); "
        "sys.stdin.readline(); Path('awake').touch(); sys.stdin.read(); Path('closed').touch()"
    )
    monkeypatch.setattr(backend, "_host_args", lambda _: [sys.executable, "-c", script])
    handle = await backend.start(launch)
    try:
        await wait_for_file(work / "ready.json")
        pid, cwd, root, lease = json.loads((work / "ready.json").read_text())
        assert pid == handle.process_id and pid != os.getpid()
        assert os.path.samefile(cwd, work)
        assert root == str(tmp_path) and lease == "secret-lease"
        assert (await backend.probe(handle)).alive
        await backend.wake(handle)
        await wait_for_file(work / "awake")
        await backend.stop(handle, force=False)
        assert (work / "closed").exists()
        assert not (await backend.probe(handle)).alive
        with pytest.raises(RuntimeError, match="已退出"):
            await backend.wake(handle)
    finally:
        await backend.stop(handle, force=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("force", [False, True])
async def test_unresponsive_child_is_reaped(tmp_path, monkeypatch, force):
    backend = SubprocessBackend(stop_timeout=0.05)
    monkeypatch.setattr(backend, "_host_args", lambda _: [sys.executable, "-c", "import time; time.sleep(60)"])
    handle = await backend.start(TeammateLaunch(tmp_path, tmp_path, "t", "a", 1, "lease", ""))
    process = backend._processes[handle.reference]
    try:
        await asyncio.wait_for(backend.stop(handle, force=force), 5)
        assert process.poll() is not None
        assert process.stdin.closed
    finally:
        await backend.stop(handle, force=True)


@pytest.mark.asyncio
async def test_stale_handle_does_not_target_reused_pid():
    backend = SubprocessBackend()
    handle = BackendHandle(backend.backend, "stale", os.getpid())
    assert not (await backend.probe(handle)).alive
    await backend.stop(handle, force=True)


@pytest.mark.asyncio
async def test_supervisor_closes_child_and_preserves_member(tmp_path, monkeypatch):
    from dataclasses import replace
    from unittest.mock import MagicMock

    from mycode.models.teams import TeammateState
    from mycode.teams.supervisor import TeammateSupervisor
    from tests.unit.teams.test_host import make_host

    env = make_host(tmp_path)
    backend = SubprocessBackend(stop_timeout=0.05)
    monkeypatch.setattr(backend, "_host_args", lambda _: [sys.executable, "-c", "import sys; sys.stdin.read()"])
    handle = await backend.start(env.launch)
    env.store.update_member(env.lead, env.member.actor_id, lambda member: replace(
        member, backend=backend.backend, backend_ref=handle.reference, owner_pid=handle.process_id,
    ))
    supervisor = TeammateSupervisor(
        workspace_root=tmp_path, store=env.store, tasks=MagicMock(), worktrees=MagicMock(),
        detector=BackendDetector(), adapters={backend.backend: backend}, session_creator=MagicMock(),
    )
    supervisor._handles[(env.lead.team_id, env.member.actor_id)] = handle
    try:
        await supervisor.close_local_hosts()
        member = env.store.load_team(env.lead.team_id).members[0]
        assert member.state is TeammateState.SUSPENDED
        assert member.backend_ref is None and member.owner_pid is None
        assert not (await backend.probe(handle)).alive
    finally:
        await backend.stop(handle, force=True)


def test_removed_backend_is_rejected_and_saved_members_migrate(tmp_path):
    from mycode.models.agents import AgentToolRequest
    from mycode.teams.store import _member_to_json, _member_from_json
    from tests.unit.teams.test_host import make_host

    with pytest.raises(ValueError):
        BackendPreference("in-process")
    with pytest.raises(ValueError, match="backend"):
        AgentToolRequest(prompt="p", description="d", name="n", subagent_type="r",
                         team_name="t", backend="in-process")
    env = make_host(tmp_path)
    raw = _member_to_json(env.store.load_team(env.lead.team_id).members[0])
    raw["backend"] = "in-process"
    assert _member_from_json(raw).backend is TeammateBackend.SUBPROCESS


@pytest.mark.asyncio
async def test_file_lock_serializes_real_processes(tmp_path, monkeypatch):
    script = """
import sys,time
from pathlib import Path
from mycode.teams.locks import ExclusiveFileLock
root=Path(sys.argv[1])
for _ in range(20):
    with ExclusiveFileLock(root/'counter.lock', 'child', max_attempts=500):
        path=root/'counter'
        value=int(path.read_text())
        time.sleep(0.002)
        path.write_text(str(value+1))
"""
    (tmp_path / "counter").write_text("0")
    backend = SubprocessBackend()
    monkeypatch.setattr(backend, "_host_args", lambda _: [sys.executable, "-c", script, str(tmp_path)])
    handles = []
    try:
        for _ in range(3):
            handles.append(await backend.start(TeammateLaunch(tmp_path, tmp_path, "t", "a", 1, "l", "")))
        for handle in handles:
            process = backend._processes[handle.reference]
            assert await asyncio.wait_for(asyncio.to_thread(process.wait), 15) == 0
        assert (tmp_path / "counter").read_text() == "60"
    finally:
        for handle in handles:
            await backend.stop(handle, force=True)


@pytest.mark.asyncio
async def test_file_lock_survives_contention_and_recovers_after_crash(tmp_path, monkeypatch):
    from mycode.teams.locks import ExclusiveFileLock, TeamLockError

    script = """
import sys,time
from pathlib import Path
from mycode.teams.locks import ExclusiveFileLock
root=Path(sys.argv[1])
with ExclusiveFileLock(root/'held.lock', 'child'):
    (root/'ready').touch()
    time.sleep(60)
"""
    backend = SubprocessBackend()
    monkeypatch.setattr(backend, "_host_args", lambda _: [sys.executable, "-c", script, str(tmp_path)])
    handle = await backend.start(TeammateLaunch(tmp_path, tmp_path, "t", "a", 1, "l", ""))
    try:
        await wait_for_file(tmp_path / "ready")
        with pytest.raises(TeamLockError):
            ExclusiveFileLock(tmp_path / "held.lock", "parent", max_attempts=1).acquire()
        assert (await backend.probe(handle)).alive
        await backend.stop(handle, force=True)
        with ExclusiveFileLock(tmp_path / "held.lock", "parent", max_attempts=1):
            pass
        assert (tmp_path / "held.lock").exists()
    finally:
        await backend.stop(handle, force=True)
