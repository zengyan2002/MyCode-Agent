"""验证后端自动选择顺序和显式选择失败语义。"""

from __future__ import annotations

import asyncio

import pytest

from mycode.models.teams import BackendPreference, TeammateBackend
from mycode.teams.backends.detection import BackendDetectionError, BackendDetector
from mycode.teams.backends.base import TeammateLaunch
from mycode.teams.backends.in_process import InProcessBackend


def test_auto_uses_in_process_when_no_pane_backend_exists(monkeypatch) -> None:
    monkeypatch.setattr("mycode.teams.backends.detection.shutil.which", lambda *args, **kwargs: None)

    selected = BackendDetector({"PATH": ""}).select(BackendPreference.AUTO)

    assert selected is TeammateBackend.IN_PROCESS


def test_explicit_tmux_unavailable_reports_error_without_fallback(monkeypatch) -> None:
    monkeypatch.setattr("mycode.teams.backends.detection.shutil.which", lambda *args, **kwargs: None)

    with pytest.raises(BackendDetectionError, match="显式指定 tmux"):
        BackendDetector({"PATH": ""}).select(BackendPreference.TMUX)


@pytest.mark.asyncio
async def test_in_process_wake_is_consumed_once(tmp_path) -> None:
    """重复通知可合并，但下一次等待必须等到新的通知。"""
    first = asyncio.Event()
    second = asyncio.Event()

    async def host(launch, wait_for_wake):
        await wait_for_wake()
        first.set()
        await wait_for_wake()
        second.set()
        await asyncio.Future()

    backend = InProcessBackend(host)
    launch = TeammateLaunch(tmp_path, tmp_path, "team", "member", 1, "lease", "")
    handle = await backend.start(launch)
    try:
        await backend.wake(handle)
        await backend.wake(handle)
        await asyncio.wait_for(first.wait(), 2)
        assert not second.is_set()
        assert not backend.wake_event(handle.reference).is_set()
        await backend.wake(handle)
        await asyncio.wait_for(second.wait(), 2)
        assert not backend.wake_event(handle.reference).is_set()
    finally:
        await backend.stop(handle, force=True)
