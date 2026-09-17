"""验证工具摘要和短窗口停滞判断。"""

from dataclasses import replace

import pytest

from mycode.agent.progress import action_fingerprint, result_fingerprint
from mycode.models.messages import ToolCall
from mycode.models.tools import ToolErrorCode, ToolExecutionResult
from mycode.agent.progress import ProgressMonitor, ProgressSignal, ProgressSeverity
from mycode.constants import (
    PROGRESS_TRACE_MAX_STEPS, PROGRESS_WARN_AFTER_ROUNDS, PROGRESS_REPLAN_AFTER_ROUNDS,
)
from mycode.models.tools import ToolAccess, ToolInvocation


def monitor():
    return ProgressMonitor(max_steps=PROGRESS_TRACE_MAX_STEPS,
                           warn_after_rounds=PROGRESS_WARN_AFTER_ROUNDS,
                           replan_after_rounds=PROGRESS_REPLAN_AFTER_ROUNDS)


def observe(current, *steps):
    invocations = tuple(ToolInvocation(ToolCall(str(i), "read", {"path": name}),
                                      access, 1, i)
                        for i, (name, output, access) in enumerate(steps))
    return current.observe_round(invocations, tuple(output for _, output, _ in steps))


def read(name="a", **changes):
    return name, result(**changes), ToolAccess.READ


def result(**changes):
    return replace(ToolExecutionResult(
        tool_call_id="one", tool_name="read", success=True, content="正文",
        error_code=None, error_message=None, timed_out=False, truncated=False,
        original_size_bytes=6, duration_ms=10, metadata={"a": 1, "b": 2},
    ), **changes)


def test_action_canonicalization_and_call_id():
    first = ToolCall("one", "read", {"a": {"y": 2, "x": 1}, "b": [1, 2]})
    second = ToolCall("two", "read", {"b": [1, 2], "a": {"x": 1, "y": 2}})
    assert action_fingerprint(first) == action_fingerprint(second)


@pytest.mark.parametrize("call", [
    ToolCall("one", "grep", {"a": 1}),
    ToolCall("one", "read", {"a": 2}),
])
def test_action_changes(call):
    assert action_fingerprint(ToolCall("one", "read", {"a": 1})) != action_fingerprint(call)


def test_result_ignores_transient_fields_and_metadata_order():
    assert result_fingerprint(result()) == result_fingerprint(result(
        tool_call_id="two", duration_ms=800, metadata={"b": 2, "a": 1},
    ))


@pytest.mark.parametrize("changes", [
    {"tool_name": "grep"}, {"content": "变化"},
    {"success": False, "error_code": ToolErrorCode.NOT_FOUND, "error_message": "missing"},
    {"timed_out": True}, {"truncated": True}, {"original_size_bytes": 100},
    {"metadata": {"a": 2}},
])
def test_result_payload_changes(changes):
    assert result_fingerprint(result()) != result_fingerprint(result(**changes))


def test_result_error_details_change():
    failure = result(success=False, error_code=ToolErrorCode.NOT_FOUND, error_message="missing")
    for changed in (replace(failure, error_code=ToolErrorCode.IO_ERROR),
                    replace(failure, error_message="different")):
        assert result_fingerprint(failure) != result_fingerprint(changed)


def test_streak_thresholds_and_recovery():
    current = monitor()
    decisions = [observe(current, read()) for _ in range(6)]
    assert [d.no_progress_streak for d in decisions] == [0, 1, 2, 3, 4, 5]
    assert [d.severity for d in decisions] == [ProgressSeverity.NONE] * 2 + [
        ProgressSeverity.NOTICE, *([ProgressSeverity.REPLAN] * 3)]
    assert [d.emit_warning for d in decisions] == [False, False, False, True, False, False]
    assert all(d.message for d in decisions[2:])
    assert observe(current, read(content="new")).no_progress_streak == 0
    assert [observe(current, read(content="new")).emit_warning for _ in range(3)] == [False, False, True]


def test_successful_write_and_empty_round_reset():
    current = monitor()
    observe(current, read())
    observe(current, read())
    assert observe(current, ("a", result(), ToolAccess.WRITE)).no_progress_streak == 0
    assert observe(current, read()).no_progress_streak == 1
    assert observe(current).no_progress_streak == 0


def test_repeated_failure_and_changed_error():
    current = monitor()
    failure = read(success=False, error_code=ToolErrorCode.INVALID_PATTERN, error_message="bad")
    assert observe(current, failure).no_progress_streak == 0
    assert observe(current, failure).no_progress_streak == 1
    assert observe(current, read(success=False, error_code=ToolErrorCode.IO_ERROR,
                                 error_message="bad")).no_progress_streak == 0


def test_window_is_bounded_and_old_actions_are_forgotten():
    current = monitor()
    for index in range(101):
        assert observe(current, read(str(index))).no_progress_streak == 0
    assert len(current._steps) == 8
    assert all(not hasattr(step, "content") and not hasattr(step, "arguments") for step in current._steps)
    assert observe(current, read("0")).no_progress_streak == 0
    assert observe(monitor(), read("100")).no_progress_streak == 0


def test_oscillation_requires_distinct_actions_and_equal_results():
    current = monitor()
    for name in ("a", "b", "a", "b"):
        decision = observe(current, read(name))
    assert decision.signal is ProgressSignal.OSCILLATION
    repeated = monitor()
    for _ in range(4):
        decision = observe(repeated, read())
    assert decision.signal is ProgressSignal.REPEATED_ACTION


def test_oscillation_changed_result():
    current = monitor()
    for step in (read("a"), read("b"), read("a", content="changed"), read("b")):
        decision = observe(current, step)
    assert decision.signal is not ProgressSignal.OSCILLATION


def test_mixed_round_new_information_wins_over_abab_suffix():
    current = monitor()
    observe(current, read("a"), read("b"))
    decision = observe(current, read("new"), read("a"), read("b"), read("a"), read("b"))
    assert decision.signal is ProgressSignal.NONE
    assert decision.no_progress_streak == 0


def test_first_batch_repeats_do_not_count_as_stalled_round():
    current = monitor()
    assert observe(current, *([read()] * 12)).no_progress_streak == 0
    assert observe(current, read()).no_progress_streak == 1


def test_compare_only_latest_result_for_action():
    current = monitor()
    for content in ("x", "y", "x"):
        assert observe(current, read(content=content)).no_progress_streak == 0
