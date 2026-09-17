"""通过真实 AgentLoop/Runner 验证进度提醒与工具轮次提交。"""

import json
from dataclasses import replace

import pytest

from mycode.agent.progress import PROGRESS_NOTICE, PROGRESS_REPLAN, ProgressMonitor
from mycode.agent.cancellation import CancellationToken
from mycode.errors import ContextWindowExceededError
from mycode.models.events import AgentWarningEvent, FinalReplyEvent, AgentRunOptions, ToolStartedEvent
from mycode.models.messages import TextBlock, ToolCall, ToolResultMessage
from mycode.models.provider import ModelStopReason, ToolChoice
from mycode.models.tools import ToolAccess, ToolErrorCode
from mycode.models.prompts import RuntimeInstruction, RuntimeInstructionKind
from mycode.models.model_calls import ModelCallPurpose
from mycode.context.manager import CompactionOutcome, CompactionOutcomeKind
from mycode.tools.base import ToolOutput
from tests.unit.agent.test_loop import (
    FakeProvider, FakeContextManager, ScriptedTool, build_agent, collect, completed, history,
)


def tool_response(index, name="read"):
    return completed(ModelStopReason.TOOL_USE, ToolCall(f"call-{index}", name, {}))


def progress_messages(request):
    return [item.content for item in request.prompt.runtime
            if item.content in (PROGRESS_NOTICE, PROGRESS_REPLAN)]


@pytest.mark.asyncio
async def test_tool_round_commits_results_and_final_answer(tmp_path):
    tools = [ScriptedTool("read", ToolAccess.READ), ScriptedTool("other", ToolAccess.READ)]
    provider = FakeProvider([
        completed(ModelStopReason.TOOL_USE, ToolCall("a", "read", {}), ToolCall("b", "other", {})),
        completed(ModelStopReason.END_TURN, TextBlock("done")),
    ])
    agent = build_agent(tmp_path, provider, tools)
    events = await collect(agent, "inspect")
    assert [t.calls for t in tools] == [1, 1]
    assert len(provider.requests) == 2
    assert any(isinstance(e, FinalReplyEvent) and e.text == "done" for e in events)
    assert len([m for m in history(agent) if isinstance(m, ToolResultMessage)]) == 2
    assert not any(isinstance(e, AgentWarningEvent) for e in events)


@pytest.mark.asyncio
async def test_repeated_rounds_warn_once_and_recover(tmp_path):
    names = ["read"] * 6 + ["other"] + ["read"] * 3
    tools = [ScriptedTool(name, ToolAccess.READ) for name in ("read", "other")]
    provider = FakeProvider([*(tool_response(i, name) for i, name in enumerate(names)),
                             completed(ModelStopReason.END_TURN, TextBlock("done"))])
    agent = build_agent(tmp_path, provider, tools)
    events = await collect(agent, "inspect")
    assert [progress_messages(r) for r in provider.requests] == [
        [], [], [], [PROGRESS_NOTICE], [PROGRESS_REPLAN], [PROGRESS_REPLAN],
        [PROGRESS_REPLAN], [], [], [PROGRESS_NOTICE], [PROGRESS_REPLAN],
    ]
    assert [e.message for e in events if isinstance(e, AgentWarningEvent)] == [PROGRESS_REPLAN] * 2
    assert len(provider.requests) == len(names) + 1
    assert sum(t.calls for t in tools) == len(names)
    results = [m for m in history(agent) if isinstance(m, ToolResultMessage)]
    assert [json.loads(m.content)["tool_call_id"] for m in results] == [f"call-{i}" for i in range(len(names))]
    assert all(PROGRESS_REPLAN not in str(m) and PROGRESS_NOTICE not in str(m) for m in history(agent))


@pytest.mark.asyncio
async def test_notice_survives_context_retry_but_not_next_response(tmp_path):
    provider = FakeProvider([*(tool_response(i) for i in range(3)),
        ContextWindowExceededError("too long"), tool_response(3, "other"),
        completed(ModelStopReason.END_TURN, TextBlock("done"))])
    context = FakeContextManager()
    agent = build_agent(tmp_path, provider,
                        [ScriptedTool(n, ToolAccess.READ) for n in ("read", "other")],
                        context_manager=context)
    await collect(agent, "inspect")
    assert len(provider.requests) == 6
    assert progress_messages(provider.requests[3]) == [PROGRESS_NOTICE]
    assert progress_messages(provider.requests[4]) == [PROGRESS_NOTICE]
    assert progress_messages(provider.requests[5]) == []
    assert len(context.modes) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", [False, True])
async def test_finalization_wins_over_progress_instruction(tmp_path, retry):
    responses = [tool_response(i) for i in range(3)]
    if retry:
        responses.append(ContextWindowExceededError("too long"))
    responses.append(completed(ModelStopReason.END_TURN, TextBlock("<final-report>done</final-report>")))
    provider = FakeProvider(responses)
    context = FakeContextManager()
    if retry:
        async def compact(mode, cancellation, *, model_call_budget, **kwargs):
            number = model_call_budget.begin(ModelCallPurpose.COMPACTION)
            record = model_call_budget.finish(number, None)
            return CompactionOutcome(CompactionOutcomeKind.SUCCEEDED, "压缩成功",
                                     model_call_records=(record,))
        context.compact = compact
    agent = build_agent(tmp_path, provider, [ScriptedTool("read", ToolAccess.READ)], context_manager=context)
    events = await collect(agent, "inspect", options=AgentRunOptions(max_model_calls=6 if retry else 4))
    last = provider.requests[-1]
    assert last.tool_choice is ToolChoice.NONE
    assert not last.tools and not progress_messages(last)
    assert isinstance(events[-1], FinalReplyEvent)
    assert events[-1].model_calls == (6 if retry else 4)
    if retry:
        assert progress_messages(provider.requests[-2]) == [PROGRESS_NOTICE]


@pytest.mark.asyncio
async def test_fixed_runtime_filters_parent_progress_and_keeps_other_notices(tmp_path, monkeypatch):
    provider = FakeProvider([*(tool_response(i) for i in range(3)),
        ContextWindowExceededError("too long"),
        completed(ModelStopReason.END_TURN, TextBlock("done"))])
    context = FakeContextManager()
    agent = build_agent(tmp_path, provider, [ScriptedTool("read", ToolAccess.READ)], context_manager=context)
    turn_runner = agent._turn_runner
    original = turn_runner.stream
    ordinary = RuntimeInstruction(RuntimeInstructionKind.RUNTIME_NOTICE, "ordinary")
    parent = RuntimeInstruction(RuntimeInstructionKind.RUNTIME_NOTICE, PROGRESS_REPLAN)

    async def fixed_stream(run):
        async for event in original(replace(run, fixed_runtime=(ordinary, parent))):
            yield event

    monkeypatch.setattr(turn_runner, "stream", fixed_stream)
    await collect(agent, "inspect")
    assert [progress_messages(r) for r in provider.requests] == [[], [], [], [PROGRESS_NOTICE], [PROGRESS_NOTICE]]
    assert all(ordinary in r.prompt.runtime for r in provider.requests)
    assert all(parent not in r.prompt.runtime for r in context.preview_requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["cancel", "deadline", "close", "exception"])
async def test_aborted_tool_round_is_not_observed(tmp_path, monkeypatch, ending):
    observed = []
    original = ProgressMonitor.observe_round

    def record(self, invocations, results):
        observed.append(results)
        return original(self, invocations, results)

    monkeypatch.setattr(ProgressMonitor, "observe_round", record)
    provider = FakeProvider([tool_response(0)])
    tool = ScriptedTool("read", ToolAccess.READ, delay=0.3)
    agent = build_agent(tmp_path, provider, [tool])
    token = CancellationToken()
    if ending == "exception":
        from mycode.tools.scheduler import ToolScheduleSession
        original_stream = ToolScheduleSession.stream

        async def broken_stream(self):
            async for event in original_stream(self):
                yield event
                raise RuntimeError("test stream failure")

        monkeypatch.setattr(ToolScheduleSession, "stream", broken_stream)
        with pytest.raises(RuntimeError, match="test stream failure"):
            await collect(agent, "inspect")
    else:
        stream = agent.stream_turn("inspect", cancellation=token,
            options=AgentRunOptions(overall_timeout_seconds=0.05 if ending == "deadline" else None))
        async for event in stream:
            if isinstance(event, ToolStartedEvent):
                if ending == "cancel":
                    token.cancel()
                elif ending == "close":
                    await stream.aclose()
                    break
    assert observed == []
    results = [m for m in history(agent) if isinstance(m, ToolResultMessage)]
    assert len(results) == 1
    assert json.loads(results[0].content)["error_code"] == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [ToolErrorCode.INVALID_PATTERN, ToolErrorCode.TIMEOUT])
async def test_normal_failed_results_are_observed(tmp_path, monkeypatch, code):
    tool = ScriptedTool("read", ToolAccess.READ)

    async def fail(arguments, context):
        return ToolOutput.fail(code, "same failure")

    monkeypatch.setattr(tool, "execute", fail)
    provider = FakeProvider([*(tool_response(i) for i in range(4)),
                            completed(ModelStopReason.END_TURN, TextBlock("done"))])
    agent = build_agent(tmp_path, provider, [tool])
    await collect(agent, "inspect")
    assert progress_messages(provider.requests[-1]) == [PROGRESS_REPLAN]


@pytest.mark.asyncio
async def test_main_turn_counter_is_not_inherited(tmp_path):
    provider = FakeProvider([*(tool_response(i) for i in range(3)),
        completed(ModelStopReason.END_TURN, TextBlock("one")),
        tool_response(4), completed(ModelStopReason.END_TURN, TextBlock("two"))])
    agent = build_agent(tmp_path, provider, [ScriptedTool("read", ToolAccess.READ)])
    await collect(agent, "first")
    await collect(agent, "second")
    assert progress_messages(provider.requests[3]) == [PROGRESS_NOTICE]
    assert all(not progress_messages(r) for r in provider.requests[4:])


@pytest.mark.asyncio
async def test_real_file_edit_then_read_does_not_warn(tmp_path):
    from mycode.tools.builtin import create_builtin_registry

    (tmp_path / "sample.txt").write_text("version 1", encoding="utf-8")
    calls = [
        ToolCall("read-1", "read_file", {"path": "sample.txt"}),
        ToolCall("edit", "edit_file", {"path": "sample.txt", "old_text": "version 1", "new_text": "version 2"}),
        ToolCall("read-2", "read_file", {"path": "sample.txt"}),
    ]
    provider = FakeProvider([*(completed(ModelStopReason.TOOL_USE, call) for call in calls),
                             completed(ModelStopReason.END_TURN, TextBlock("done"))])
    agent = build_agent(tmp_path, provider, registry=create_builtin_registry())
    events = await collect(agent, "update file")
    outputs = [json.loads(m.content) for m in history(agent) if isinstance(m, ToolResultMessage)]
    assert all(output["success"] for output in outputs)
    assert "version 1" in outputs[0]["content"] and "version 2" in outputs[2]["content"]
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "version 2"
    assert not any(progress_messages(r) for r in provider.requests)
    assert not any(isinstance(e, AgentWarningEvent) for e in events)


@pytest.mark.asyncio
async def test_improving_test_results_do_not_warn(tmp_path, monkeypatch):
    test_tool = ScriptedTool("pytest", ToolAccess.READ)
    edit = ScriptedTool("edit", ToolAccess.WRITE)
    outputs = [ToolOutput.fail(ToolErrorCode.COMMAND_FAILED, "8 failed"),
               ToolOutput.fail(ToolErrorCode.COMMAND_FAILED, "3 failed"),
               ToolOutput.ok("all passed")]

    async def execute(arguments, context):
        return outputs.pop(0)

    monkeypatch.setattr(test_tool, "execute", execute)
    provider = FakeProvider([*(tool_response(i, name) for i, name in
                              enumerate(["pytest", "edit", "pytest", "edit", "pytest"])),
                             completed(ModelStopReason.END_TURN, TextBlock("done"))])
    agent = build_agent(tmp_path, provider, [test_tool, edit])
    events = await collect(agent, "fix tests")
    assert not outputs
    assert not any(progress_messages(r) for r in provider.requests)
    assert not any(isinstance(e, AgentWarningEvent) for e in events)
