"""从独立 Agent 的构造入口验证提示和 Turn 隔离。"""

from dataclasses import replace
import json

import pytest

from mycode.agent.progress import PROGRESS_REPLAN, PROGRESS_NOTICE
from mycode.models.agents import IndependentAgentOrigin, BackgroundTaskStatus
from mycode.models.messages import TextBlock, ToolCall, ToolResultMessage
from mycode.models.permissions import PermissionMode
from mycode.models.prompts import RuntimeInstruction, RuntimeInstructionKind
from mycode.models.provider import ModelStopReason
from mycode.models.tools import ToolView
from tests.unit.agent.test_loop import FakeProvider, completed
from tests.unit.agent.test_runner_progress import progress_messages
from tests.unit.agents.test_runtime import _builder, _spec
from mycode.tools.builtin.files import ReadFileTool


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", [IndependentAgentOrigin.DEFINITION, IndependentAgentOrigin.FORK])
async def test_independent_entry_progress_and_next_turn_isolation(tmp_path, origin):
    (tmp_path / "sample.txt").write_text("same contents", encoding="utf-8")
    def read_response(index):
        return completed(ModelStopReason.TOOL_USE,
                         ToolCall(str(index), "read_file", {"path": "sample.txt"}))

    provider = FakeProvider([
        *(read_response(i) for i in range(4)),
        completed(ModelStopReason.END_TURN, TextBlock("first done")),
        read_response(5), completed(ModelStopReason.END_TURN, TextBlock("second done")),
    ])
    builder, hooks, registry = _builder(tmp_path, provider)
    tool = ReadFileTool()
    registry.register(tool)
    ordinary = RuntimeInstruction(RuntimeInstructionKind.RUNTIME_NOTICE, "parent ordinary notice")
    inherited = (ordinary, RuntimeInstruction(RuntimeInstructionKind.RUNTIME_NOTICE, PROGRESS_REPLAN))
    spec = replace(_spec("progress-one", tmp_path), origin=origin, max_model_calls=10,
                   inherited_runtime=inherited, permission_mode=PermissionMode.ALLOW,
                   tool_view=ToolView(final_allowlist=frozenset({"read_file"})))
    try:
        first_runner = builder.build(spec)
        first = await first_runner.start().wait()
        second_runner = builder.build(replace(spec, run_id="progress-two"))
        second = await second_runner.start().wait()
    finally:
        await hooks.close()
    assert first.status is BackgroundTaskStatus.COMPLETED
    assert second.status is BackgroundTaskStatus.COMPLETED
    assert first.final_text == "first done" and second.final_text == "second done"
    assert [progress_messages(r) for r in provider.requests] == [
        [], [], [], [PROGRESS_NOTICE], [PROGRESS_REPLAN], [], [],
    ]
    assert first.usage.tool_calls == 4 and second.usage.tool_calls == 1
    outputs = [json.loads(m.content) for runner in (first_runner, second_runner)
               for m in runner.history if isinstance(m, ToolResultMessage)]
    assert len(outputs) == 5 and all(output["success"] for output in outputs)
    if origin is IndependentAgentOrigin.FORK:
        assert all(ordinary in request.prompt.runtime for request in provider.requests)
