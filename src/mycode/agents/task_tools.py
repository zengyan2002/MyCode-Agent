"""提供模型查询和停止当前主会话后台子 Agent 的 SYSTEM 工具。"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

from mycode.agents.tasks import TaskManager
from mycode.models.agents import BackgroundTaskSnapshot
from mycode.models.json_types import JsonValue
from mycode.models.tools import ToolAccess, ToolDefinition, ToolErrorCode
from mycode.tools.base import ToolContext, ToolOutput


def _task_payload(snapshot: BackgroundTaskSnapshot) -> dict[str, JsonValue]:
    """把一个后台任务快照转换成模型可读取的 JSON 字段。

    Args:
        snapshot: TaskManager 在查询时刻复制出的不可变任务快照。

    Returns:
        包含任务身份、状态、时间、结果和用量的普通字典。尚未开始或尚未
        结束的时间字段为 ``None``，尚无结果时 result 和 usage 也为
        ``None``。
    """

    result = snapshot.result
    workspace = result.workspace_report if result is not None else None
    return {
        "task_id": snapshot.task_id,
        "name": snapshot.name,
        "description": snapshot.description,
        "source": snapshot.source,
        "status": snapshot.status.value,
        "created_at": snapshot.created_at.isoformat(),
        "started_at": (
            snapshot.started_at.isoformat()
            if snapshot.started_at is not None
            else None
        ),
        "ended_at": (
            snapshot.ended_at.isoformat()
            if snapshot.ended_at is not None
            else None
        ),
        "result": (
            None
            if result is None
            else {
                "final_text": result.final_text,
                "partial_text": result.partial_text,
                "error": result.error,
            }
        ),
        "usage": (
            None
            if result is None
            else {
                "model_calls": result.usage.model_calls,
                "input_tokens": result.usage.input_tokens,
                "cached_input_tokens": result.usage.cached_input_tokens,
                "output_tokens": result.usage.output_tokens,
                "tool_calls": result.usage.tool_calls,
                "duration_ms": result.usage.duration_ms,
            }
        ),
        "workspace": (
            None
            if workspace is None
            else {
                "path": str(workspace.workspace.root),
                "branch": workspace.workspace.branch,
                "worktree_name": workspace.workspace.worktree_name,
                "action": workspace.action.value,
                "reason": workspace.reason,
                "warnings": list(workspace.warnings),
            }
        ),
    }


def _json_output(value: JsonValue) -> ToolOutput:
    """把任务字段编码成稳定、可读的 UTF-8 JSON 工具结果。

    Args:
        value: 由任务快照转换出的 JSON 兼容值。

    Returns:
        正文使用两个空格缩进且保留中文的成功 ToolOutput。
    """

    return ToolOutput.ok(json.dumps(value, ensure_ascii=False, indent=2))


class _TaskToolBase:
    """保存任务工具共同使用的任务管理器和动态会话读取函数。

    Attributes:
        _manager: 当前进程唯一的后台任务管理器。
        _session_id_getter: 每次执行工具时返回当前主会话 ID 的函数；使用
            函数而不是启动时的字符串，保证切换会话后查询新的会话范围。
    """

    def __init__(
        self,
        manager: TaskManager,
        session_id_getter: Callable[[], str],
    ) -> None:
        """保存任务工具执行时需要的两个真实应用对象。

        Args:
            manager: 负责后台任务状态、结果和取消的 TaskManager。
            session_id_getter: 无参数调用后返回当前主会话 ID 的函数。

        Returns:
            不返回数据；新工具实例后续会按当前会话隔离任务。
        """

        self._manager = manager
        self._session_id_getter = session_id_getter


class TaskListTool(_TaskToolBase):
    """列出当前主会话创建的全部后台子 Agent 任务。"""

    @property
    def definition(self) -> ToolDefinition:
        """返回 TaskList 的无参数调用格式。

        Returns:
            名称为 ``TaskList`` 的只读 SYSTEM 工具定义。
        """

        return ToolDefinition(
            name="TaskList",
            description="列出当前主会话的后台子 Agent 任务及其状态。",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            access=ToolAccess.READ,
        )

    async def execute(
        self,
        arguments: Mapping[str, JsonValue],
        context: ToolContext,
    ) -> ToolOutput:
        """读取当前会话任务并返回 JSON 数组。

        Args:
            arguments: 已通过空对象 Schema 校验的参数，函数不会读取它。
            context: 当前工具上下文；任务归属由动态会话函数决定，因此这里
                不读取上下文字段。

        Returns:
            每项都包含任务状态、时间、结果和用量的 JSON 数组；没有任务时
            返回空数组。
        """

        del arguments, context
        tasks = self._manager.list(self._session_id_getter())
        completed = [task.result for task in tasks if task.result is not None]
        cached_values = [
            result.usage.cached_input_tokens
            for result in completed
            if result is not None
            and result.usage.cached_input_tokens is not None
        ]
        return _json_output(
            {
                "tasks": [_task_payload(task) for task in tasks],
                "summary": {
                    "total": len(tasks),
                    "model_calls": sum(
                        result.usage.model_calls
                        for result in completed
                        if result is not None
                    ),
                    "input_tokens": sum(
                        result.usage.input_tokens
                        for result in completed
                        if result is not None
                    ),
                    "cached_input_tokens": (
                        sum(cached_values) if cached_values else None
                    ),
                    "output_tokens": sum(
                        result.usage.output_tokens
                        for result in completed
                        if result is not None
                    ),
                    "tool_calls": sum(
                        result.usage.tool_calls
                        for result in completed
                        if result is not None
                    ),
                    "duration_ms": sum(
                        result.usage.duration_ms
                        for result in completed
                        if result is not None
                    ),
                },
            }
        )


class TaskGetTool(_TaskToolBase):
    """读取当前主会话中一个后台子 Agent 的完整状态和结果。"""

    @property
    def definition(self) -> ToolDefinition:
        """返回要求 ``task_id`` 的 TaskGet 调用格式。

        Returns:
            名称为 ``TaskGet`` 的只读 SYSTEM 工具定义。
        """

        return ToolDefinition(
            name="TaskGet",
            description="立即查询当前主会话中一个后台子 Agent 的完整状态和结果。",
            input_schema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "minLength": 1},
                },
                "required": ["task_id"],
                "additionalProperties": False,
            },
            access=ToolAccess.READ,
        )

    async def execute(
        self,
        arguments: Mapping[str, JsonValue],
        context: ToolContext,
    ) -> ToolOutput:
        """查询一个任务并阻止跨会话读取。

        Args:
            arguments: 包含 Agent 工具返回的 ``task_id`` 字符串。
            context: 当前工具上下文；这里只保留统一工具协议，不读取字段。

        Returns:
            找到时返回任务 JSON；任务不存在或属于其他会话时返回
            ``not_found`` 工具错误。
        """

        del context
        task_id = str(arguments["task_id"])
        try:
            task = self._manager.get(self._session_id_getter(), task_id)
        except KeyError as exc:
            return ToolOutput.fail(ToolErrorCode.NOT_FOUND, str(exc))
        return _json_output(_task_payload(task))


class TaskStopTool(_TaskToolBase):
    """请求取消当前主会话中排队或运行的后台子 Agent。"""

    @property
    def definition(self) -> ToolDefinition:
        """返回要求 ``task_id`` 的 TaskStop 调用格式。

        Returns:
            名称为 ``TaskStop`` 的写入类 SYSTEM 工具定义。
        """

        return ToolDefinition(
            name="TaskStop",
            description="取消当前主会话中排队或运行的后台子 Agent 任务。",
            input_schema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "minLength": 1},
                },
                "required": ["task_id"],
                "additionalProperties": False,
            },
            access=ToolAccess.WRITE,
        )

    async def execute(
        self,
        arguments: Mapping[str, JsonValue],
        context: ToolContext,
    ) -> ToolOutput:
        """发送取消信号并返回请求后的任务快照。

        Args:
            arguments: 包含需要停止的 ``task_id`` 字符串。
            context: 当前工具上下文；这里只保留统一工具协议，不读取字段。

        Returns:
            找到时返回取消请求后的任务 JSON；任务不存在或属于其他会话时
            返回 ``not_found`` 工具错误。
        """

        del context
        task_id = str(arguments["task_id"])
        try:
            task = await self._manager.stop(
                self._session_id_getter(),
                task_id,
            )
        except KeyError as exc:
            return ToolOutput.fail(ToolErrorCode.NOT_FOUND, str(exc))
        return _json_output(_task_payload(task))
