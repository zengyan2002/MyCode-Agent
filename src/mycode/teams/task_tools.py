"""向 Lead 和成员提供独立于后台 TaskManager 的共享任务工具。"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping

from mycode.models.json_types import JsonValue
from mycode.models.teams import (
    TeamActorContext,
    TeamTaskCreateRequest,
    TeamTaskPriority,
    TeamTaskQuery,
    TeamTaskStatus,
    TeamTaskUpdateRequest,
    TeamTaskView,
)
from mycode.models.tools import ToolAccess, ToolDefinition, ToolErrorCode
from mycode.teams.service import TeamService
from mycode.teams.tasks import TeamTaskBoard
from mycode.tools.base import ToolContext, ToolOutput


TaskCompletionValidator = Callable[
    [TeamActorContext, TeamTaskUpdateRequest],
    Awaitable[None],
]


def _actor(context: ToolContext) -> TeamActorContext:
    """返回 ToolContext 中不能由模型伪造的团队 Actor。

    Args:
        context: 当前 Agent 的本地工具上下文。

    Returns:
        运行时写入的 Lead 或成员 Actor。

    Raises:
        RuntimeError: 当前会话没有团队身份。
    """

    if context.team_actor is None:
        raise RuntimeError("当前会话没有 Agent Team 身份")
    return context.team_actor


def _task_payload(view: TeamTaskView) -> dict[str, JsonValue]:
    """把任务视图转换成含依赖派生字段的 JSON 对象。

    Args:
        view: TaskBoard 在读取时计算出的任务和依赖状态。

    Returns:
        可直接序列化给模型的任务字段。
    """

    task = view.task
    return {
        "task_id": task.task_id, "title": task.title, "description": task.description,
        "task_kind": task.task_kind, "priority": task.priority.value,
        "status": task.status.value, "owner_id": task.owner_id,
        "blocked_by": list(task.blocked_by), "blocked": view.blocked,
        "blocks": list(view.blocks), "assigned": view.assigned, "claimable": view.claimable,
        "progress": task.progress, "result": task.result,
        "commit_hashes": list(task.commit_hashes), "attempts": len(task.attempts),
    }


class TeamTaskCreateTool:
    """让 Lead 创建带优先级和依赖的共享任务。

    Attributes:
        service: 创建任务并唤醒空闲成员的团队用例服务。
    """

    def __init__(self, service: TeamService) -> None:
        """保存实际创建任务的 TeamService。

        Args:
            service: 当前工作区装配完成的团队服务。

        Returns:
            不返回数据。
        """

        self.service = service

    @property
    def definition(self) -> ToolDefinition:
        """返回 TeamTaskCreate 的字段格式和写访问分类。

        Returns:
            注册表使用的团队任务创建工具定义。
        """

        return ToolDefinition(
            "TeamTaskCreate", "创建团队共享任务，并唤醒空闲成员自主认领。",
            {"type": "object", "properties": {
                "title": {"type": "string", "minLength": 1},
                "description": {"type": "string", "minLength": 1},
                "task_kind": {"type": "string", "enum": ["code", "research"]},
                "priority": {"type": "string", "enum": ["high", "normal", "low"]},
                "blocked_by": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
            }, "required": ["title", "description"], "additionalProperties": False}, ToolAccess.WRITE,
        )

    async def execute(self, arguments: Mapping[str, JsonValue], context: ToolContext) -> ToolOutput:
        """创建任务并返回新任务 ID 和状态。

        Args:
            arguments: 标题、说明、类型、优先级和依赖 ID。
            context: 提供当前 Lead Actor 的工具上下文。

        Returns:
            新任务 JSON；依赖或身份无效时返回 INVALID_ARGUMENTS。
        """
        try:
            task = await self.service.create_task(_actor(context), TeamTaskCreateRequest(
                title=str(arguments["title"]), description=str(arguments["description"]),
                task_kind=str(arguments.get("task_kind", "code")),  # type: ignore[arg-type]
                priority=TeamTaskPriority(str(arguments.get("priority", "normal"))),
                blocked_by=tuple(str(item) for item in arguments.get("blocked_by", [])),  # type: ignore[union-attr]
            ))
            return ToolOutput.ok(json.dumps({"task_id": task.task_id, "status": task.status.value}, ensure_ascii=False))
        except Exception as exc:
            return ToolOutput.fail(ToolErrorCode.INVALID_ARGUMENTS, str(exc))


class TeamTaskListTool:
    """按身份列出可领取任务和调用者自己的任务。

    Attributes:
        board: 读取共享任务和依赖派生状态的任务板。
    """

    def __init__(self, board: TeamTaskBoard) -> None:
        """保存负责过滤和排序任务的共享任务板。

        Args:
            board: 与团队 Store 共用 tasks.json 的任务板。

        Returns:
            不返回数据。
        """

        self.board = board

    @property
    def definition(self) -> ToolDefinition:
        """返回 TeamTaskList 的过滤参数和只读访问分类。

        Returns:
            注册表使用的团队任务列表工具定义。
        """

        return ToolDefinition(
            "TeamTaskList", "列出团队任务；成员默认只看到可领取任务和自己的任务。",
            {"type": "object", "properties": {
                "status": {"type": "string", "enum": ["todo", "working", "completed", "failed", "cancelled"]},
                "owner_id": {"type": "string"}, "claimable_only": {"type": "boolean"},
            }, "additionalProperties": False}, ToolAccess.READ,
        )

    async def execute(self, arguments: Mapping[str, JsonValue], context: ToolContext) -> ToolOutput:
        """查询并稳定排序共享任务。

        Args:
            arguments: 可选状态、负责人和只看可领取过滤条件。
            context: 提供当前 Lead 或成员 Actor 的上下文。

        Returns:
            任务视图 JSON 数组。
        """
        try:
            query = TeamTaskQuery(
                status=TeamTaskStatus(str(arguments["status"])) if "status" in arguments else None,
                owner_id=str(arguments["owner_id"]) if "owner_id" in arguments else None,
                claimable_only=bool(arguments.get("claimable_only", False)),
            )
            views = self.board.list(_actor(context), query)
            return ToolOutput.ok(json.dumps([_task_payload(item) for item in views], ensure_ascii=False, indent=2))
        except Exception as exc:
            return ToolOutput.fail(ToolErrorCode.INVALID_ARGUMENTS, str(exc))


class TeamTaskGetTool:
    """读取当前团队中一个可见任务的完整状态。

    Attributes:
        board: 按调用者身份查询任务的共享任务板。
    """

    def __init__(self, board: TeamTaskBoard) -> None:
        """保存负责单任务查询的共享任务板。

        Args:
            board: 与团队 Store 共用 tasks.json 的任务板。

        Returns:
            不返回数据。
        """

        self.board = board

    @property
    def definition(self) -> ToolDefinition:
        """返回 TeamTaskGet 的 task_id 参数格式。

        Returns:
            注册表使用的单项团队任务查询工具定义。
        """

        return ToolDefinition("TeamTaskGet", "按 ID 查询一个团队任务。", {"type": "object", "properties": {"task_id": {"type": "string", "minLength": 1}}, "required": ["task_id"], "additionalProperties": False}, ToolAccess.READ)
    async def execute(self, arguments: Mapping[str, JsonValue], context: ToolContext) -> ToolOutput:
        """返回单个任务视图；不可见与不存在使用同一错误。

        Args:
            arguments: 包含要查询的 task_id。
            context: 提供当前 Lead 或成员 Actor 的上下文。

        Returns:
            可见任务的完整 JSON；不可见或不存在时返回 NOT_FOUND。
        """

        try:
            return ToolOutput.ok(json.dumps(_task_payload(self.board.get(_actor(context), str(arguments["task_id"]))), ensure_ascii=False, indent=2))
        except Exception as exc:
            return ToolOutput.fail(ToolErrorCode.NOT_FOUND, str(exc))


class TeamTaskClaimTool:
    """让成员原子认领一个未分配或已指派给自己的任务。

    Attributes:
        board: 在文件锁内决定唯一认领者的共享任务板。
    """

    def __init__(self, board: TeamTaskBoard) -> None:
        """保存执行原子认领的任务板。

        Args:
            board: 与团队 Store 共用 tasks.json 的任务板。

        Returns:
            不返回数据。
        """

        self.board = board

    @property
    def definition(self) -> ToolDefinition:
        """返回 TeamTaskClaim 的任务和可选检查轮次参数。

        Returns:
            注册表使用的团队任务认领工具定义。
        """

        return ToolDefinition("TeamTaskClaim", "原子认领一个可领取任务；每个成员同时只能有一个 working 任务。", {"type": "object", "properties": {"task_id": {"type": "string", "minLength": 1}, "round_id": {"type": "string"}}, "required": ["task_id"], "additionalProperties": False}, ToolAccess.WRITE)
    async def execute(self, arguments: Mapping[str, JsonValue], context: ToolContext) -> ToolOutput:
        """执行锁内认领并返回唯一成功的负责人。

        Args:
            arguments: task_id 和可选自主认领 round_id。
            context: 提供当前成员 Actor 的工具上下文。

        Returns:
            认领后的任务 ID、working 状态和 owner；失败时返回 BLOCKED。
        """

        try:
            task = self.board.claim(_actor(context), str(arguments["task_id"]), str(arguments["round_id"]) if "round_id" in arguments else None)
            return ToolOutput.ok(json.dumps({"task_id": task.task_id, "status": task.status.value, "owner_id": task.owner_id}, ensure_ascii=False))
        except Exception as exc:
            return ToolOutput.fail(ToolErrorCode.BLOCKED, str(exc))


class TeamTaskUpdateTool:
    """按显式字段更新任务，不接受任意 patch。

    Attributes:
        board: 执行字段权限和状态机的共享任务板。
        service: 主进程中负责完成前 Git 验证的 TeamService。
        completion_validator: 外部成员 Host 使用的等价完成检查函数。
    """

    def __init__(
        self,
        board: TeamTaskBoard,
        service: TeamService | None = None,
        completion_validator: TaskCompletionValidator | None = None,
    ) -> None:
        """保存任务板，并可接入生产环境的提交完成门禁。

        Args:
            board: 负责普通状态转换和持久化的共享任务板。
            service: 生产环境 TeamService；提供时，``completed`` 更新会先
                检查负责人 Worktree 和提交归属。独立任务板测试可不传。
            completion_validator: 独立 Host 使用的完成前检查函数；主进程传入
                service 时不需要该参数。

        Returns:
            不返回数据。
        """

        self.board = board
        self.service = service
        self.completion_validator = completion_validator

    @property
    def definition(self) -> ToolDefinition:
        """返回 TeamTaskUpdate 允许修改的显式字段集合。

        Returns:
            注册表使用的团队任务更新工具定义。
        """

        return ToolDefinition("TeamTaskUpdate", "更新团队任务状态、负责人、进展、结果、提交或依赖。", {"type": "object", "properties": {
            "task_id": {"type": "string", "minLength": 1}, "status": {"type": "string", "enum": ["todo", "working", "completed", "failed", "cancelled"]},
            "owner": {"type": "string"}, "priority": {"type": "string", "enum": ["high", "normal", "low"]},
            "progress": {"type": "string"}, "result": {"type": "string"},
            "commit_hashes": {"type": "array", "items": {"type": "string"}},
            "add_blocked_by": {"type": "array", "items": {"type": "string"}}, "remove_blocked_by": {"type": "array", "items": {"type": "string"}},
            "failure_reason": {"type": "string"},
        }, "required": ["task_id"], "additionalProperties": False}, ToolAccess.WRITE)
    async def execute(self, arguments: Mapping[str, JsonValue], context: ToolContext) -> ToolOutput:
        """把已校验字段转成 TeamTaskUpdateRequest 并执行状态机。

        Args:
            arguments: 只包含 schema 列出的显式更新字段。
            context: 提供当前 Lead 或成员 Actor 的上下文。

        Returns:
            更新后的任务状态与 owner；越权或非法转换返回 BLOCKED。
        """
        try:
            request = TeamTaskUpdateRequest(
                task_id=str(arguments["task_id"]),
                status=TeamTaskStatus(str(arguments["status"])) if "status" in arguments else None,
                owner=str(arguments["owner"]) if "owner" in arguments else None,
                priority=TeamTaskPriority(str(arguments["priority"])) if "priority" in arguments else None,
                progress=str(arguments["progress"]) if "progress" in arguments else None,
                result=str(arguments["result"]) if "result" in arguments else None,
                commit_hashes=tuple(str(item) for item in arguments["commit_hashes"]) if "commit_hashes" in arguments else None,  # type: ignore[union-attr]
                add_blocked_by=tuple(str(item) for item in arguments.get("add_blocked_by", [])),  # type: ignore[union-attr]
                remove_blocked_by=tuple(str(item) for item in arguments.get("remove_blocked_by", [])),  # type: ignore[union-attr]
                failure_reason=str(arguments["failure_reason"]) if "failure_reason" in arguments else None,
            )
            actor = _actor(context)
            if self.service is not None:
                task = await self.service.update_task(actor, request)
            else:
                if (
                    request.status is TeamTaskStatus.COMPLETED
                    and self.completion_validator is not None
                ):
                    await self.completion_validator(actor, request)
                task = self.board.update(actor, request)
            return ToolOutput.ok(json.dumps({"task_id": task.task_id, "status": task.status.value, "owner_id": task.owner_id}, ensure_ascii=False))
        except Exception as exc:
            return ToolOutput.fail(ToolErrorCode.BLOCKED, str(exc))
