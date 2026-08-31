"""向模型提供团队创建、查询、删除、接管和成员停止工具。"""

from __future__ import annotations

import json
from collections.abc import Mapping

from mycode.models.json_types import JsonValue
from mycode.models.teams import TeamCreateRequest
from mycode.models.tools import ToolAccess, ToolDefinition, ToolErrorCode
from mycode.teams.service import TeamService
from mycode.tools.base import ToolContext, ToolOutput


def _actor(context: ToolContext):
    """取得工具上下文中的可信团队身份。

    Args:
        context: 当前 Agent 的本地工具上下文。

    Returns:
        Lead 或成员 ``TeamActorContext``。

    Raises:
        RuntimeError: 当前会话没有团队身份。
    """

    if context.team_actor is None:
        raise RuntimeError("当前会话没有 Agent Team 身份")
    return context.team_actor


def _snapshot_payload(
    snapshot,
    *,
    deletion_blockers: tuple[str, ...] = (),
) -> dict[str, JsonValue]:
    """把团队快照转换成模型可直接读取的小型 JSON 对象。

    Args:
        snapshot: TeamStateStore 返回的当前一致快照。
        deletion_blockers: TeamDelete 只读预检得到的全部阻塞原因。

    Returns:
        团队、成员、任务和验证摘要，不包含邮箱正文或租约原文。
    """

    return {
        "team": {
            "team_id": snapshot.team.team_id,
            "name": snapshot.team.name,
            "description": snapshot.team.description,
            "lifecycle": snapshot.team.lifecycle.value,
            "lead": {
                "actor_id": "lead",
                "session_id": snapshot.team.lead_session_id,
                "generation": snapshot.team.lead_generation,
            },
            "lead_generation": snapshot.team.lead_generation,
        },
        "members": [
            {
                "agent_id": item.agent_id,
                "name": item.name,
                "role": item.role_name,
                "state": item.state.value,
                "backend": item.backend.value,
                "worktree": str(item.worktree_path),
                "branch": item.branch,
                "current_task_id": item.current_task_id,
            }
            for item in snapshot.members
        ],
        "tasks": [
            {
                "task_id": item.task_id,
                "title": item.title,
                "status": item.status.value,
                "owner_id": item.owner_id,
                "blocked_by": list(item.blocked_by),
                "commit_hashes": list(item.commit_hashes),
            }
            for item in snapshot.tasks
        ],
        "integration": {
            "merged_commits": list(snapshot.integration.merged_commits),
            "current_source_branch": snapshot.integration.current_source_branch,
            "merge_attempt": snapshot.integration.merge_attempt,
            "conflicted_files": [str(item) for item in snapshot.integration.conflicted_files],
            "blocked_by_validation": snapshot.integration.blocked_by_validation,
            "validation_repair_task_id": snapshot.integration.validation_repair_task_id,
        },
        "deletion_blockers": list(deletion_blockers),
    }


class TeamCreateTool:
    """创建当前主会话唯一的存续团队并自动进入 Coordinator。

    Attributes:
        service: 创建团队、保存 TeamBinding 并更新本地 Actor 的服务。
    """

    def __init__(self, service: TeamService) -> None:
        """保存实际创建团队和会话 binding 的服务。

        Args:
            service: 当前工作区唯一的 TeamService。

        Returns:
            不返回数据。
        """

        self.service = service

    @property
    def definition(self) -> ToolDefinition:
        """返回 TeamCreate 的名称、说明和参数格式。

        Returns:
            注册表使用的写工具定义。
        """

        return ToolDefinition(
            "TeamCreate",
            "创建长期 Agent Team；成功后当前主会话自动成为只负责协调的 Lead。",
            {
                "type": "object",
                "properties": {
                    "team_name": {"type": "string", "minLength": 1, "pattern": r"\S"},
                    "description": {"type": "string"},
                },
                "required": ["team_name"],
                "additionalProperties": False,
            },
            ToolAccess.WRITE,
        )

    async def execute(self, arguments: Mapping[str, JsonValue], context: ToolContext) -> ToolOutput:
        """创建团队并返回初始团队快照。

        Args:
            arguments: 包含 team_name 和可选 description 的已校验参数。
            context: 当前主 Agent 上下文；创建前应没有 team_actor。

        Returns:
            成功时返回团队 JSON，失败时返回结构化错误。
        """

        if context.team_actor is not None:
            return ToolOutput.fail(ToolErrorCode.BLOCKED, "当前会话已经管理一个团队")
        try:
            snapshot = self.service.create(
                TeamCreateRequest(
                    str(arguments["team_name"]), str(arguments.get("description", ""))
                )
            )
            return ToolOutput.ok(json.dumps(_snapshot_payload(snapshot), ensure_ascii=False, indent=2))
        except Exception as exc:
            return ToolOutput.fail(ToolErrorCode.INVALID_ARGUMENTS, str(exc))


class TeamGetTool:
    """读取当前团队花名册、任务、后端和合并验证摘要。

    Attributes:
        service: 校验 Actor 并读取一致团队快照的服务。
    """

    def __init__(self, service: TeamService) -> None:
        """保存提供团队查询用例的服务。

        Args:
            service: 当前工作区装配完成的 TeamService。

        Returns:
            不返回数据。
        """

        self.service = service

    @property
    def definition(self) -> ToolDefinition:
        """返回不接收参数的 TeamGet 只读工具定义。

        Returns:
            注册表使用的团队状态查询工具定义。
        """

        return ToolDefinition(
            "TeamGet", "查询当前 Agent Team 的完整状态摘要。",
            {"type": "object", "properties": {}, "additionalProperties": False},
            ToolAccess.READ,
        )

    async def execute(self, arguments: Mapping[str, JsonValue], context: ToolContext) -> ToolOutput:
        """查询调用者所在团队。

        Args:
            arguments: 空对象参数，不读取字段。
            context: 提供可信团队 Actor 的工具上下文。

        Returns:
            当前团队 JSON，身份无效时返回 BLOCKED。
        """

        del arguments
        try:
            actor = _actor(context)
            snapshot = self.service.get(actor)
            deletion = await self.service.integration.deletion_preflight(actor.team_id)
            return ToolOutput.ok(
                json.dumps(
                    _snapshot_payload(
                        snapshot,
                        deletion_blockers=deletion.blockers,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        except Exception as exc:
            return ToolOutput.fail(ToolErrorCode.BLOCKED, str(exc))


class TeamDeleteTool:
    """在全部前置条件满足后删除团队运行资源。

    Attributes:
        service: 执行删除预检、停止成员和全量清理的服务。
    """

    def __init__(self, service: TeamService) -> None:
        """保存提供团队删除用例的服务。

        Args:
            service: 当前工作区装配完成的 TeamService。

        Returns:
            不返回数据。
        """

        self.service = service

    @property
    def definition(self) -> ToolDefinition:
        """返回不接收参数的 TeamDelete 写工具定义。

        Returns:
            注册表使用的团队删除工具定义。
        """

        return ToolDefinition(
            "TeamDelete", "删除当前团队；有活动成员、working 任务、脏目录或未合并提交时只返回阻塞原因。",
            {"type": "object", "properties": {}, "additionalProperties": False}, ToolAccess.WRITE,
        )

    async def execute(self, arguments: Mapping[str, JsonValue], context: ToolContext) -> ToolOutput:
        """执行无副作用预检和通过后的全量清理。

        Args:
            arguments: 空对象参数。
            context: 提供当前 Lead Actor 的上下文。

        Returns:
            allowed、阻塞原因和实际删除资源组成的 JSON。
        """

        del arguments
        try:
            report = await self.service.delete(_actor(context))
            payload = {
                "team_id": report.team_id,
                "allowed": report.allowed,
                "blockers": list(report.blockers),
                "removed_resources": list(report.removed_resources),
            }
            if not report.allowed:
                return ToolOutput.fail(
                    ToolErrorCode.BLOCKED, "团队当前不能删除", content=json.dumps(payload, ensure_ascii=False, indent=2)
                )
            return ToolOutput.ok(json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception as exc:
            return ToolOutput.fail(ToolErrorCode.INTERNAL_ERROR, str(exc))


class TeamTakeoverTool:
    """经终端用户确认后接管一个孤立团队。

    Attributes:
        service: 请求用户确认并轮换 Lead generation 的服务。
    """

    def __init__(self, service: TeamService) -> None:
        """保存提供显式接管用例的服务。

        Args:
            service: 当前工作区装配完成的 TeamService。

        Returns:
            不返回数据。
        """

        self.service = service

    @property
    def definition(self) -> ToolDefinition:
        """返回 TeamTakeover 的 team_id 参数格式。

        Returns:
            注册表使用的团队接管工具定义。
        """

        return ToolDefinition(
            "TeamTakeover", "显式接管一个存续团队，并使旧 Lead generation 失效。",
            {
                "type": "object",
                "properties": {"team_id": {"type": "string", "minLength": 1}},
                "required": ["team_id"], "additionalProperties": False,
            }, ToolAccess.WRITE,
        )

    async def execute(self, arguments: Mapping[str, JsonValue], context: ToolContext) -> ToolOutput:
        """请求终端确认并返回新 Lead 身份摘要。

        Args:
            arguments: 包含要接管的 team_id。
            context: 当前主会话上下文；已有团队身份时拒绝接管。

        Returns:
            新 team ID 和 generation，或用户拒绝/接管失败错误。
        """

        if context.team_actor is not None:
            return ToolOutput.fail(ToolErrorCode.BLOCKED, "当前会话已经管理一个团队")
        try:
            actor = await self.service.takeover(str(arguments["team_id"]))
            return ToolOutput.ok(json.dumps({"team_id": actor.team_id, "lead_generation": actor.generation}, ensure_ascii=False))
        except Exception as exc:
            return ToolOutput.fail(ToolErrorCode.BLOCKED, str(exc))


class TeamMemberStopTool:
    """停止长期团队成员，不影响现有 TaskStop 后台任务语义。

    Attributes:
        service: 控制成员后端并保留任务、会话和 Worktree 的服务。
    """

    def __init__(self, service: TeamService) -> None:
        """保存提供成员停止用例的服务。

        Args:
            service: 当前工作区装配完成的 TeamService。

        Returns:
            不返回数据。
        """

        self.service = service

    @property
    def definition(self) -> ToolDefinition:
        """返回 TeamMemberStop 的成员 ID 和强停参数格式。

        Returns:
            注册表使用的团队成员停止工具定义。
        """

        return ToolDefinition(
            "TeamMemberStop", "停止一个 Agent Team 成员 Host，并保留任务和 Worktree 供 Lead 交接。",
            {
                "type": "object",
                "properties": {
                    "member_id": {"type": "string", "minLength": 1},
                    "force": {"type": "boolean"},
                },
                "required": ["member_id"], "additionalProperties": False,
            }, ToolAccess.WRITE,
        )

    async def execute(self, arguments: Mapping[str, JsonValue], context: ToolContext) -> ToolOutput:
        """停止指定成员并返回终态。

        Args:
            arguments: member_id 和可选 force。
            context: 提供当前 Lead Actor 的上下文。

        Returns:
            成员 ID、名称和 terminated 状态。
        """

        try:
            member = await self.service.member_stop(
                _actor(context), str(arguments["member_id"]), force=bool(arguments.get("force", False))
            )
            return ToolOutput.ok(json.dumps({"agent_id": member.agent_id, "name": member.name, "state": member.state.value}, ensure_ascii=False))
        except Exception as exc:
            return ToolOutput.fail(ToolErrorCode.BLOCKED, str(exc))
