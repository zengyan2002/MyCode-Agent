"""创建团队单元测试共用的真实 Store 记录。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mycode.models.teams import (
    TeamActorContext,
    TeammateBackend,
    TeammateRecord,
    TeammateState,
)
from mycode.teams.store import TeamStateStore


def create_team(
    root: Path,
    *,
    name: str = "refactor",
) -> tuple[TeamStateStore, TeamActorContext]:
    """在临时工作区创建一个空团队和第一代 Lead Actor。

    Args:
        root: pytest 提供的临时工作区绝对路径。
        name: 本用例使用的团队名称。

    Returns:
        已写入磁盘的 TeamStateStore 和对应 Lead Actor。
    """

    store = TeamStateStore(root)
    team = store.create_team(name, "测试团队", "20260815-120000-abcd")
    return store, TeamActorContext(team.team_id, "lead", "lead", 1)


def add_member(
    store: TeamStateStore,
    actor: TeamActorContext,
    *,
    agent_id: str = "agent-alice",
    name: str = "alice",
) -> TeamActorContext:
    """向团队写入一个可认领任务的空闲成员。

    Args:
        store: 已创建团队的 Store。
        actor: 当前有效 Lead Actor。
        agent_id: 成员不可变内部 ID。
        name: SendMessage 使用的团队内名称。

    Returns:
        与新成员 generation 对应的成员 Actor。
    """

    now = datetime.now().astimezone()
    worktree = (store.workspace_root / ".mycode" / "worktrees" / agent_id).resolve()
    store.add_member(
        TeammateRecord(
            agent_id=agent_id,
            team_id=actor.team_id,
            name=name,
            role_name="general-purpose",
            model_override=None,
            session_id="20260815-120001-abcd",
            worktree_name=f"team-{agent_id}",
            worktree_path=worktree,
            branch=f"codex/team-{agent_id}",
            backend=TeammateBackend.IN_PROCESS,
            backend_ref=None,
            state=TeammateState.IDLE,
            runtime_generation=1,
            owner_pid=None,
            lease_token_hash=None,
            plan_mode_required=False,
            current_task_id=None,
            created_at=now,
            updated_at=now,
        )
    )
    return TeamActorContext(actor.team_id, agent_id, "member", 1)
