"""团队 index、成员和一致快照的磁盘持久化。"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

from mycode.models.teams import (
    BackendPreference,
    ClaimScanRound,
    TeamActorContext,
    TeamEvent,
    TeamIntegrationState,
    TeamLifecycle,
    TeamRecord,
    TeamSnapshot,
    TeamTaskAttempt,
    TeamTaskPriority,
    TeamTaskRecord,
    TeamTaskStatus,
    TeammateBackend,
    TeammateRecord,
    TeammateState,
)
from mycode.teams.locks import ExclusiveFileLock


class TeamStoreError(RuntimeError):
    """表示团队记录不存在、Actor 已失效或磁盘更新无法完成。"""


_T = TypeVar("_T")


def _now() -> datetime:
    """返回团队快照使用的带时区当前时间。

    Returns:
        当前本地时区的 datetime。
    """

    return datetime.now().astimezone()


def _atomic_json(path: Path, value: object) -> None:
    """把一个 JSON 快照完整同步到临时文件后原子替换目标。

    Args:
        path: 目标 JSON 文件绝对路径。
        value: 可以由 json.dumps 编码的对象。

    Returns:
        替换成功时不返回数据。

    Raises:
        TeamStoreError: 创建、写入或替换文件失败。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, raw = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(raw)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except (OSError, TypeError, ValueError) as exc:
        raise TeamStoreError(f"无法保存 {path.name}：{exc}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _read_json(path: Path, default: _T | None = None) -> Any | _T:
    """读取一个完整 JSON 文件，并在明确提供默认值时兼容文件不存在。

    Args:
        path: 要读取的 JSON 文件。
        default: 文件不存在时返回的值；None 表示不存在也是错误。

    Returns:
        解码后的 JSON 数据或传入的默认值。
    """

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        if default is not None:
            return default
        raise TeamStoreError(f"记录不存在：{path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TeamStoreError(f"无法读取 {path.name}：{exc}") from exc


def _team_to_json(team: TeamRecord) -> dict[str, Any]:
    """把团队记录转换成 team.json 可保存的字段。

    Args:
        team: 当前团队身份和生命周期记录。

    Returns:
        只包含 JSON 标量和数组的字典。
    """

    return {
        "team_id": team.team_id,
        "name": team.name,
        "description": team.description,
        "lead_session_id": team.lead_session_id,
        "lead_generation": team.lead_generation,
        "lifecycle": team.lifecycle.value,
        "member_ids": list(team.member_ids),
        "created_at": team.created_at.isoformat(),
        "updated_at": team.updated_at.isoformat(),
        "revision": team.revision,
    }


def _team_from_json(raw: dict[str, Any]) -> TeamRecord:
    """把 team.json 对象还原为经过模型校验的团队记录。

    Args:
        raw: JSON 解码后的团队字段。

    Returns:
        可供 generation 和 revision 校验使用的 TeamRecord。
    """

    return TeamRecord(
        team_id=str(raw["team_id"]),
        name=str(raw["name"]),
        description=str(raw.get("description", "")),
        lead_session_id=str(raw["lead_session_id"]),
        lead_generation=int(raw["lead_generation"]),
        lifecycle=TeamLifecycle(str(raw["lifecycle"])),
        member_ids=tuple(str(item) for item in raw.get("member_ids", [])),
        created_at=datetime.fromisoformat(str(raw["created_at"])),
        updated_at=datetime.fromisoformat(str(raw["updated_at"])),
        revision=int(raw["revision"]),
    )


def _member_to_json(member: TeammateRecord) -> dict[str, Any]:
    """把成员花名册记录转换成独立 JSON 文件字段。

    Args:
        member: 需要持久化的成员身份和运行状态。

    Returns:
        路径、枚举和时间均已转成字符串的字典。
    """

    return {
        "agent_id": member.agent_id,
        "team_id": member.team_id,
        "name": member.name,
        "role_name": member.role_name,
        "model_override": member.model_override,
        "session_id": member.session_id,
        "worktree_name": member.worktree_name,
        "worktree_path": str(member.worktree_path),
        "branch": member.branch,
        "backend": member.backend.value,
        "backend_ref": member.backend_ref,
        "state": member.state.value,
        "runtime_generation": member.runtime_generation,
        "owner_pid": member.owner_pid,
        "lease_token_hash": member.lease_token_hash,
        "plan_mode_required": member.plan_mode_required,
        "current_task_id": member.current_task_id,
        "created_at": member.created_at.isoformat(),
        "updated_at": member.updated_at.isoformat(),
    }


def _member_from_json(raw: dict[str, Any]) -> TeammateRecord:
    """把成员 JSON 还原成可供 Supervisor 使用的记录。

    Args:
        raw: JSON 解码后的成员字段。

    Returns:
        已恢复 Path、枚举和带时区时间的 TeammateRecord。
    """

    return TeammateRecord(
        agent_id=str(raw["agent_id"]),
        team_id=str(raw["team_id"]),
        name=str(raw["name"]),
        role_name=str(raw["role_name"]),
        model_override=raw.get("model_override"),
        session_id=str(raw["session_id"]),
        worktree_name=str(raw["worktree_name"]),
        worktree_path=Path(str(raw["worktree_path"])),
        branch=str(raw["branch"]),
        backend=TeammateBackend(str(raw["backend"])),
        backend_ref=raw.get("backend_ref"),
        state=TeammateState(str(raw["state"])),
        runtime_generation=int(raw["runtime_generation"]),
        owner_pid=raw.get("owner_pid"),
        lease_token_hash=raw.get("lease_token_hash"),
        plan_mode_required=bool(raw.get("plan_mode_required", False)),
        current_task_id=raw.get("current_task_id"),
        created_at=datetime.fromisoformat(str(raw["created_at"])),
        updated_at=datetime.fromisoformat(str(raw["updated_at"])),
    )


def task_to_json(task: TeamTaskRecord) -> dict[str, Any]:
    """把任务记录转换为 tasks.json 可以保存的对象。

    Args:
        task: 需要持久化的团队任务。

    Returns:
        只包含 JSON 标量、数组和对象的字典。
    """

    return {
        "task_id": task.task_id,
        "team_id": task.team_id,
        "title": task.title,
        "description": task.description,
        "task_kind": task.task_kind,
        "priority": task.priority.value,
        "status": task.status.value,
        "owner_id": task.owner_id,
        "blocked_by": list(task.blocked_by),
        "progress": task.progress,
        "result": task.result,
        "commit_hashes": list(task.commit_hashes),
        "attempts": [
            {
                "number": item.number,
                "owner_id": item.owner_id,
                "started_at": item.started_at.isoformat(),
                "paused_at": item.paused_at.isoformat() if item.paused_at else None,
                "ended_at": item.ended_at.isoformat() if item.ended_at else None,
                "failure_reason": item.failure_reason,
            }
            for item in task.attempts
        ],
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }


def task_from_json(raw: dict[str, Any]) -> TeamTaskRecord:
    """把 tasks.json 中的一项还原成团队任务。

    Args:
        raw: JSON 解码后的任务对象。

    Returns:
        完成字段类型转换和模型校验的 TeamTaskRecord。
    """

    return TeamTaskRecord(
        task_id=str(raw["task_id"]),
        team_id=str(raw["team_id"]),
        title=str(raw["title"]),
        description=str(raw["description"]),
        task_kind=str(raw.get("task_kind", "code")),  # type: ignore[arg-type]
        priority=TeamTaskPriority(str(raw["priority"])),
        status=TeamTaskStatus(str(raw["status"])),
        owner_id=raw.get("owner_id"),
        blocked_by=tuple(str(item) for item in raw.get("blocked_by", [])),
        progress=raw.get("progress"),
        result=raw.get("result"),
        commit_hashes=tuple(str(item) for item in raw.get("commit_hashes", [])),
        attempts=tuple(
            TeamTaskAttempt(
                number=int(item["number"]),
                owner_id=str(item["owner_id"]),
                started_at=datetime.fromisoformat(str(item["started_at"])),
                paused_at=(datetime.fromisoformat(str(item["paused_at"])) if item.get("paused_at") else None),
                ended_at=(datetime.fromisoformat(str(item["ended_at"])) if item.get("ended_at") else None),
                failure_reason=item.get("failure_reason"),
            )
            for item in raw.get("attempts", [])
        ),
        created_at=datetime.fromisoformat(str(raw["created_at"])),
        updated_at=datetime.fromisoformat(str(raw["updated_at"])),
        completed_at=(datetime.fromisoformat(str(raw["completed_at"])) if raw.get("completed_at") else None),
    )


def _integration_to_json(state: TeamIntegrationState) -> dict[str, Any]:
    """把合并、冲突和验证状态转换成 integration.json 字段。

    Args:
        state: 当前团队的 TeamIntegrationState。

    Returns:
        可由原子 JSON 写入器保存的字典。
    """

    return {
        "team_id": state.team_id,
        "merged_commits": list(state.merged_commits),
        "current_source_branch": state.current_source_branch,
        "merge_attempt": state.merge_attempt,
        "conflicted_files": [str(path) for path in state.conflicted_files],
        "blocked_by_validation": state.blocked_by_validation,
        "validation_repair_task_id": state.validation_repair_task_id,
        "validation_reports": [
            {
                "command": report.command,
                "scope": report.scope,
                "exit_code": report.exit_code,
                "head": report.head,
                "ran_at": report.ran_at.isoformat(),
            }
            for report in state.validation_reports
        ],
        "updated_at": state.updated_at.isoformat(),
    }


class TeamStateStore:
    """持久化团队身份、成员和一致快照，不启动 Agent 或执行 Git。

    Attributes:
        workspace_root: MyCode 启动时冻结的主工作区绝对路径。
        root: 当前工作区的 ``.mycode/teams`` 目录。
    """

    def __init__(self, workspace_root: Path) -> None:
        """绑定主工作区并准备稳定的团队路径。

        Args:
            workspace_root: MyCode 当前主工作区绝对路径。

        Returns:
            不返回数据；目录在首次创建团队时生成。
        """

        self.workspace_root = workspace_root.resolve(strict=True)
        self.root = self.workspace_root / ".mycode" / "teams"
        self._index_path = self.root / "index.json"
        self._index_lock = self.root / "locks" / "index.lock"

    def team_dir(self, team_id: str) -> Path:
        """返回指定团队保存全部运行数据的目录。

        Args:
            team_id: 系统生成且不含路径分隔符的团队 ID。

        Returns:
            位于团队存储根目录下的绝对或相对目录路径。

        Raises:
            TeamStoreError: 团队 ID 为空或包含路径分隔符。
        """

        if not team_id or any(char in team_id for char in "/\\"):
            raise TeamStoreError("团队 ID 格式无效")
        return self.root / team_id

    def create_team(self, name: str, description: str, lead_session_id: str) -> TeamRecord:
        """创建一个团队，并原子登记名称与 Lead session。

        Args:
            name: 工作区内存续团队唯一的可读名称。
            description: 团队本次负责的用户目标。
            lead_session_id: 创建团队的主会话 ID。

        Returns:
            已写入 team.json 的初始 TeamRecord。
        """

        clean_name = name.strip()
        if not clean_name:
            raise TeamStoreError("团队名称不能为空")
        with ExclusiveFileLock(self._index_lock, f"lead:{lead_session_id}"):
            index = _read_json(self._index_path, {"revision": 0, "teams": {}})
            teams = dict(index.get("teams", {}))
            for item in teams.values():
                if item["name"] == clean_name:
                    raise TeamStoreError(f"存续团队名称已存在：{clean_name}")
                if item["lead_session_id"] == lead_session_id:
                    raise TeamStoreError("当前 Lead 会话已经拥有一个存续团队")
            team_id = f"team-{secrets.token_hex(6)}"
            now = _now()
            team = TeamRecord(
                team_id=team_id,
                name=clean_name,
                description=description.strip(),
                lead_session_id=lead_session_id,
                lead_generation=1,
                lifecycle=TeamLifecycle.ACTIVE,
                member_ids=(),
                created_at=now,
                updated_at=now,
                revision=0,
            )
            directory = self.team_dir(team_id)
            try:
                (directory / "members").mkdir(parents=True)
                for child in ("mailboxes", "cursors", "sessions", "runtime", "locks"):
                    (directory / child).mkdir()
                _atomic_json(directory / "team.json", _team_to_json(team))
                _atomic_json(directory / "tasks.json", {"revision": 0, "tasks": [], "scans": []})
                _atomic_json(directory / "integration.json", _integration_to_json(TeamIntegrationState(team_id)))
                (directory / "events.jsonl").touch()
                (directory / "mailboxes" / "lead.jsonl").touch()
                _atomic_json(directory / "cursors" / "lead.json", {"byte_offset": 0, "last_message_id": None})
                teams[team_id] = {"name": clean_name, "lead_session_id": lead_session_id}
                _atomic_json(self._index_path, {"revision": int(index.get("revision", 0)) + 1, "teams": teams})
            except Exception:
                shutil.rmtree(directory, ignore_errors=True)
                raise
            return team

    def load_team(self, team_id: str) -> TeamSnapshot:
        """从磁盘读取一个团队的当前一致视图。

        Args:
            team_id: 需要读取的不可变团队 ID。

        Returns:
            团队、全部成员、任务、扫描轮次和集成状态组成的快照。
        """

        directory = self.team_dir(team_id)
        team = _team_from_json(_read_json(directory / "team.json"))
        members = tuple(
            _member_from_json(_read_json(directory / "members" / f"{member_id}.json"))
            for member_id in team.member_ids
        )
        task_raw = _read_json(directory / "tasks.json", {"tasks": [], "scans": []})
        tasks = tuple(task_from_json(item) for item in task_raw.get("tasks", []))
        scans = tuple(
            ClaimScanRound(
                round_id=str(item["round_id"]),
                team_id=str(item["team_id"]),
                task_ids=tuple(str(value) for value in item.get("task_ids", [])),
                expected_member_ids=tuple(
                    str(value) for value in item.get("expected_member_ids", [])
                ),
                finished_member_ids=tuple(
                    str(value) for value in item.get("finished_member_ids", [])
                ),
                claimed_task_ids=tuple(
                    str(value) for value in item.get("claimed_task_ids", [])
                ),
                created_at=datetime.fromisoformat(str(item["created_at"])),
            )
            for item in task_raw.get("scans", [])
        )
        integration_raw = _read_json(directory / "integration.json")
        from mycode.models.teams import ValidationReport
        integration = TeamIntegrationState(
            team_id=team_id,
            merged_commits=tuple(integration_raw.get("merged_commits", [])),
            current_source_branch=integration_raw.get("current_source_branch"),
            merge_attempt=int(integration_raw.get("merge_attempt", 0)),
            conflicted_files=tuple(Path(item) for item in integration_raw.get("conflicted_files", [])),
            blocked_by_validation=bool(integration_raw.get("blocked_by_validation", False)),
            validation_repair_task_id=integration_raw.get("validation_repair_task_id"),
            validation_reports=tuple(
                ValidationReport(
                    command=str(item["command"]),
                    scope=str(item["scope"]),  # type: ignore[arg-type]
                    exit_code=int(item["exit_code"]),
                    head=str(item["head"]),
                    ran_at=datetime.fromisoformat(str(item["ran_at"])),
                )
                for item in integration_raw.get("validation_reports", [])
            ),
            updated_at=datetime.fromisoformat(str(integration_raw["updated_at"])),
        )
        return TeamSnapshot(team, members, tasks, scans, integration)

    def team_for_lead(self, session_id: str) -> TeamRecord | None:
        """查找当前由一个主会话管理的存续团队。

        Args:
            session_id: 主会话 ID。

        Returns:
            找到时返回 TeamRecord；没有绑定时返回 None。
        """

        index = _read_json(self._index_path, {"teams": {}})
        for team_id, item in index.get("teams", {}).items():
            if item.get("lead_session_id") == session_id:
                return self.load_team(team_id).team
        return None

    def add_member(self, member: TeammateRecord) -> TeamSnapshot:
        """把一个 starting 成员加入团队并创建其邮箱和 cursor。

        Args:
            member: Supervisor 已经准备好 Worktree、generation 和租约的成员。

        Returns:
            写入后的完整团队快照。
        """

        directory = self.team_dir(member.team_id)
        lock_path = directory / "locks" / "team.lock"
        with ExclusiveFileLock(lock_path, f"member:{member.agent_id}"):
            snapshot = self.load_team(member.team_id)
            if snapshot.team.lifecycle is not TeamLifecycle.ACTIVE:
                raise TeamStoreError("团队正在清理，不能增加成员")
            if member.name == "lead" or any(item.name == member.name for item in snapshot.members):
                raise TeamStoreError(f"团队成员名称不可用：{member.name}")
            _atomic_json(directory / "members" / f"{member.agent_id}.json", _member_to_json(member))
            (directory / "mailboxes" / f"{member.agent_id}.jsonl").touch(exist_ok=False)
            _atomic_json(directory / "cursors" / f"{member.agent_id}.json", {"byte_offset": 0, "last_message_id": None})
            _atomic_json(directory / "runtime" / f"{member.agent_id}.json", {})
            team = replace(
                snapshot.team,
                member_ids=(*snapshot.team.member_ids, member.agent_id),
                updated_at=_now(),
                revision=snapshot.team.revision + 1,
            )
            _atomic_json(directory / "team.json", _team_to_json(team))
        return self.load_team(member.team_id)

    def update_member(
        self,
        actor: TeamActorContext,
        member_id: str,
        mutation: Callable[[TeammateRecord], TeammateRecord],
        *,
        lease_token: str | None = None,
    ) -> TeammateRecord:
        """在成员锁内校验 Actor 和租约后替换一条成员记录。

        Args:
            actor: 本地运行时确认的 Lead 或成员身份。
            member_id: 要更新的成员 ID。
            mutation: 根据锁内最新记录生成新记录的函数。
            lease_token: 成员 Host 写状态时提供的租约原文；Lead 更新时为空。

        Returns:
            已持久化的新成员记录。
        """

        self.require_actor(actor)
        directory = self.team_dir(actor.team_id)
        with ExclusiveFileLock(directory / "locks" / f"member-{member_id}.lock", actor.actor_id):
            path = directory / "members" / f"{member_id}.json"
            current = _member_from_json(_read_json(path))
            if actor.actor_kind == "member":
                if actor.actor_id != member_id or actor.generation != current.runtime_generation:
                    raise TeamStoreError("成员 Host 身份或 generation 已失效")
                digest = hashlib.sha256((lease_token or "").encode()).hexdigest()
                if digest != current.lease_token_hash:
                    raise TeamStoreError("成员 Host 租约无效")
            updated = mutation(current)
            if updated.agent_id != current.agent_id or updated.team_id != current.team_id:
                raise TeamStoreError("成员更新不能改变身份或团队")
            _atomic_json(path, _member_to_json(updated))
            if updated.state is not current.state:
                self.append_event(
                    TeamEvent(
                        event_id=f"event-{secrets.token_hex(6)}",
                        team_id=actor.team_id,
                        kind="member_lifecycle",
                        actor_id=member_id,
                        payload={
                            "member_id": member_id,
                            "name": updated.name,
                            "from_state": current.state.value,
                            "to_state": updated.state.value,
                        },
                        created_at=_now(),
                    )
                )
            return updated

    def append_event(self, event: TeamEvent) -> None:
        """把一条结构化系统事件追加并同步到团队事件文件。

        Args:
            event: 已包含团队、类型、触发者和具体字段的事件。

        Returns:
            完整 JSON 行同步到磁盘后不返回数据。

        Raises:
            TeamStoreError: 事件属于其他团队或文件无法追加、同步。
        """

        directory = self.team_dir(event.team_id)
        if not (directory / "team.json").is_file():
            raise TeamStoreError(f"团队不存在：{event.team_id}")
        payload = {
            "event_id": event.event_id,
            "team_id": event.team_id,
            "kind": event.kind,
            "actor_id": event.actor_id,
            "payload": event.payload,
            "created_at": event.created_at.isoformat(),
        }
        try:
            with ExclusiveFileLock(
                directory / "locks" / "events.lock",
                event.actor_id or "system",
            ):
                with (directory / "events.jsonl").open(
                    "a",
                    encoding="utf-8",
                    newline="\n",
                ) as handle:
                    handle.write(
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
        except OSError as exc:
            raise TeamStoreError(f"无法写入团队事件：{exc}") from exc

    def set_member_current_task(
        self,
        team_id: str,
        member_id: str,
        task_id: str | None,
        *,
        actor: str,
    ) -> TeammateRecord:
        """由 TaskBoard 同步成员当前任务，不接受模型直接调用。

        Args:
            team_id: 成员所属团队 ID。
            member_id: 要更新的成员 ID。
            task_id: 新的 working 任务；清空时传 None。
            actor: 写入锁记录和错误信息使用的内部调用者名称。

        Returns:
            已写入磁盘的新成员记录。
        """

        directory = self.team_dir(team_id)
        path = directory / "members" / f"{member_id}.json"
        with ExclusiveFileLock(directory / "locks" / f"member-{member_id}.lock", actor):
            current = _member_from_json(_read_json(path))
            updated = replace(current, current_task_id=task_id, updated_at=_now())
            _atomic_json(path, _member_to_json(updated))
            return updated

    def save_runtime_prompt(
        self,
        team_id: str,
        member_id: str,
        prompt: str,
    ) -> None:
        """保存成员尚未处理的首次指令，供独立 Host 启动后读取。

        Args:
            team_id: 成员所属团队 ID。
            member_id: 花名册中的不可变成员 ID。
            prompt: Lead 创建成员时给出的完整首次工作说明。

        Returns:
            写入成功时不返回数据。

        Raises:
            TeamStoreError: 成员不存在，或运行记录无法原子写入。
        """

        directory = self.team_dir(team_id)
        if not (directory / "members" / f"{member_id}.json").is_file():
            raise TeamStoreError(f"团队成员不存在：{member_id}")
        with ExclusiveFileLock(
            directory / "locks" / f"runtime-{member_id}.lock",
            f"runtime-prompt:{member_id}",
        ):
            _atomic_json(
                directory / "runtime" / f"{member_id}.json",
                {"pending_prompt": prompt},
            )

    def load_runtime_prompt(self, team_id: str, member_id: str) -> str:
        """读取成员尚未确认完成的首次指令，但不提前删除它。

        Args:
            team_id: 成员所属团队 ID。
            member_id: 花名册中的不可变成员 ID。

        Returns:
            已保存的首次指令；没有待处理指令时返回空字符串。

        Raises:
            TeamStoreError: 运行记录不是对象，或 ``pending_prompt`` 不是文本。
        """

        raw = _read_json(self.team_dir(team_id) / "runtime" / f"{member_id}.json")
        prompt = raw.get("pending_prompt", "")
        if not isinstance(prompt, str):
            raise TeamStoreError("成员运行记录中的 pending_prompt 必须是文本")
        return prompt

    def clear_runtime_prompt(self, team_id: str, member_id: str) -> None:
        """在成员成功处理首次指令后清空持久化副本。

        Args:
            team_id: 成员所属团队 ID。
            member_id: 花名册中的不可变成员 ID。

        Returns:
            清空成功时不返回数据。重复调用仍视为成功。
        """

        directory = self.team_dir(team_id)
        with ExclusiveFileLock(
            directory / "locks" / f"runtime-{member_id}.lock",
            f"runtime-prompt:{member_id}",
        ):
            _atomic_json(directory / "runtime" / f"{member_id}.json", {})

    def update_integration(
        self,
        actor: TeamActorContext,
        mutation: Callable[[TeamIntegrationState], TeamIntegrationState],
    ) -> TeamIntegrationState:
        """在独占锁内替换团队的合并和验证状态。

        Args:
            actor: 当前有效 Lead 身份；成员不能更新主分支集成记录。
            mutation: 接收锁内最新状态并返回新状态的函数。

        Returns:
            已原子写入 ``integration.json`` 的新状态。

        Raises:
            TeamStoreError: Actor 不是当前 Lead，或 mutation 改变了 team ID。
        """

        self.require_actor(actor)
        if actor.actor_kind != "lead":
            raise TeamStoreError("只有 Lead 能更新合并和验证状态")
        directory = self.team_dir(actor.team_id)
        with ExclusiveFileLock(
            directory / "locks" / "integration.lock", actor.actor_id
        ):
            current = self.load_team(actor.team_id).integration
            updated = mutation(current)
            if updated.team_id != current.team_id:
                raise TeamStoreError("集成状态更新不能改变团队 ID")
            _atomic_json(directory / "integration.json", _integration_to_json(updated))
            return updated

    def remove_partial_member(self, team_id: str, agent_id: str) -> None:
        """删除启动失败且尚未产生工作成果的成员旁路文件。

        Args:
            team_id: 成员所属团队 ID。
            agent_id: 本次启动失败的成员 ID。

        Returns:
            成员记录和旁路文件不存在时也视为完成。
        """

        directory = self.team_dir(team_id)
        with ExclusiveFileLock(directory / "locks" / "team.lock", f"rollback:{agent_id}"):
            team = _team_from_json(_read_json(directory / "team.json"))
            for path in (
                directory / "members" / f"{agent_id}.json",
                directory / "mailboxes" / f"{agent_id}.jsonl",
                directory / "cursors" / f"{agent_id}.json",
                directory / "runtime" / f"{agent_id}.json",
            ):
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    raise TeamStoreError(f"无法回滚成员文件 {path.name}：{exc}") from exc
            updated = replace(
                team,
                member_ids=tuple(item for item in team.member_ids if item != agent_id),
                updated_at=_now(),
                revision=team.revision + 1,
            )
            _atomic_json(directory / "team.json", _team_to_json(updated))

    def require_actor(self, actor: TeamActorContext) -> TeamRecord:
        """确认调用者属于团队且 generation 仍是当前版本。

        Args:
            actor: 工具上下文提供的可信调用者身份。

        Returns:
            当前团队记录。
        """

        snapshot = self.load_team(actor.team_id)
        if snapshot.team.lifecycle is not TeamLifecycle.ACTIVE:
            raise TeamStoreError("团队当前不接受新的写操作")
        if actor.actor_kind == "lead":
            if actor.generation != snapshot.team.lead_generation:
                raise TeamStoreError("Lead generation 已失效")
        elif actor.actor_kind == "member":
            member = next(
                (item for item in snapshot.members if item.agent_id == actor.actor_id),
                None,
            )
            if member is None:
                raise TeamStoreError("成员不属于当前团队")
            if actor.generation != member.runtime_generation:
                raise TeamStoreError("成员 generation 已失效")
        else:
            raise TeamStoreError(f"未知团队调用者类型：{actor.actor_kind}")
        return snapshot.team

    def require_cleanup_actor(self, actor: TeamActorContext) -> TeamRecord:
        """允许当前 Lead 在 active、cleaning 或 cleanup_failed 状态查询/续清理。

        Args:
            actor: ToolContext 提供的本地可信团队身份。

        Returns:
            generation 仍有效且调用者为 Lead 的团队记录。

        Raises:
            TeamStoreError: 调用者不是 Lead 或 generation 已失效。
        """

        snapshot = self.load_team(actor.team_id)
        if actor.actor_kind != "lead":
            raise TeamStoreError("只有 Lead 能查询或继续清理团队")
        if actor.generation != snapshot.team.lead_generation:
            raise TeamStoreError("Lead generation 已失效")
        return snapshot.team

    def takeover(self, team_id: str, new_lead_session_id: str) -> TeamRecord:
        """把孤立团队交给新主会话并使旧 Lead generation 失效。

        Args:
            team_id: 等待接管的团队 ID。
            new_lead_session_id: 用户确认接管后的当前主会话 ID。

        Returns:
            已替换 Lead 会话并递增 generation 的团队记录。

        Raises:
            TeamStoreError: 新会话已有团队、目标团队不活跃或状态文件无效。
        """

        with ExclusiveFileLock(self._index_lock, f"takeover:{new_lead_session_id}"):
            index = _read_json(self._index_path)
            for other_id, item in index.get("teams", {}).items():
                if other_id != team_id and item.get("lead_session_id") == new_lead_session_id:
                    raise TeamStoreError("新 Lead 会话已经拥有其他团队")
            directory = self.team_dir(team_id)
            team = _team_from_json(_read_json(directory / "team.json"))
            if team.lifecycle is not TeamLifecycle.ACTIVE:
                raise TeamStoreError(
                    f"团队处于 {team.lifecycle.value} 状态，不能更换 Lead"
                )
            updated = replace(
                team,
                lead_session_id=new_lead_session_id,
                lead_generation=team.lead_generation + 1,
                updated_at=_now(),
                revision=team.revision + 1,
            )
            _atomic_json(directory / "team.json", _team_to_json(updated))
            teams = dict(index["teams"])
            teams[team_id] = {"name": updated.name, "lead_session_id": new_lead_session_id}
            _atomic_json(self._index_path, {"revision": int(index.get("revision", 0)) + 1, "teams": teams})
            return updated

    def begin_cleanup(self, actor: TeamActorContext) -> TeamSnapshot:
        """在删除前置检查已经通过后冻结团队写入。

        Args:
            actor: 已通过删除预检的当前 Lead 身份。

        Returns:
            lifecycle 已改成 ``cleaning`` 的团队快照。

        Raises:
            TeamStoreError: Lead 身份或 generation 已失效。
        """

        self.require_actor(actor)
        directory = self.team_dir(actor.team_id)
        with ExclusiveFileLock(directory / "locks" / "team.lock", actor.actor_id):
            snapshot = self.load_team(actor.team_id)
            updated = replace(
                snapshot.team,
                lifecycle=TeamLifecycle.CLEANING,
                updated_at=_now(),
                revision=snapshot.team.revision + 1,
            )
            _atomic_json(directory / "team.json", _team_to_json(updated))
        return self.load_team(actor.team_id)

    def mark_cleanup_failed(self, team_id: str, error: str) -> None:
        """记录实际删除阶段的错误，保留剩余资源供再次清理。

        Args:
            team_id: 清理过程中发生错误的团队 ID。
            error: 适合展示给用户的实际失败原因。

        Returns:
            团队目录已经不存在时直接返回；否则保存失败状态后不返回数据。
        """

        directory = self.team_dir(team_id)
        path = directory / "team.json"
        if not path.exists():
            return
        team = _team_from_json(_read_json(path))
        _atomic_json(path, _team_to_json(replace(team, lifecycle=TeamLifecycle.CLEANUP_FAILED, updated_at=_now(), revision=team.revision + 1)))
        _atomic_json(directory / "cleanup-error.json", {"error": error, "at": _now().isoformat()})

    def load_cleanup_progress(self, team_id: str) -> tuple[str, ...]:
        """读取上一次清理已经实际移除的资源标识。

        Args:
            team_id: 正处于 cleaning 或 cleanup_failed 的团队 ID。

        Returns:
            按完成顺序保存的资源标识；首次清理返回空元组。
        """

        raw = _read_json(
            self.team_dir(team_id) / "cleanup-progress.json",
            {"removed_resources": []},
        )
        return tuple(str(item) for item in raw.get("removed_resources", []))

    def save_cleanup_progress(
        self,
        team_id: str,
        removed_resources: tuple[str, ...],
    ) -> None:
        """在每项外部资源删除后原子保存可续清理进度。

        Args:
            team_id: 正在清理的团队 ID。
            removed_resources: 已确认删除的 Worktree、分支或会话绑定标识。

        Returns:
            进度文件同步到团队目录后不返回数据。
        """

        _atomic_json(
            self.team_dir(team_id) / "cleanup-progress.json",
            {"removed_resources": list(removed_resources)},
        )

    def finish_cleanup(self, team_id: str) -> None:
        """可回滚地移除 index 项和整个团队目录，不创建归档记录。

        Args:
            team_id: 外部资源已经清理完毕的团队 ID。

        Returns:
            index 和团队目录均不存在时不返回数据。

        Raises:
            TeamStoreError: 目录改名、index 更新或最终目录删除失败；能够回滚
                时会恢复原目录和 index，使下一次 TeamDelete 可以续清理。
        """

        with ExclusiveFileLock(self._index_lock, f"cleanup:{team_id}"):
            index = _read_json(self._index_path)
            teams = dict(index.get("teams", {}))
            entry = teams.get(team_id)
            directory = self.team_dir(team_id)
            tombstone = self.root / f".deleting-{team_id}"
            try:
                directory.replace(tombstone)
            except OSError as exc:
                raise TeamStoreError(f"无法冻结待删除团队目录：{exc}") from exc
            teams.pop(team_id, None)
            try:
                _atomic_json(
                    self._index_path,
                    {
                        "revision": int(index.get("revision", 0)) + 1,
                        "teams": teams,
                    },
                )
            except Exception:
                tombstone.replace(directory)
                raise
            try:
                shutil.rmtree(tombstone)
            except OSError as exc:
                try:
                    tombstone.replace(directory)
                    if entry is not None:
                        teams[team_id] = entry
                    _atomic_json(
                        self._index_path,
                        {
                            "revision": int(index.get("revision", 0)) + 2,
                            "teams": teams,
                        },
                    )
                except Exception as rollback_exc:
                    raise TeamStoreError(
                        f"团队目录删除失败且回滚失败：{exc}；{rollback_exc}"
                    ) from rollback_exc
                raise TeamStoreError(f"无法删除团队目录：{exc}") from exc
