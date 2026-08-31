"""共享任务的依赖、认领、状态转换和检查轮次。"""

from __future__ import annotations

import secrets
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from mycode.models.teams import (
    ClaimScanRound,
    TeamActorContext,
    TeamEvent,
    TeamTaskAttempt,
    TeamTaskCreateRequest,
    TeamTaskPriority,
    TeamTaskQuery,
    TeamTaskRecord,
    TeamTaskStatus,
    TeamTaskUpdateRequest,
    TeamTaskView,
    TeammateState,
)
from mycode.teams.locks import ExclusiveFileLock
from mycode.teams.store import (
    TeamStateStore,
    TeamStoreError,
    _atomic_json,
    _read_json,
    task_from_json,
    task_to_json,
)


class TeamTaskError(RuntimeError):
    """表示任务不存在、依赖无效、状态转换非法或调用者越权。"""


def _now() -> datetime:
    """返回任务和检查轮次持久化使用的带时区当前时间。

    Returns:
        当前本地时区的 datetime。
    """

    return datetime.now().astimezone()


def _scan_to_json(scan: ClaimScanRound) -> dict[str, Any]:
    """把自主认领检查轮次转换成 tasks.json 字段。

    Args:
        scan: 需要持久化的检查轮次。

    Returns:
        只包含 JSON 标量和数组的字典。
    """

    return {
        "round_id": scan.round_id,
        "team_id": scan.team_id,
        "task_ids": list(scan.task_ids),
        "expected_member_ids": list(scan.expected_member_ids),
        "finished_member_ids": list(scan.finished_member_ids),
        "claimed_task_ids": list(scan.claimed_task_ids),
        "created_at": scan.created_at.isoformat(),
    }


def _scan_from_json(raw: dict[str, Any]) -> ClaimScanRound:
    """把 tasks.json 中的一轮检查还原为 ClaimScanRound。

    Args:
        raw: JSON 解码后的轮次字段。

    Returns:
        已恢复时间和 ID 元组的检查轮次。
    """

    return ClaimScanRound(
        round_id=str(raw["round_id"]),
        team_id=str(raw["team_id"]),
        task_ids=tuple(str(item) for item in raw.get("task_ids", [])),
        expected_member_ids=tuple(str(item) for item in raw.get("expected_member_ids", [])),
        finished_member_ids=tuple(str(item) for item in raw.get("finished_member_ids", [])),
        claimed_task_ids=tuple(str(item) for item in raw.get("claimed_task_ids", [])),
        created_at=datetime.fromisoformat(str(raw["created_at"])),
    )


class TeamTaskBoard:
    """在 tasks 锁内维护共享任务和成员自主认领结果。

    Attributes:
        store: 提供团队身份、成员状态和原子 JSON 路径的持久化入口。
    """

    def __init__(self, store: TeamStateStore) -> None:
        """保存团队 Store；实际任务文件按 Actor 的 team ID 选择。

        Args:
            store: 当前工作区唯一的 TeamStateStore。

        Returns:
            不返回数据。
        """

        self.store = store

    def create(
        self,
        actor: TeamActorContext,
        request: TeamTaskCreateRequest,
    ) -> TeamTaskRecord:
        """由 Lead 创建任务并拒绝无效或成环依赖。

        Args:
            actor: 本地确认的团队调用者，必须是当前 Lead。
            request: 标题、说明、类型、优先级和直接前置任务。

        Returns:
            已写入 tasks.json 的新任务。
        """

        self._require_lead(actor)
        with self._lock(actor.team_id, actor.actor_id):
            revision, tasks, scans = self._load(actor.team_id)
            known = {item.task_id for item in tasks}
            if any(item not in known for item in request.blocked_by):
                raise TeamTaskError("任务依赖必须引用当前团队中已经存在的任务")
            task_id = f"task-{secrets.token_hex(5)}"
            now = _now()
            task = TeamTaskRecord(
                task_id=task_id,
                team_id=actor.team_id,
                title=request.title.strip(),
                description=request.description.strip(),
                task_kind=request.task_kind,
                priority=request.priority,
                blocked_by=tuple(dict.fromkeys(request.blocked_by)),
                created_at=now,
                updated_at=now,
            )
            tasks.append(task)
            self._validate_graph(tasks)
            self._save(actor.team_id, revision + 1, tasks, scans)
            return task

    def list(
        self,
        actor: TeamActorContext,
        query: TeamTaskQuery | None = None,
    ) -> tuple[TeamTaskView, ...]:
        """返回调用者有权查看的任务及读取时计算的依赖状态。

        Args:
            actor: 当前 Lead 或成员身份。
            query: 可选状态、负责人和只看可领取条件。

        Returns:
            按优先级和创建时间稳定排序的任务视图。
        """

        self.store.require_actor(actor)
        _, tasks, _ = self._load(actor.team_id)
        views = [self._view(item, tasks) for item in tasks]
        query = query or TeamTaskQuery()
        if actor.actor_kind == "member":
            views = [
                item for item in views
                if (
                    item.task.owner_id == actor.actor_id
                    or (item.claimable and item.task.owner_id is None)
                )
            ]
        if query.status is not None:
            views = [item for item in views if item.task.status is query.status]
        if query.owner_id is not None:
            views = [item for item in views if item.task.owner_id == query.owner_id]
        if query.claimable_only:
            views = [item for item in views if item.claimable]
        order = {
            TeamTaskPriority.HIGH: 0,
            TeamTaskPriority.NORMAL: 1,
            TeamTaskPriority.LOW: 2,
        }
        views.sort(key=lambda item: (order[item.task.priority], item.task.created_at, item.task.task_id))
        return tuple(views)

    def get(self, actor: TeamActorContext, task_id: str) -> TeamTaskView:
        """读取一项任务，并对成员应用与列表相同的可见范围。

        Args:
            actor: 当前 Lead 或成员身份。
            task_id: 要查询的团队任务 ID。

        Returns:
            任务和派生依赖状态。
        """

        for view in self.list(actor):
            if view.task.task_id == task_id:
                return view
        raise TeamTaskError(f"任务不存在或当前成员不可见：{task_id}")

    def claim(
        self,
        actor: TeamActorContext,
        task_id: str,
        round_id: str | None = None,
    ) -> TeamTaskRecord:
        """让成员原子认领未分配任务或开始分配给自己的任务。

        Args:
            actor: 当前成员身份。
            task_id: 准备进入 working 的任务 ID。
            round_id: 本次唤醒对应的检查轮次；主动查看时可以为空。

        Returns:
            已进入 working 的任务记录。
        """

        self._require_member(actor)
        with self._lock(actor.team_id, actor.actor_id):
            revision, tasks, scans = self._load(actor.team_id)
            index = self._index(tasks, task_id)
            task = tasks[index]
            view = self._view(task, tasks)
            if not view.claimable:
                raise TeamTaskError("任务当前不可领取：可能已被领取、依赖未完成或已结束")
            if task.owner_id not in {None, actor.actor_id}:
                raise TeamTaskError("任务已经分配给其他成员")
            snapshot = self.store.load_team(actor.team_id)
            member = next((item for item in snapshot.members if item.agent_id == actor.actor_id), None)
            if member is None:
                raise TeamTaskError("当前成员不存在")
            if member.current_task_id is not None or any(
                item.status is TeamTaskStatus.WORKING and item.owner_id == actor.actor_id
                for item in tasks
            ):
                raise TeamTaskError("成员已经有一个 working 任务")
            attempts = task.attempts
            if attempts and attempts[-1].ended_at is None:
                current = replace(attempts[-1], owner_id=actor.actor_id, paused_at=None)
                attempts = (*attempts[:-1], current)
            else:
                if len(attempts) >= 2:
                    raise TeamTaskError("任务已经用完两次执行机会")
                attempts = (*attempts, TeamTaskAttempt(len(attempts) + 1, actor.actor_id, _now()))
            updated = replace(
                task,
                status=TeamTaskStatus.WORKING,
                owner_id=actor.actor_id,
                attempts=attempts,
                updated_at=_now(),
            )
            tasks[index] = updated
            if round_id is not None:
                scan_index = next((i for i, item in enumerate(scans) if item.round_id == round_id), None)
                if scan_index is not None:
                    scan = scans[scan_index]
                    scans[scan_index] = replace(
                        scan,
                        claimed_task_ids=tuple(dict.fromkeys((*scan.claimed_task_ids, task_id))),
                    )
            self._save(actor.team_id, revision + 1, tasks, scans)
            try:
                self.store.set_member_current_task(
                    actor.team_id, actor.actor_id, task_id, actor="task-board"
                )
            except Exception:
                # tasks.json 和 team.json 是两个独立快照。成员快照写入
                # 失败时立即恢复任务旧值，避免出现“任务已领取但
                # 成员没有 current_task_id”的可观察不一致。
                tasks[index] = task
                self._save(actor.team_id, revision + 2, tasks, scans)
                raise
            return updated

    def update(
        self,
        actor: TeamActorContext,
        request: TeamTaskUpdateRequest,
    ) -> TeamTaskRecord:
        """按 Lead/成员字段权限更新任务并执行状态机。

        Args:
            actor: 当前 Lead 或成员身份。
            request: 显式列出的待修改字段。

        Returns:
            已持久化的新任务记录。
        """

        self.store.require_actor(actor)
        with self._lock(actor.team_id, actor.actor_id):
            revision, tasks, scans = self._load(actor.team_id)
            index = self._index(tasks, request.task_id)
            task = tasks[index]
            if actor.actor_kind == "member":
                if task.owner_id != actor.actor_id:
                    raise TeamTaskError("成员只能更新本人负责的任务")
                if request.owner is not None or request.priority is not None or request.add_blocked_by or request.remove_blocked_by:
                    raise TeamTaskError("成员不能修改负责人、优先级或依赖")
            else:
                request = self._normalize_lead_owner(actor, task, request)
            updated = self._apply_update(actor, task, request, tasks)
            tasks[index] = updated
            self._validate_graph(tasks)
            self._save(actor.team_id, revision + 1, tasks, scans)
            if task.status is TeamTaskStatus.WORKING and updated.status is not TeamTaskStatus.WORKING and task.owner_id:
                self.store.set_member_current_task(actor.team_id, task.owner_id, None, actor="task-board")
            return updated

    def _normalize_lead_owner(
        self,
        actor: TeamActorContext,
        task: TeamTaskRecord,
        request: TeamTaskUpdateRequest,
    ) -> TeamTaskUpdateRequest:
        """把 Lead 提供的成员名称解析为 ID，并执行重新分配门禁。

        Args:
            actor: 已通过 generation 校验的当前 Lead。
            task: 锁内读取的任务最新记录。
            request: Lead 提交的显式更新字段。

        Returns:
            owner 已统一为成员 ID 的更新请求；没有 owner 更新时原样返回。

        Raises:
            TeamTaskError: 目标成员不存在或已终止，或者任务没有先暂停并
                保存交接内容。
        """

        if request.owner is None:
            return request
        snapshot = self.store.load_team(actor.team_id)
        target = next(
            (
                member
                for member in snapshot.members
                if request.owner in {member.agent_id, member.name}
            ),
            None,
        )
        if target is None:
            raise TeamTaskError(f"负责人不是当前团队成员：{request.owner}")
        if target.state in {TeammateState.FAILED, TeammateState.TERMINATED}:
            raise TeamTaskError(f"成员 {target.name} 当前不能接手任务")
        if target.agent_id == task.owner_id:
            return replace(request, owner=target.agent_id)

        handoff = request.progress or request.result or task.progress or task.result
        previous = next(
            (member for member in snapshot.members if member.agent_id == task.owner_id),
            None,
        )
        previous_failed = previous is not None and previous.state in {
            TeammateState.FAILED,
            TeammateState.TERMINATED,
        }
        if task.status is TeamTaskStatus.WORKING:
            if not previous_failed:
                raise TeamTaskError("working 任务必须先暂停并保存交接内容，再重新分配")
            if request.status is not TeamTaskStatus.TODO or not handoff:
                raise TeamTaskError("故障成员任务转交时必须改为 todo 并保存交接内容")
        elif task.owner_id is not None and task.status is TeamTaskStatus.TODO:
            paused = bool(task.attempts and task.attempts[-1].paused_at is not None)
            if not paused or not handoff:
                raise TeamTaskError("已指派任务重新分配前必须暂停并保存交接内容")
        elif task.status is TeamTaskStatus.FAILED:
            failure_context = bool(
                handoff
                or (
                    task.attempts
                    and task.attempts[-1].failure_reason
                )
            )
            if not failure_context:
                raise TeamTaskError("失败任务转交前必须保留失败原因或阶段结果")
        return replace(request, owner=target.agent_id)

    def open_scan(
        self,
        team_id: str,
        task_ids: tuple[str, ...],
        member_ids: tuple[str, ...],
    ) -> ClaimScanRound:
        """为实际唤醒成功的成员建立一次任务检查轮次。

        Args:
            team_id: 要检查共享任务的团队 ID。
            task_ids: 本轮希望成员查看的候选任务 ID。
            member_ids: Supervisor 已确认唤醒成功的成员 ID。

        Returns:
            保存有效候选任务和待回复成员的检查轮次。
        """

        with self._lock(team_id, "supervisor"):
            revision, tasks, scans = self._load(team_id)
            known = {item.task_id for item in tasks}
            valid_tasks = tuple(item for item in task_ids if item in known)
            scan = ClaimScanRound(
                round_id=f"scan-{secrets.token_hex(5)}",
                team_id=team_id,
                task_ids=valid_tasks,
                expected_member_ids=tuple(dict.fromkeys(member_ids)),
                finished_member_ids=(),
                claimed_task_ids=(),
                created_at=_now(),
            )
            scans.append(scan)
            self._save(team_id, revision + 1, tasks, scans)
            return scan

    def finish_scan(
        self,
        round_id: str,
        member_id: str,
        *,
        team_id: str,
    ) -> tuple[TeamEvent, ...]:
        """登记成员完成检查，并在全员结束后产生无人认领事件。

        Args:
            round_id: Host 本轮收到的检查轮次 ID。
            member_id: 刚结束 Agent Loop 的成员 ID。
            team_id: 轮次所属团队 ID。

        Returns:
            本次刚产生的系统事件；轮次尚未完成时为空。
        """

        with self._lock(team_id, member_id):
            revision, tasks, scans = self._load(team_id)
            index = next((i for i, item in enumerate(scans) if item.round_id == round_id), None)
            if index is None:
                raise TeamTaskError(f"检查轮次不存在：{round_id}")
            scan = scans[index]
            if member_id not in scan.expected_member_ids:
                raise TeamTaskError("成员不属于本轮实际唤醒集合")
            updated = replace(
                scan,
                finished_member_ids=tuple(dict.fromkeys((*scan.finished_member_ids, member_id))),
            )
            scans[index] = updated
            events: list[TeamEvent] = []
            if set(updated.finished_member_ids) == set(updated.expected_member_ids):
                for task_id in updated.task_ids:
                    task = next((item for item in tasks if item.task_id == task_id), None)
                    if task is not None and self._view(task, tasks).claimable and task_id not in updated.claimed_task_ids:
                        events.append(
                            TeamEvent(
                                event_id=f"event-{secrets.token_hex(6)}",
                                team_id=team_id,
                                kind="unclaimed_task",
                                actor_id=None,
                                payload={"task_id": task_id, "round_id": round_id},
                                created_at=_now(),
                            )
                        )
                scans.pop(index)
            self._save(team_id, revision + 1, tasks, scans)
            for event in events:
                self._append_event(event)
            return tuple(events)

    def _apply_update(
        self,
        actor: TeamActorContext,
        task: TeamTaskRecord,
        request: TeamTaskUpdateRequest,
        tasks: list[TeamTaskRecord],
    ) -> TeamTaskRecord:
        """根据当前状态和字段权限生成一条更新后任务。

        Args:
            actor: 发起更新的本地 Lead 或成员身份。
            task: 更新前的任务记录。
            request: 状态、负责人、依赖、优先级和交接内容变更。
            tasks: 同一团队的完整任务列表，用来校验依赖。

        Returns:
            已校验字段权限、状态转换和 attempt 门限的新任务记录。

        Raises:
            TeamTaskError: 更新违反任务状态、负责人、依赖或尝试次数规则。
        """

        if task.status in {TeamTaskStatus.COMPLETED, TeamTaskStatus.CANCELLED}:
            raise TeamTaskError("完成或取消的任务不能重新打开")
        owner = request.owner if actor.actor_kind == "lead" and request.owner is not None else task.owner_id
        priority = request.priority if actor.actor_kind == "lead" and request.priority is not None else task.priority
        blocked = [item for item in task.blocked_by if item not in request.remove_blocked_by]
        blocked.extend(item for item in request.add_blocked_by if item not in blocked)
        status = request.status or task.status
        attempts = task.attempts
        completed_at = task.completed_at
        if task.status is TeamTaskStatus.WORKING and status is TeamTaskStatus.TODO:
            if not attempts:
                raise TeamTaskError("working 任务缺少执行记录")
            attempts = (*attempts[:-1], replace(attempts[-1], paused_at=_now()))
        elif task.status is TeamTaskStatus.WORKING and status in {TeamTaskStatus.COMPLETED, TeamTaskStatus.FAILED}:
            if not attempts:
                raise TeamTaskError("working 任务缺少执行记录")
            failure = request.failure_reason if status is TeamTaskStatus.FAILED else None
            if status is TeamTaskStatus.FAILED and not failure:
                raise TeamTaskError("任务失败必须提供 failure_reason")
            attempts = (*attempts[:-1], replace(attempts[-1], ended_at=_now(), failure_reason=failure))
            completed_at = _now() if status is TeamTaskStatus.COMPLETED else None
            if status is TeamTaskStatus.COMPLETED:
                if task.task_kind == "code" and not (request.commit_hashes or task.commit_hashes):
                    raise TeamTaskError("代码任务完成前必须报告提交标识")
                if task.task_kind == "research" and not (request.result or task.result):
                    raise TeamTaskError("调查任务完成前必须提交结构化结果")
        elif task.status is TeamTaskStatus.FAILED and status is TeamTaskStatus.TODO:
            if actor.actor_kind != "lead":
                raise TeamTaskError("只有 Lead 能重置失败任务")
            if len(attempts) != 1:
                raise TeamTaskError("任务第二次失败后不能再次重置")
        elif task.status is TeamTaskStatus.TODO and status is TeamTaskStatus.CANCELLED:
            if actor.actor_kind != "lead":
                raise TeamTaskError("只有 Lead 能取消任务")
        elif status != task.status:
            raise TeamTaskError(f"不允许的任务状态转换：{task.status.value} → {status.value}")
        return replace(
            task,
            status=status,
            owner_id=owner,
            priority=priority,
            blocked_by=tuple(blocked),
            progress=request.progress if request.progress is not None else task.progress,
            result=request.result if request.result is not None else task.result,
            commit_hashes=request.commit_hashes if request.commit_hashes is not None else task.commit_hashes,
            attempts=attempts,
            updated_at=_now(),
            completed_at=completed_at,
        )

    @staticmethod
    def _view(task: TeamTaskRecord, tasks: list[TeamTaskRecord]) -> TeamTaskView:
        """根据当前全部任务计算依赖和可领取派生字段。

        Args:
            task: 需要生成视图的原始任务。
            tasks: 同一锁内读取的团队任务列表。

        Returns:
            包含 blocked、blocks、assigned 和 claimable 的任务视图。
        """

        completed = {item.task_id for item in tasks if item.status is TeamTaskStatus.COMPLETED}
        blocked = any(item not in completed for item in task.blocked_by)
        blocks = tuple(item.task_id for item in tasks if task.task_id in item.blocked_by)
        claimable = (
            task.status is TeamTaskStatus.TODO
            and not blocked
            and (not task.attempts or len(task.attempts) < 2 or task.attempts[-1].ended_at is None)
        )
        return TeamTaskView(task, blocked, blocks, task.owner_id is not None, claimable)

    @staticmethod
    def _validate_graph(tasks: list[TeamTaskRecord]) -> None:
        """拒绝自环、跨团队缺失引用和循环依赖。

        Args:
            tasks: 本次修改后准备整体写回的任务列表。

        Returns:
            依赖图合法时不返回数据。

        Raises:
            TeamTaskError: 依赖自身、引用不存在任务或形成循环。
        """

        graph = {item.task_id: item.blocked_by for item in tasks}
        if any(node in deps for node, deps in graph.items()):
            raise TeamTaskError("任务不能依赖自身")
        visiting: set[str] = set()
        visited: set[str] = set()
        def visit(node: str) -> None:
            """深度遍历一个任务节点，并用 visiting 集合检测回边。

            Args:
                node: 当前正在检查的任务 ID。

            Returns:
                当前节点及其依赖检查完成后不返回数据。

            Raises:
                TeamTaskError: 发现循环依赖或引用了团队外任务。
            """

            if node in visiting:
                raise TeamTaskError("任务依赖形成循环")
            if node in visited:
                return
            visiting.add(node)
            for dependency in graph.get(node, ()):
                if dependency not in graph:
                    raise TeamTaskError("任务依赖必须属于同一团队")
                visit(dependency)
            visiting.remove(node)
            visited.add(node)
        for node in graph:
            visit(node)

    def _load(self, team_id: str) -> tuple[int, list[TeamTaskRecord], list[ClaimScanRound]]:
        """读取团队任务文件的 revision、任务和检查轮次。

        Args:
            team_id: 需要读取的团队 ID。

        Returns:
            当前 revision、任务列表和检查轮次列表。
        """

        raw = _read_json(self.store.team_dir(team_id) / "tasks.json")
        return (
            int(raw.get("revision", 0)),
            [task_from_json(item) for item in raw.get("tasks", [])],
            [_scan_from_json(item) for item in raw.get("scans", [])],
        )

    def _save(self, team_id: str, revision: int, tasks: list[TeamTaskRecord], scans: list[ClaimScanRound]) -> None:
        """原子写回一份完整任务快照。

        Args:
            team_id: 任务所属团队 ID。
            revision: 本次写入的新 revision。
            tasks: 完整任务列表。
            scans: 完整自主认领检查轮次列表。

        Returns:
            快照同步到磁盘后不返回数据。
        """

        _atomic_json(
            self.store.team_dir(team_id) / "tasks.json",
            {"revision": revision, "tasks": [task_to_json(item) for item in tasks], "scans": [_scan_to_json(item) for item in scans]},
        )

    def _lock(self, team_id: str, actor: str) -> ExclusiveFileLock:
        """构造一个标记实际写入者的团队任务文件锁。

        Args:
            team_id: 需要修改任务的团队 ID。
            actor: 写入者 ID，用于锁占用诊断。

        Returns:
            尚未获取的 ExclusiveFileLock。
        """

        return ExclusiveFileLock(self.store.team_dir(team_id) / "locks" / "tasks.lock", actor)

    @staticmethod
    def _index(tasks: list[TeamTaskRecord], task_id: str) -> int:
        """查找一个任务在当前锁内列表中的位置。

        Args:
            tasks: 当前团队完整任务列表。
            task_id: 需要查找的任务 ID。

        Returns:
            任务的零基索引。

        Raises:
            TeamTaskError: 任务不存在。
        """

        index = next((i for i, item in enumerate(tasks) if item.task_id == task_id), None)
        if index is None:
            raise TeamTaskError(f"任务不存在：{task_id}")
        return index

    def _require_lead(self, actor: TeamActorContext) -> None:
        """验证 Actor generation 有效且身份为 Lead。

        Args:
            actor: 本地运行时提供的团队身份。

        Returns:
            Actor 可以管理任务时不返回数据。
        """

        self.store.require_actor(actor)
        if actor.actor_kind != "lead":
            raise TeamTaskError("只有 Lead 能创建或管理团队任务")

    def _require_member(self, actor: TeamActorContext) -> None:
        """验证 Actor generation 有效且身份为成员。

        Args:
            actor: 本地运行时提供的团队身份。

        Returns:
            Actor 可以自主认领任务时不返回数据。
        """

        self.store.require_actor(actor)
        if actor.actor_kind != "member":
            raise TeamTaskError("只有团队成员能认领任务")

    def _append_event(self, event: TeamEvent) -> None:
        """把任务扫描或生命周期事件追加并同步到团队 JSONL。

        Args:
            event: 已构造的团队事件。

        Returns:
            事件写入磁盘后不返回数据。

        Raises:
            TeamTaskError: 事件文件无法追加或同步。
        """

        self.store.append_event(event)
