"""创建、唤醒、恢复和停止团队成员的运行后端与 Worktree。"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime

from mycode.models.teams import (
    SpawnTeammateRequest,
    TeamActorContext,
    TeamTaskQuery,
    TeammateBackend,
    TeammateRecord,
    TeammateState,
)
from mycode.models.worktrees import WorktreeTaskOutcome
from mycode.persistence.sessions import SessionManager
from mycode.teams.backends.base import (
    BackendHandle,
    TeammateBackendAdapter,
    TeammateLaunch,
)
from mycode.teams.backends.detection import BackendDetector
from mycode.teams.store import TeamStateStore
from mycode.teams.tasks import TeamTaskBoard
from mycode.worktrees.manager import WorktreeManager


MemberSessionCreator = Callable[[str], str]
_HOST_HANDSHAKE_TIMEOUT_SECONDS = 10.0
_HOST_HANDSHAKE_POLL_SECONDS = 0.05


class TeammateSupervisor:
    """协调成员身份、独立 Worktree、后端句柄和任务扫描。

    Attributes:
        workspace_root: 主仓库绝对路径，用于构造后端启动数据。
        store: 团队和成员记录持久化入口。
        tasks: 共享任务看板。
        worktrees: 创建和清理成员独立目录的 WorktreeManager。
        detector: 创建成员前只执行一次的后端检测器。
        adapters: 三种后端枚举到真实控制器的映射。
        session_creator: 在团队 sessions 目录创建成员会话并返回 ID 的函数。
    """

    def __init__(
        self,
        *,
        workspace_root,
        store: TeamStateStore,
        tasks: TeamTaskBoard,
        worktrees: WorktreeManager,
        detector: BackendDetector,
        adapters: Mapping[TeammateBackend, TeammateBackendAdapter],
        session_creator: MemberSessionCreator,
    ) -> None:
        """保存创建和控制成员所需的生产组件。

        Args:
            workspace_root: 当前主仓库绝对路径。
            store: 团队身份和成员状态 Store。
            tasks: 任务查询、扫描和状态更新入口。
            worktrees: 已启动的 WorktreeManager。
            detector: 固定优先级后端检测器。
            adapters: 每个可选后端对应的真实 adapter。
            session_creator: 传入 team ID 后创建成员会话并返回 session ID。

        Returns:
            不返回数据；成员在 ``spawn`` 时才创建。
        """

        self.workspace_root = workspace_root.resolve(strict=True)
        self.store = store
        self.tasks = tasks
        self.worktrees = worktrees
        self.detector = detector
        self.adapters = dict(adapters)
        self.session_creator = session_creator
        self._handles: dict[tuple[str, str], BackendHandle] = {}
        self._assignments = {}

    async def spawn(
        self,
        actor: TeamActorContext,
        request: SpawnTeammateRequest,
    ) -> TeammateRecord:
        """一次性选定后端，再创建 Worktree、成员记录和 Host。

        Args:
            actor: 当前有效 Lead 身份。
            request: 成员名称、角色、首次提示和后端偏好。

        Returns:
            后端启动且探测存活后的成员记录。

        Raises:
            RuntimeError: 调用者不是 Lead、后端不可用、启动失败或握手失败。
                选定后端失败时不会改用其他后端。
        """

        team = self.store.require_actor(actor)
        if actor.actor_kind != "lead":
            raise RuntimeError("只有 Lead 能创建团队成员")
        if request.team_name != team.name:
            raise RuntimeError("成员请求的团队名称与当前团队不一致")
        selected = self.detector.select(request.backend)
        adapter = self.adapters.get(selected)
        if adapter is None:
            raise RuntimeError(f"后端没有完成装配：{selected.value}")
        agent_id = f"agent-{secrets.token_hex(6)}"
        assignment = await self.worktrees.create_for_team_member(
            team_id=team.team_id,
            agent_id=agent_id,
            lead_session_id=team.lead_session_id,
        )
        session_id = self.session_creator(team.team_id)
        lease = secrets.token_urlsafe(24)
        now = _now()
        member = TeammateRecord(
            agent_id=agent_id,
            team_id=team.team_id,
            name=request.name.strip(),
            role_name=request.role_name.strip(),
            model_override=request.model_override,
            session_id=session_id,
            worktree_name=assignment.worktree_name or "",
            worktree_path=assignment.root,
            branch=assignment.branch or "",
            backend=selected,
            backend_ref=None,
            state=TeammateState.STARTING,
            runtime_generation=1,
            owner_pid=None,
            lease_token_hash=hashlib.sha256(lease.encode()).hexdigest(),
            plan_mode_required=request.plan_mode_required,
            current_task_id=None,
            created_at=now,
            updated_at=now,
        )
        self.store.add_member(member)
        self.store.save_runtime_prompt(team.team_id, agent_id, request.prompt)
        launch = TeammateLaunch(
            workspace_root=self.workspace_root,
            worktree_path=assignment.root,
            team_id=team.team_id,
            agent_id=agent_id,
            generation=1,
            lease_token=lease,
            prompt=request.prompt,
        )
        try:
            handle = await adapter.start(launch)
            probe = await adapter.probe(handle)
            if not probe.alive:
                raise RuntimeError(f"成员 Host 启动后未存活：{probe.detail}")
            self._handles[(team.team_id, agent_id)] = handle
            self._assignments[(team.team_id, agent_id)] = assignment
            self.store.update_member(
                actor,
                agent_id,
                lambda current: replace(
                    current,
                    backend_ref=handle.reference,
                    owner_pid=handle.process_id,
                    updated_at=_now(),
                ),
            )
            return await self._await_host_handshake(
                team.team_id,
                agent_id,
                adapter,
                handle,
            )
        except Exception:
            handle = self._handles.pop((team.team_id, agent_id), None)
            if handle is not None:
                try:
                    await adapter.stop(handle, force=True)
                except Exception:
                    pass
            self.store.remove_partial_member(team.team_id, agent_id)
            await self.worktrees.finish_task(assignment, WorktreeTaskOutcome.CANCELLED)
            raise

    async def wake(self, team_id: str, member_id: str) -> bool:
        """唤醒一个 idle/suspended 成员；running 成员不重复触发。

        Args:
            team_id: 成员所属团队 ID。
            member_id: 要通知的成员 ID。

        Returns:
            实际调用后端 wake 时返回 True；成员忙碌或已结束返回 False。
        """

        member = next(
            item for item in self.store.load_team(team_id).members if item.agent_id == member_id
        )
        if member.state not in {TeammateState.IDLE, TeammateState.SUSPENDED}:
            return False
        handle = self._handle_for(member)
        await self.adapters[member.backend].wake(handle)
        return True

    async def wake_for_claimable_tasks(
        self,
        actor: TeamActorContext,
        task_ids: tuple[str, ...],
    ):
        """唤醒全部空闲成员，并只用成功者创建检查轮次。

        Args:
            actor: 创建任务的当前 Lead 身份。
            task_ids: 本次新开放、需要成员查看的任务 ID。

        Returns:
            ``TeamTaskBoard.open_scan`` 创建的检查轮次。
        """

        successful: list[str] = []
        for member in self.store.load_team(actor.team_id).members:
            try:
                if await self.wake(actor.team_id, member.agent_id):
                    successful.append(member.agent_id)
            except Exception:
                continue
        return self.tasks.open_scan(actor.team_id, task_ids, tuple(successful))

    async def stop(
        self,
        actor: TeamActorContext,
        member_id: str,
        *,
        force: bool,
    ) -> TeammateRecord:
        """停止成员后端，并把终态写为 terminated。

        Args:
            actor: 当前有效 Lead 身份。
            member_id: 要停止的团队成员 ID。
            force: True 时允许 adapter 强制结束进程或 task。

        Returns:
            已持久化 terminated 状态的成员记录。
        """

        self.store.require_actor(actor)
        if actor.actor_kind != "lead":
            raise RuntimeError("只有 Lead 能停止成员")
        member = next(
            item for item in self.store.load_team(actor.team_id).members if item.agent_id == member_id
        )
        await self.adapters[member.backend].stop(self._handle_for(member), force=force)
        await self.worktrees.release_team_member_lease(member.worktree_name)
        return self.store.update_member(
            actor,
            member_id,
            lambda current: replace(
                current, state=TeammateState.TERMINATED, updated_at=_now()
            ),
        )

    async def restore(self, team_id: str) -> tuple[str, ...]:
        """探测已登记成员，并按原后端恢复已经停止的 Host。

        Args:
            team_id: 原 Lead 会话恢复后重新连接的团队 ID。

        Returns:
            每个成员一条用户可读恢复结果。恢复时不重新检测后端，也不创建
            新会话或 Worktree。
        """

        snapshot = self.store.load_team(team_id)
        lead = TeamActorContext(
            team_id,
            "lead",
            "lead",
            snapshot.team.lead_generation,
        )
        reports: list[str] = []
        for member in snapshot.members:
            if member.state is TeammateState.TERMINATED:
                reports.append(f"{member.name}: 已终止，不自动恢复")
                continue
            if member.backend_ref is None:
                alive = False
            else:
                handle = self._handle_for(member)
                try:
                    probe = await self.adapters[member.backend].probe(handle)
                    alive = probe.alive
                except Exception:
                    alive = False
            if alive:
                reports.append(f"{member.name}: 仍在运行")
                continue
            try:
                await self._restart_member(lead, member)
            except Exception as exc:
                reports.append(f"{member.name}: 恢复失败：{exc}")
            else:
                reports.append(f"{member.name}: 已按 {member.backend.value} 恢复")
        return tuple(reports)

    async def close_local_hosts(self) -> None:
        """关闭当前进程持有的 in-process Host，并保留全部团队磁盘状态。

        Returns:
            所有本地 Host 已取消并标为 ``suspended`` 后返回。tmux 和 iTerm2
            Host 属于独立进程，不会在主程序退出时停止。
        """

        for (team_id, member_id), handle in tuple(self._handles.items()):
            if handle.backend is not TeammateBackend.IN_PROCESS:
                continue
            await self.adapters[handle.backend].stop(handle, force=False)
            self._handles.pop((team_id, member_id), None)
            try:
                snapshot = self.store.load_team(team_id)
                actor = TeamActorContext(
                    team_id,
                    "lead",
                    "lead",
                    snapshot.team.lead_generation,
                )
                self.store.update_member(
                    actor,
                    member_id,
                    lambda current: replace(
                        current,
                        state=TeammateState.SUSPENDED,
                        backend_ref=None,
                        owner_pid=None,
                        updated_at=_now(),
                    ),
                )
            except Exception:
                # 应用关闭仍需继续释放其他资源。成员磁盘记录未删除，下一次
                # restore 会根据后端句柄探测结果再次尝试恢复。
                continue

    def _handle_for(self, member: TeammateRecord) -> BackendHandle:
        """取得内存句柄或从持久化 backend_ref 重建控制句柄。

        Args:
            member: Store 中读取到的当前成员记录。

        Returns:
            adapter 可以用于 probe、wake 和 stop 的 ``BackendHandle``。
        """

        existing = self._handles.get((member.team_id, member.agent_id))
        if existing is not None:
            return existing
        if member.backend_ref is None:
            raise RuntimeError("成员尚无可控制的后端引用")
        return BackendHandle(member.backend, member.backend_ref, member.owner_pid)

    async def _await_host_handshake(
        self,
        team_id: str,
        member_id: str,
        adapter: TeammateBackendAdapter,
        handle: BackendHandle,
    ) -> TeammateRecord:
        """等待 Host 自己完成会话恢复并写入 ``running`` 状态。

        Args:
            team_id: 新成员所属团队 ID。
            member_id: 正在启动的成员 ID。
            adapter: 已经选定且不得降级的后端控制器。
            handle: ``adapter.start`` 返回的真实后端句柄。

        Returns:
            Host 已完成初始化并写为 ``running`` 的最新成员记录。

        Raises:
            RuntimeError: Host 报告失败、提前退出，或十秒内没有完成握手。
        """

        loop = asyncio.get_running_loop()
        deadline = loop.time() + _HOST_HANDSHAKE_TIMEOUT_SECONDS
        while loop.time() < deadline:
            member = next(
                item
                for item in self.store.load_team(team_id).members
                if item.agent_id == member_id
            )
            if member.state is TeammateState.RUNNING:
                return member
            if member.state is TeammateState.FAILED:
                raise RuntimeError("成员 Host 恢复会话时失败")
            probe = await adapter.probe(handle)
            if not probe.alive:
                raise RuntimeError(f"成员 Host 在握手前退出：{probe.detail}")
            await asyncio.sleep(_HOST_HANDSHAKE_POLL_SECONDS)
        raise RuntimeError("成员 Host 启动握手超时（10 秒）")

    async def _restart_member(
        self,
        actor: TeamActorContext,
        member: TeammateRecord,
    ) -> TeammateRecord:
        """为已有成员轮换租约，并在原后端上恢复持久化会话。

        Args:
            actor: 当前有效 Lead 身份。
            member: 需要恢复的现有花名册记录。

        Returns:
            新 Host 完成握手后的成员记录。

        Raises:
            RuntimeError: 原后端未装配、启动失败或握手失败。
        """

        adapter = self.adapters.get(member.backend)
        if adapter is None:
            raise RuntimeError(f"原后端没有完成装配：{member.backend.value}")
        lease = secrets.token_urlsafe(24)
        generation = member.runtime_generation + 1
        self.store.update_member(
            actor,
            member.agent_id,
            lambda current: replace(
                current,
                state=TeammateState.STARTING,
                runtime_generation=generation,
                lease_token_hash=hashlib.sha256(lease.encode()).hexdigest(),
                backend_ref=None,
                owner_pid=None,
                updated_at=_now(),
            ),
        )
        launch = TeammateLaunch(
            workspace_root=self.workspace_root,
            worktree_path=member.worktree_path,
            team_id=member.team_id,
            agent_id=member.agent_id,
            generation=generation,
            lease_token=lease,
            prompt=self.store.load_runtime_prompt(
                member.team_id,
                member.agent_id,
            ),
        )
        try:
            handle = await adapter.start(launch)
            self._handles[(member.team_id, member.agent_id)] = handle
            self.store.update_member(
                actor,
                member.agent_id,
                lambda current: replace(
                    current,
                    backend_ref=handle.reference,
                    owner_pid=handle.process_id,
                    updated_at=_now(),
                ),
            )
            return await self._await_host_handshake(
                member.team_id,
                member.agent_id,
                adapter,
                handle,
            )
        except Exception:
            self._handles.pop((member.team_id, member.agent_id), None)
            self.store.update_member(
                actor,
                member.agent_id,
                lambda current: replace(
                    current,
                    state=TeammateState.FAILED,
                    updated_at=_now(),
                ),
            )
            raise


def _now() -> datetime:
    """返回成员状态持久化使用的带时区当前时间。

    Returns:
        当前本地时区的 ``datetime``。
    """

    return datetime.now().astimezone()
