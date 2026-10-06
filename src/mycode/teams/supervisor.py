"""创建、唤醒、恢复和停止团队成员的运行后端与 Worktree。"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import logging
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime

from mycode.models.teams import (
    SpawnTeammateRequest,
    TeamActorContext,
    TeamTaskQuery,
    TeamTaskStatus,
    TeammateBackend,
    TeammateRecord,
    TeammateState,
    TeamWatchdogSettings,
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
from mycode.teams.mailbox import TeamMailbox
from mycode.teams.locks import ExclusiveFileLock, TeamLockError
from mycode.teams.watchdog import LeadHeartbeat
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
        launch_environment: Mapping[str, str] | None = None,
        idle_ttl_seconds: float = 1800.0,
        reaper_interval_seconds: float = 60.0,
        watchdog_settings: TeamWatchdogSettings = TeamWatchdogSettings(),
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
        # 父进程冻结的非敏感沙箱配置，首次启动和恢复成员都传递。
        self.launch_environment = dict(launch_environment or {})
        self._handles: dict[tuple[str, str], BackendHandle] = {}
        # 冻结实际启动的身份，旧 Supervisor 关闭时不得控制后来接管的成员。
        self._launches: dict[tuple[str, str], TeammateLaunch] = {}
        self._assignments = {}
        self.idle_ttl_seconds = idle_ttl_seconds
        self.reaper_interval_seconds = reaper_interval_seconds
        self._control_lock = asyncio.Lock()
        self._reaper_task: asyncio.Task | None = None
        self.watchdog_settings = watchdog_settings
        self._owners: dict[str, LeadHeartbeat] = {}

    @staticmethod
    def _owns_record(member: TeammateRecord, launch: TeammateLaunch) -> bool:
        """匹配实际启动身份，或本次空闲回收刚撤销的旧运行。"""
        retired = (member.runtime_generation == launch.generation + 1
            and member.state is TeammateState.SUSPENDED and member.lease_token_hash is None
            and member.last_stop_reason == "idle_reaped")
        return member.owner_instance_id == launch.owner_instance_id and (
            member.runtime_generation == launch.generation or retired)

    async def ensure_owner(self, actor: TeamActorContext) -> LeadHeartbeat:
        """在启动成员之前取得当前 Lead 代数的运行权并开始续租。"""
        owner = self._owners.get(actor.team_id)
        if owner is not None and owner.actor == actor:
            return owner
        if owner is not None:
            await owner.close("lead_replaced")
        owner = LeadHeartbeat(self.store, actor, self.watchdog_settings)
        owner.start()
        self._owners[actor.team_id] = owner
        return owner

    async def release_owners(self, reason="application_shutdown") -> None:
        """不再协调原团队时停止续租，让其成员自行停止。"""
        for team_id, owner in tuple(self._owners.items()):
            await owner.close(reason)
            self._owners.pop(team_id, None)

    async def _stop_old_host(self, member: TeammateRecord, *, reason: str, force=False) -> None:
        """确认旧 Host 已结束后才允许恢复新 generation，不按磁盘 PID 操作。"""
        key = (member.team_id, member.agent_id)
        handle = self._handles.get(key)
        launch = self._launches.get(key)
        if handle is not None and launch is not None and not self._owns_record(member, launch):
            if handle.backend is TeammateBackend.SUBPROCESS:
                await self.adapters[handle.backend].stop(handle, force=force)
            # 旧终端 pane ID 可能在 server 重启后复用；失去身份时不再按旧 ID 关闭。
            # 旧 Host 会通过看门狗自行退出。
            self._handles.pop(key, None)
            self._launches.pop(key, None)
            handle = None
        owned = handle is not None
        if handle is None and member.backend_ref is not None:
            handle = BackendHandle(member.backend, member.backend_ref, member.owner_pid)
        team = self.store.load_team(member.team_id).team
        actor = TeamActorContext(member.team_id, "lead", "lead", team.lead_generation)
        self.store.request_member_stop(actor, member.agent_id, member.runtime_generation, reason)
        lock_path = self.store.team_dir(member.team_id) / "locks" / f"member-runtime-{member.agent_id}.lock"
        if not force and member.owner_instance_id:
            if (not owned and not lock_path.exists() and member.state in {
                    TeammateState.STARTING, TeammateState.RUNNING, TeammateState.IDLE}):
                raise RuntimeError("缺少旧成员运行锁，无法确认会话已释放，未启动新 Host")
            deadline = asyncio.get_running_loop().time() + self.watchdog_settings.shutdown_grace + self.watchdog_settings.poll_interval + 1
            while lock_path.exists():
                guard = ExclusiveFileLock(lock_path, "await-member-exit", max_attempts=1)
                try:
                    guard.acquire()
                except TeamLockError:
                    if asyncio.get_running_loop().time() >= deadline:
                        raise RuntimeError("旧成员尚未完成清理，不能并行恢复")
                    await asyncio.sleep(0.05)
                else:
                    guard.release()
                    break
        elif not owned and member.backend is TeammateBackend.SUBPROCESS and member.state in {
                TeammateState.STARTING, TeammateState.RUNNING, TeammateState.IDLE}:
            raise RuntimeError("旧版 subprocess Host 不受当前进程管理，请先停止旧程序后恢复")
        if handle is not None:
            await self.adapters[member.backend].stop(handle, force=force)
            probe = await self.adapters[member.backend].probe(handle)
            if probe.alive:
                raise RuntimeError("旧成员后端仍存活，未启动新 Host")
        self._handles.pop(key, None)
        self._launches.pop(key, None)

    def start_reaper(self) -> None:
        """只回收当前 Supervisor 持有的 Host，不根据磁盘 PID 杀进程。"""
        if self.idle_ttl_seconds > 0 and (self._reaper_task is None or self._reaper_task.done()):
            self._reaper_task = asyncio.create_task(self._reaper_loop())

    async def close_reaper(self) -> None:
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            await asyncio.gather(self._reaper_task, return_exceptions=True)
            self._reaper_task = None

    async def _reaper_loop(self) -> None:
        while True:
            await asyncio.sleep(self.reaper_interval_seconds)
            try:
                await self.reap_idle()
            except Exception:
                logging.getLogger(__name__).exception("团队成员空闲回收失败，将在下轮重试")

    async def _clean_worktree(self, member: TeammateRecord) -> bool:
        """Git 检查失败、超时或目录不可用时保守跳过。"""
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                "git", "status", "--porcelain", "--untracked-files=all",
                cwd=member.worktree_path, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            async with asyncio.timeout(2.0):
                output, _ = await process.communicate()
            return process.returncode == 0 and not output.strip()
        except (OSError, TimeoutError):
            return False
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()

    async def reap_idle(self) -> tuple[str, ...]:
        """先检查 Git，再在任务、邮箱、成员锁内重新确认并撤销旧 Host 租约。"""
        if self.idle_ttl_seconds <= 0:
            return ()
        reaped = []
        async with self._control_lock:
            for (team_id, member_id), handle in tuple(self._handles.items()):
                snapshot = self.store.load_team(team_id)
                member = next(item for item in snapshot.members if item.agent_id == member_id)
                launch = self._launches.get((team_id, member_id))
                if (member.backend_ref != handle.reference or (
                        launch is not None and not self._owns_record(member, launch))):
                    continue
                if member.state is TeammateState.SUSPENDED and member.lease_token_hash is None:
                    # 上轮已经撤销租约，但后端停止失败；保留句柄并重试清理。
                    await self.adapters[member.backend].stop(handle, force=False)
                    self._handles.pop((team_id, member_id), None)
                    self._launches.pop((team_id, member_id), None)
                    lead = TeamActorContext(team_id, "lead", "lead", snapshot.team.lead_generation)
                    self.store.update_member(lead, member_id, lambda latest: replace(
                        latest, backend_ref=None, owner_pid=None) if
                        latest.runtime_generation == member.runtime_generation else latest)
                    continue
                if member.state is not TeammateState.IDLE:
                    continue
                if (_now() - (member.last_active_at or member.updated_at)).total_seconds() < self.idle_ttl_seconds:
                    continue
                if member.current_task_id is not None:
                    continue
                if not await self._clean_worktree(member):
                    continue
                directory = self.store.team_dir(team_id)
                lead = TeamActorContext(team_id, "lead", "lead", snapshot.team.lead_generation)
                actor = TeamActorContext(team_id, member_id, "member", member.runtime_generation)
                # 锁顺序与任务认领一致：tasks -> member；邮箱发送不持有成员锁。
                with ExclusiveFileLock(directory / "locks" / "tasks.lock", "idle-reaper"):
                    with ExclusiveFileLock(directory / "locks" / f"mailbox-{member_id}.lock", "idle-reaper"):
                        current = self.store.load_team(team_id)
                        if any(t.owner_id == member_id and t.status is TeamTaskStatus.WORKING for t in current.tasks):
                            continue
                        if TeamMailbox(self.store).read_unread(actor) or self.tasks.pending_scans(actor):
                            continue
                        def suspend(latest):
                            if (latest.state is not TeammateState.IDLE
                                    or latest.runtime_generation != member.runtime_generation
                                    or latest.current_task_id is not None
                                    or (_now() - (latest.last_active_at or latest.updated_at)).total_seconds() < self.idle_ttl_seconds):
                                return latest
                            return replace(latest, state=TeammateState.SUSPENDED,
                                           last_stop_reason="idle_reaped",
                                           runtime_generation=latest.runtime_generation + 1,
                                           lease_token_hash=None, updated_at=_now())
                        suspended = self.store.update_member(lead, member_id, suspend)
                        if suspended.state is not TeammateState.SUSPENDED:
                            continue
                # 不删除会话、Worktree、分支，也不释放 Worktree 的保留租约。
                await self.adapters[member.backend].stop(handle, force=False)
                self._handles.pop((team_id, member_id), None)
                self._launches.pop((team_id, member_id), None)
                self.store.update_member(lead, member_id, lambda latest: replace(
                    latest, backend_ref=None, owner_pid=None) if
                    latest.runtime_generation == suspended.runtime_generation else latest)
                reaped.append(member_id)
        return tuple(reaped)

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

        self.start_reaper()
        team = self.store.require_actor(actor)
        if actor.actor_kind != "lead":
            raise RuntimeError("只有 Lead 能创建团队成员")
        if request.team_name != team.name:
            raise RuntimeError("成员请求的团队名称与当前团队不一致")
        owner = await self.ensure_owner(actor)
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
            owner_lead_session_id=owner.lead_session_id,
            owner_lead_generation=actor.generation,
            owner_instance_id=owner.owner_instance_id,
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
            environment=self.launch_environment,
            owner_lead_session_id=owner.lead_session_id,
            owner_lead_generation=actor.generation,
            owner_instance_id=owner.owner_instance_id,
            watchdog=self.watchdog_settings,
        )
        handle = None
        try:
            handle = await adapter.start(launch)
            self._handles[(team.team_id, agent_id)] = handle
            self._launches[(team.team_id, agent_id)] = launch
            probe = await adapter.probe(handle)
            if not probe.alive:
                raise RuntimeError(f"成员 Host 启动后未存活：{probe.detail}")
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
            if handle is not None:
                try:
                    await self._stop_old_host(self.store.read_member(team.team_id, agent_id), reason="startup_failed", force=True)
                except Exception as cleanup_error:
                    self.store.update_member(actor, agent_id, lambda current: replace(current,
                        state=TeammateState.FAILED, last_stop_reason="cleanup_unconfirmed", updated_at=_now()))
                    raise RuntimeError("成员启动失败且后端清理未确认，已保留会话与 Worktree") from cleanup_error
            self._handles.pop((team.team_id, agent_id), None)
            self._launches.pop((team.team_id, agent_id), None)
            self.store.remove_partial_member(team.team_id, agent_id)
            await self.worktrees.finish_task(assignment, WorktreeTaskOutcome.CANCELLED)
            raise

    async def wake(self, team_id: str, member_id: str) -> bool:
        """唤醒一个 idle/suspended 成员；running 成员不重复触发。

        Args:
            team_id: 成员所属团队 ID。
            member_id: 要通知的成员 ID。

        Returns:
            已检查后端或恢复成员时返回 True；持久化通知仍是唤醒依据。
        """

        async with self._control_lock:
            snapshot = self.store.load_team(team_id)
            member = next(item for item in snapshot.members if item.agent_id == member_id)
            if member.state is TeammateState.SUSPENDED:
                # 回收与唤醒共用锁，先释放旧句柄，再轮换租约恢复原会话。
                lead = TeamActorContext(team_id, "lead", "lead", snapshot.team.lead_generation)
                await self._stop_old_host(member, reason="member_wake")
                await self._restart_member(lead, self.store.read_member(team_id, member_id))
                return True
            if member.state is not TeammateState.IDLE:
                return False
            lead = TeamActorContext(team_id, "lead", "lead", snapshot.team.lead_generation)
            self.store.update_member(lead, member_id, lambda current: replace(
                current, last_active_at=_now(), updated_at=_now()))
            await self.adapters[member.backend].wake(self._handle_for(member))
            return True

    async def wake_for_claimable_tasks(
        self,
        actor: TeamActorContext,
        task_ids: tuple[str, ...],
    ):
        """任务板已持久化认领通知；管道唤醒仅用于让空闲成员立即检查。"""
        for member in self.store.load_team(actor.team_id).members:
            try:
                await self.wake(actor.team_id, member.agent_id)
            except Exception:
                # 成员仍会从磁盘检查通知，正在运行的成员在回合结束后处理。
                continue

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
        await self._stop_old_host(member, reason="explicit_stop", force=force)
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

        self.start_reaper()
        snapshot = self.store.load_team(team_id)
        lead = TeamActorContext(
            team_id,
            "lead",
            "lead",
            snapshot.team.lead_generation,
        )
        owner = await self.ensure_owner(lead)
        reports: list[str] = []
        for member in snapshot.members:
            if member.state is TeammateState.TERMINATED:
                reports.append(f"{member.name}: 已终止，不自动恢复")
                continue
            handle = self._handles.get((team_id, member.agent_id))
            if handle is not None and member.owner_instance_id == owner.owner_instance_id:
                probe = await self.adapters[member.backend].probe(handle)
                if probe.alive and member.state in {TeammateState.RUNNING, TeammateState.IDLE}:
                    reports.append(f"{member.name}: 仍由当前 Lead 监管")
                    continue
            try:
                await self._stop_old_host(member, reason="lead_replaced")
                await self._restart_member(lead, self.store.read_member(team_id, member.agent_id))
            except Exception as exc:
                reports.append(f"{member.name}: 恢复失败：{exc}")
            else:
                reports.append(f"{member.name}: 已按 {member.backend.value} 恢复")
        return tuple(reports)

    async def close_local_hosts(self) -> None:
        """释放 Lead 租约并停止所有持有后端，保留会话和工作区。"""
        await self.close_reaper()
        await self.release_owners()
        errors = []
        for (team_id, member_id), handle in tuple(self._handles.items()):
            try:
                member = self.store.read_member(team_id, member_id)
                launch = self._launches.get((team_id, member_id))
                if launch is not None and not self._owns_record(member, launch):
                    if handle.backend is TeammateBackend.SUBPROCESS:
                        await self.adapters[handle.backend].stop(handle, force=False)
                    self._handles.pop((team_id, member_id), None)
                    self._launches.pop((team_id, member_id), None)
                    continue  # 只清理自己的旧句柄，不发布新成员停止意图或改写状态。
                await self._stop_old_host(member, reason="application_shutdown")
                team = self.store.load_team(team_id).team
                actor = TeamActorContext(team_id, "lead", "lead", team.lead_generation)
                self.store.update_member(actor, member_id, lambda current: replace(current,
                    state=TeammateState.SUSPENDED, backend_ref=None, owner_pid=None,
                    lease_token_hash=None, last_stop_reason="application_shutdown", updated_at=_now())
                    if current.runtime_generation == member.runtime_generation and current.state is not TeammateState.TERMINATED else current)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("部分成员停止未确认：" + "; ".join(errors))

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
        """等待 Host 完成会话恢复，进入执行或空闲等待状态。

        Args:
            team_id: 新成员所属团队 ID。
            member_id: 正在启动的成员 ID。
            adapter: 已经选定且不得降级的后端控制器。
            handle: ``adapter.start`` 返回的真实后端句柄。

        Returns:
            Host 已完成初始化并写为 ``running`` 或 ``idle`` 的最新成员记录。

        Raises:
            RuntimeError: Host 报告失败、提前退出，或十秒内没有完成握手。
        """

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.watchdog_settings.startup_grace + _HOST_HANDSHAKE_TIMEOUT_SECONDS
        while loop.time() < deadline:
            member = next(
                item
                for item in self.store.load_team(team_id).members
                if item.agent_id == member_id
            )
            if member.state in {TeammateState.RUNNING, TeammateState.IDLE}:
                return member
            if member.state is TeammateState.FAILED:
                raise RuntimeError("成员 Host 恢复会话时失败")
            probe = await adapter.probe(handle)
            if not probe.alive:
                raise RuntimeError(f"成员 Host 在握手前退出：{probe.detail}")
            await asyncio.sleep(_HOST_HANDSHAKE_POLL_SECONDS)
        raise RuntimeError("成员 Host 启动心跳或会话恢复握手超时")

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
        owner = await self.ensure_owner(actor)
        lease = secrets.token_urlsafe(24)
        generation = member.runtime_generation + 1
        self.store.update_member(
            actor,
            member.agent_id,
            lambda current: replace(
                current,
                state=TeammateState.STARTING,
                runtime_generation=generation,
                owner_lead_session_id=owner.lead_session_id,
                owner_lead_generation=actor.generation,
                owner_instance_id=owner.owner_instance_id,
                stop_request_generation=None,
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
            environment=self.launch_environment,
            owner_lead_session_id=owner.lead_session_id,
            owner_lead_generation=actor.generation,
            owner_instance_id=owner.owner_instance_id,
            watchdog=self.watchdog_settings,
        )
        handle = None
        try:
            handle = await adapter.start(launch)
            self._handles[(member.team_id, member.agent_id)] = handle
            self._launches[(member.team_id, member.agent_id)] = launch
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
            if handle is not None:
                try:
                    await self._stop_old_host(self.store.read_member(member.team_id, member.agent_id), reason="startup_failed")
                except Exception:
                    pass  # 保留句柄供应用退出或下次清理使用，不猜测进程已停止。
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
