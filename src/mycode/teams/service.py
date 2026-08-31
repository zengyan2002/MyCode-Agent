"""提供团队创建、查询、恢复、接管、成员控制和全量删除用例。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from mycode.models.sessions import SessionRuntimeMetadata
from mycode.models.teams import (
    SpawnTeammateRequest,
    TeamActorContext,
    TeamBinding,
    TeamCreateRequest,
    TeamDeletionReport,
    TeamLifecycle,
    TeamSnapshot,
    TeamTaskCreateRequest,
    TeamTaskRecord,
    TeamTaskStatus,
    TeamTaskUpdateRequest,
    TeammateRecord,
    TeammateState,
)
from mycode.persistence.sessions import SessionManager
from mycode.teams.integration import TeamIntegrationService
from mycode.teams.store import TeamStateStore, TeamStoreError
from mycode.teams.supervisor import TeammateSupervisor
from mycode.teams.tasks import TeamTaskBoard
from mycode.worktrees.manager import WorktreeManager


TakeoverConfirmation = Callable[[str], Awaitable[bool]]


class TeamService:
    """把团队 Store、Supervisor、任务板和 Git 清理组合成用户用例。

    Attributes:
        store: 团队身份、成员和生命周期的持久化入口。
        tasks: 共享任务板。
        supervisor: 成员后端和 Worktree 的控制器。
        integration: 提交、合并、验证和删除前置检查服务。
        sessions: 当前 Lead 主会话的 SessionManager。
        worktrees: 删除阶段实际移除成员目录和临时分支的管理器。
        confirm_takeover: 接管孤立团队前向终端用户确认的异步函数。
        actor_setter: 在团队身份变化后更新主 ToolContext 的本地回调。
    """

    def __init__(
        self,
        *,
        store: TeamStateStore,
        tasks: TeamTaskBoard,
        supervisor: TeammateSupervisor,
        integration: TeamIntegrationService,
        sessions: SessionManager,
        worktrees: WorktreeManager,
        confirm_takeover: TakeoverConfirmation,
        actor_setter: Callable[[TeamActorContext | None], None] | None = None,
    ) -> None:
        """保存所有团队用例需要的生产组件。

        Args:
            store: 当前工作区唯一的 TeamStateStore。
            tasks: 与 Store 共用 tasks.json 的任务板。
            supervisor: 创建、唤醒、恢复和停止成员的服务。
            integration: 读取 Git 事实并保存验证状态的服务。
            sessions: 当前 Lead 会话持久化管理器。
            worktrees: 已启动的 WorktreeManager。
            confirm_takeover: 显示接管影响并返回用户是否批准的函数。
            actor_setter: 团队身份变化后更新主 ToolContext 的本地回调。

        Returns:
            不返回数据。
        """

        self.store = store
        self.tasks = tasks
        self.supervisor = supervisor
        self.integration = integration
        self.sessions = sessions
        self.worktrees = worktrees
        self.confirm_takeover = confirm_takeover
        self.actor_setter = actor_setter or (lambda actor: None)

    def create(self, request: TeamCreateRequest) -> TeamSnapshot:
        """创建团队并把当前主会话绑定为第一代 Lead。

        Args:
            request: 团队名称和本次用户目标说明。

        Returns:
            刚创建、成员与任务都为空的完整团队快照。
        """

        team = self.store.create_team(
            request.team_name,
            request.description,
            self.sessions.current_id,
        )
        self.sessions.save_team_binding(TeamBinding(team.team_id, team.lead_generation))
        self.actor_setter(
            TeamActorContext(team.team_id, "lead", "lead", team.lead_generation)
        )
        return self.store.load_team(team.team_id)

    def get(self, actor: TeamActorContext) -> TeamSnapshot:
        """读取当前团队、花名册、任务和合并验证状态。

        Args:
            actor: 当前本地 Lead 或成员身份。

        Returns:
            Store 在调用时刻读取的 ``TeamSnapshot``。
        """

        try:
            self.store.require_actor(actor)
        except TeamStoreError:
            # cleanup_failed 团队已冻结普通写入，但当前 Lead 仍需要通过
            # TeamGet 查看失败原因和已删除资源，才能决定是否继续 TeamDelete。
            self.store.require_cleanup_actor(actor)
        return self.store.load_team(actor.team_id)

    async def spawn_member(
        self,
        actor: TeamActorContext,
        request: SpawnTeammateRequest,
    ) -> TeammateRecord:
        """把 Agent 工具中的团队成员请求交给 Supervisor。

        Args:
            actor: 当前有效 Lead 身份。
            request: 成员角色、名称、首次提示和后端选项。

        Returns:
            Host 已启动并登记 backend_ref 的成员记录。
        """

        return await self.supervisor.spawn(actor, request)

    async def create_task(
        self,
        actor: TeamActorContext,
        request: TeamTaskCreateRequest,
    ) -> TeamTaskRecord:
        """创建共享任务，并让全部空闲成员立即进行一次自主认领检查。

        Args:
            actor: 当前有效 Lead 身份。
            request: 任务说明、类型、优先级和直接依赖。

        Returns:
            已写入共享任务板的新任务。唤醒失败不会撤销任务；检查轮次只会
            等待实际唤醒成功的成员。
        """

        task = self.tasks.create(actor, request)
        if request.task_kind == "code":
            self.integration.register_validation_repair(actor, task.task_id)
        await self.supervisor.wake_for_claimable_tasks(actor, (task.task_id,))
        return task

    async def update_task(
        self,
        actor: TeamActorContext,
        request: TeamTaskUpdateRequest,
    ) -> TeamTaskRecord:
        """更新共享任务，并在完成前检查真实提交或调查结果。

        Args:
            actor: 当前有效 Lead 或任务负责人身份。
            request: 只包含显式更新字段的任务变更请求。

        Returns:
            已通过状态机并写入共享任务文件的新任务记录。

        Raises:
            TeamIntegrationError: ``completed`` 代码任务的 Worktree、分支或
                提交不满足完成条件。
            TeamTaskError: 状态转换或字段权限不合法。
        """

        if request.status is TeamTaskStatus.COMPLETED:
            await self.integration.validate_task_completion(
                actor,
                request.task_id,
                commit_hashes=request.commit_hashes,
                result=request.result,
            )
        updated = self.tasks.update(actor, request)
        if updated.status is TeamTaskStatus.COMPLETED:
            self.integration.resolve_validation_repair(actor, updated.task_id)
        return updated

    async def member_stop(
        self,
        actor: TeamActorContext,
        member_id: str,
        *,
        force: bool,
    ) -> TeammateRecord:
        """停止一个长期团队成员，不触碰现有后台 TaskManager 任务。

        Args:
            actor: 当前有效 Lead 身份。
            member_id: 需要结束 Host 的成员 ID。
            force: 是否跳过优雅退出并强制停止后端。

        Returns:
            状态已变为 terminated 的成员记录。
        """

        return await self.supervisor.stop(actor, member_id, force=force)

    async def restore_for_lead(self) -> tuple[TeamActorContext | None, tuple[str, ...]]:
        """按当前会话 TeamBinding 恢复原 Lead，而不是自动接管别人的团队。

        Returns:
            没有绑定时返回 ``(None, ())``；有效时返回 Lead Actor 和逐成员
            恢复报告。失效 binding 会抛错，磁盘数据保持不变。
        """

        metadata, warning = self.sessions.read_runtime_metadata(self.sessions.current_id)
        if warning is not None:
            raise RuntimeError(f"无法恢复团队绑定：{warning}")
        binding = metadata.team
        if binding is None:
            self.actor_setter(None)
            return None, ()
        team = self.store.load_team(binding.team_id).team
        if (
            team.lead_session_id != self.sessions.current_id
            or team.lead_generation != binding.lead_generation
        ):
            raise RuntimeError("当前会话的 TeamBinding 已失效，需要显式接管")
        actor = TeamActorContext(
            team.team_id, "lead", "lead", team.lead_generation
        )
        self.actor_setter(actor)
        if team.lifecycle is not TeamLifecycle.ACTIVE:
            return actor, (
                f"团队处于 {team.lifecycle.value} 状态，只允许查询或继续 TeamDelete",
            )
        return actor, await self.supervisor.restore(team.team_id)

    async def takeover(self, team_id: str) -> TeamActorContext:
        """经终端用户批准后把团队交给当前主会话并递增 generation。

        Args:
            team_id: 需要接管的存续团队 ID。

        Returns:
            新一代 Lead 的可信 ActorContext。

        Raises:
            RuntimeError: 用户拒绝接管。
        """

        team = self.store.load_team(team_id).team
        approved = await self.confirm_takeover(
            f"接管团队 {team.name!r} 会立即使旧 Lead 的写权限失效，是否继续？"
        )
        if not approved:
            raise RuntimeError("用户未批准团队接管")
        updated = self.store.takeover(team_id, self.sessions.current_id)
        self.sessions.save_team_binding(
            TeamBinding(updated.team_id, updated.lead_generation)
        )
        await self.supervisor.restore(team_id)
        actor = TeamActorContext(
            updated.team_id, "lead", "lead", updated.lead_generation
        )
        self.actor_setter(actor)
        return actor

    async def delete(self, actor: TeamActorContext) -> TeamDeletionReport:
        """先完成无副作用预检，再删除团队全部运行资源。

        Args:
            actor: 当前有效 Lead 身份。

        Returns:
            被阻塞时返回全部原因且不删除任何资源；成功时返回实际删除的
            Worktree、分支、会话 binding 和团队目录列表。
        """

        team = self.store.require_cleanup_actor(actor)
        if actor.actor_kind != "lead":
            raise RuntimeError("只有 Lead 能删除团队")
        if team.lifecycle is TeamLifecycle.ACTIVE:
            preflight = await self.integration.deletion_preflight(actor.team_id)
            if not preflight.allowed:
                return preflight
            for member in self.store.load_team(actor.team_id).members:
                if member.state is TeammateState.IDLE:
                    try:
                        await self.supervisor.stop(
                            actor,
                            member.agent_id,
                            force=False,
                        )
                    except Exception as exc:
                        return TeamDeletionReport(
                            actor.team_id,
                            False,
                            (f"成员 {member.name} 未能停止：{exc}",),
                        )
            snapshot = self.store.begin_cleanup(actor)
        else:
            snapshot = self.store.load_team(actor.team_id)
        removed = list(self.store.load_cleanup_progress(actor.team_id))
        try:
            for member in snapshot.members:
                worktree_key = f"worktree:{member.worktree_name}"
                if worktree_key not in removed:
                    await self.worktrees.release_team_member_lease(member.worktree_name)
                    await self.worktrees.remove(member.worktree_name, discard_changes=False)
                    removed.append(worktree_key)
                    self.store.save_cleanup_progress(actor.team_id, tuple(removed))
                branch_key = f"branch:{member.branch}"
                if branch_key not in removed:
                    await self.worktrees.delete_branch(
                        member.worktree_name,
                        discard_commits=False,
                    )
                    removed.append(branch_key)
                    self.store.save_cleanup_progress(actor.team_id, tuple(removed))
            binding_key = "lead-session-binding"
            if binding_key not in removed:
                self.sessions.save_team_binding(None)
                removed.append(binding_key)
                self.store.save_cleanup_progress(actor.team_id, tuple(removed))
            self.store.finish_cleanup(actor.team_id)
            self.actor_setter(None)
            removed.append("team-directory")
            return TeamDeletionReport(actor.team_id, True, (), tuple(removed))
        except Exception as exc:
            if "lead-session-binding" in removed:
                try:
                    self.sessions.save_team_binding(
                        TeamBinding(actor.team_id, actor.generation)
                    )
                    removed.remove("lead-session-binding")
                except Exception:
                    pass
            # 清理失败报告必须优先返回原始错误。即使故障恰好发生在本地磁盘
            # 已不可写的时刻，也不能让记录 cleanup_failed 的次生错误覆盖它。
            try:
                self.store.mark_cleanup_failed(actor.team_id, str(exc))
                self.store.save_cleanup_progress(actor.team_id, tuple(removed))
            except Exception:
                pass
            return TeamDeletionReport(
                actor.team_id,
                False,
                (f"清理中断：{exc}",),
                tuple(removed),
            )

    async def close_local_hosts(self) -> None:
        """关闭随主程序事件循环运行的成员，不删除团队或成员会话。

        Returns:
            所有 in-process Host 已暂停后返回。独立终端后端保持运行。
        """

        await self.supervisor.close_local_hosts()
