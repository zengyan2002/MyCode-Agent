"""管理 Worktree 的创建、租约、会话绑定、收尾和显式删除。"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from datetime import timedelta
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

from mycode.models.worktrees import (
    CommitRelation,
    InterruptedTaskSummary,
    WorkspaceAssignment,
    WorkspaceIsolationMode,
    WorkspaceResolution,
    WorktreeChangeSummary,
    WorktreeFinishAction,
    WorktreeFinishReport,
    WorktreeKind,
    WorktreeLifecycle,
    WorktreeRecord,
    WorktreeRecoveryReport,
    WorktreeRemoveReport,
    WorktreeSnapshot,
    WorktreeStateSnapshot,
    WorktreeTaskOutcome,
    WorktreeTaskOwner,
    WorktreeTaskState,
)
from mycode.worktrees.binding import WorkspaceBinding
from mycode.worktrees.git import GitWorktreeBackend, WorktreeGitError
from mycode.worktrees.initializer import WorktreeInitializer
from mycode.worktrees.names import validate_worktree_slug
from mycode.worktrees.state import WorktreeStateError, WorktreeStateStore


_T = TypeVar("_T")


class WorktreeManagerError(RuntimeError):
    """说明一个 Worktree 生命周期操作无法按已批准的保护规则完成。"""


class WorktreeManager:
    """协调一个主仓库内所有受管 Worktree 的生命周期。

    Attributes:
        repo_root: 主仓库绝对路径，受管目录固定在其 ``.mycode/worktrees`` 下。
        binding: 主 Agent 当前使用的可变工作区绑定。
        git: 执行本地非交互 Git 操作的仓库后端。
        initializer: 对新目录复制配置、链接依赖和设置 Hooks 的初始化器。
        state_store: 原子读写 ``.mycode/worktree_state.json`` 的存储。
        _index_lock: 只保护内存记录、会话映射、租约和活动会话的短锁。
        _state_write_lock: 串行化 revision 分配和小型状态文件保存。
        _operation_locks: 每个 Worktree 名称独享的操作锁；不同名称互不等待。
        _leases: 当前进程中 ``lease_id -> (worktree_name, task_id)`` 的运行占用。

    Agent 模型运行期间不持有任何锁。全局锁不会覆盖 Git、文件复制或模型调用；
    同名创建和删除只在该名称的操作锁中串行。
    """

    def __init__(
        self,
        repo_root: Path,
        binding: WorkspaceBinding,
        git: GitWorktreeBackend,
        initializer: WorktreeInitializer,
        state_store: WorktreeStateStore,
    ) -> None:
        """装配一个仓库专用 Worktree Manager。

        Args:
            repo_root: 主仓库绝对路径。
            binding: 主 Agent 的可变工作区绑定。
            git: 指向同一仓库的 Git 后端。
            initializer: 指向同一仓库的创建后初始化器。
            state_store: 指向同一仓库的状态存储。

        Returns:
            尚未加载磁盘状态的 Manager；使用前必须调用 :meth:`start`。

        Raises:
            ValueError: 依赖类型无效、绑定固定，或任一依赖指向其他仓库。
        """

        if not isinstance(repo_root, Path) or not repo_root.is_absolute():
            raise ValueError("WorktreeManager.repo_root 必须是绝对 Path")
        if not isinstance(binding, WorkspaceBinding) or binding.is_fixed:
            raise ValueError("WorktreeManager.binding 必须是可变 WorkspaceBinding")
        if not isinstance(git, GitWorktreeBackend):
            raise ValueError("WorktreeManager.git 类型无效")
        if not isinstance(initializer, WorktreeInitializer):
            raise ValueError("WorktreeManager.initializer 类型无效")
        if not isinstance(state_store, WorktreeStateStore):
            raise ValueError("WorktreeManager.state_store 类型无效")
        resolved = repo_root.resolve()
        if any(
            root != resolved
            for root in (git.repo_root, initializer.repo_root, state_store.repo_root)
        ):
            raise ValueError("WorktreeManager 的所有依赖必须指向同一个主仓库")
        self.repo_root = resolved
        self.binding = binding
        self.git = git
        self.initializer = initializer
        self.state_store = state_store
        self._index_lock = asyncio.Lock()
        self._state_write_lock = asyncio.Lock()
        self._operation_locks: dict[str, asyncio.Lock] = {}
        self._records: dict[str, WorktreeRecord] = {}
        self._session_bindings: dict[str, str] = {}
        self._leases: dict[str, tuple[str | None, str | None]] = {}
        self._revision = 0
        self._active_session_id: str | None = None
        self._started = False
        self._state_trusted = True

    async def start(
        self,
        *,
        resumed_session_id: str | None = None,
    ) -> WorktreeRecoveryReport:
        """加载磁盘状态，并把上次进程遗留的运行任务标为中断。

        Args:
            resumed_session_id: 本次启动准备恢复的主会话 ID；未恢复时为 ``None``。

        Returns:
            状态可信度、导入的 interrupted 任务和恢复警告。

        Raises:
            ValueError: 会话 ID 是空字符串。
            RuntimeError: 同一个 Manager 重复启动。
            WorktreeStateError: 标记 interrupted 后无法保存可信状态。
        """

        if resumed_session_id is not None and not resumed_session_id.strip():
            raise ValueError("resumed_session_id 不能是空字符串")
        if self._started:
            raise RuntimeError("WorktreeManager 已经启动")
        loaded = await asyncio.to_thread(self.state_store.load)
        self._started = True
        self._state_trusted = loaded.trusted
        if not loaded.trusted or loaded.snapshot is None:
            return WorktreeRecoveryReport(
                state_trusted=False,
                warnings=(loaded.error or "Worktree 状态不可信",),
            )
        snapshot = loaded.snapshot
        async with self._index_lock:
            self._records = {record.name: record for record in snapshot.records}
            self._session_bindings = dict(snapshot.session_bindings)
            self._revision = snapshot.revision
            self._active_session_id = resumed_session_id

        interrupted: list[InterruptedTaskSummary] = []
        recovery_warnings: list[str] = []
        replacements: dict[str, WorktreeRecord] = {}
        for record in snapshot.records:
            if record.task_state not in {
                WorktreeTaskState.QUEUED,
                WorktreeTaskState.RUNNING,
            }:
                continue
            if record.owner is None:
                continue
            owner_status = self._owner_process_status(record.owner_pid)
            if owner_status is not False:
                state = "仍在运行" if owner_status is True else "无法确认是否结束"
                recovery_warnings.append(
                    f"Worktree {record.name} 的原进程 {record.owner_pid} {state}，"
                    "未接管任务，也未执行清理"
                )
                continue
            updated = replace(
                record,
                lifecycle=WorktreeLifecycle.INTERRUPTED,
                task_state=WorktreeTaskState.INTERRUPTED,
                owner_pid=None,
                last_used_at=datetime.now(UTC),
            )
            replacements[record.name] = updated
            interrupted.append(
                InterruptedTaskSummary(
                    task_id=record.owner.task_id,
                    session_id=record.owner.session_id,
                    worktree_name=record.name,
                    path=record.path,
                    branch=record.branch,
                    base_commit=record.base_commit,
                    reason="上次 MyCode 进程在子任务完成前退出",
                )
            )
        if replacements:
            await self._write_state(
                lambda: self._records.update(replacements)
            )
        return WorktreeRecoveryReport(
            state_trusted=True,
            interrupted_tasks=tuple(interrupted),
            warnings=tuple(recovery_warnings),
        )

    async def close(self) -> None:
        """结束 Manager 的进程内生命周期。

        Returns:
            清空活动会话和运行租约后不返回数据。磁盘记录和会话映射保留，供
            下次 ``--resume`` 使用。
        """

        async with self._index_lock:
            self._active_session_id = None
            self._leases.clear()
        self._started = False

    async def create_manual(self, name: str, session_id: str) -> WorktreeRecord:
        """创建一个不会被周期清理的手工 Worktree。

        Args:
            name: 用户提供并经过严格校验的 slug。
            session_id: 发起创建的主会话 ID，用于诊断但不自动进入目录。

        Returns:
            初始化完成并处于 ``READY`` 的受管记录。

        Raises:
            ValueError: 名称或会话 ID 无效。
            WorktreeManagerError: 名称冲突、状态不可信、Git 或初始化失败。
        """

        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("create_manual session_id 必须是非空字符串")
        record, _ = await self._create(
            name,
            kind=WorktreeKind.MANUAL,
            owner=None,
        )
        return record

    async def create_for_task(
        self,
        task: WorktreeTaskOwner,
    ) -> WorkspaceAssignment:
        """为一项独立子 Agent 任务创建 Worktree 并签发运行租约。

        Args:
            task: 主会话、可选 TaskManager ID 和来源组成的任务归属。

        Returns:
            包含绝对路径、分支、创建基线和唯一 ``lease_id`` 的固定分配。

        Raises:
            ValueError: ``task`` 类型无效。
            WorktreeManagerError: 名称冲突、Git 创建或初始化失败。
        """

        if not isinstance(task, WorktreeTaskOwner):
            raise ValueError("create_for_task task 类型无效")
        seed = task.task_id or f"{task.session_id}:{task.origin}"
        safe = "".join(character for character in seed if character.isalnum())[:12]
        if not safe:
            safe = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
        name = f"agent-{safe.lower()}"
        record, _ = await self._create(
            name,
            kind=WorktreeKind.TASK,
            owner=task,
        )
        lease_id = uuid4().hex
        async with self._index_lock:
            self._leases[lease_id] = (record.name, task.task_id)
        return WorkspaceAssignment(
            root=record.path,
            isolation=WorkspaceIsolationMode.WORKTREE,
            worktree_name=record.name,
            branch=record.branch,
            base_commit=record.base_commit,
            lease_id=lease_id,
            parent_had_changes=("parent-had-changes" in record.warnings),
        )

    async def create_for_team_member(
        self,
        *,
        team_id: str,
        agent_id: str,
        lead_session_id: str,
    ) -> WorkspaceAssignment:
        """为长期团队成员创建带稳定 team/member owner 的 Worktree。

        Args:
            team_id: 成员所属团队的不可变 ID。
            agent_id: 新成员的不可变 ID。
            lead_session_id: 创建成员的 Lead 会话 ID，用于恢复和诊断。

        Returns:
            含独立目录、分支、基线和运行租约的 WorkspaceAssignment。
        """

        return await self.create_for_task(
            WorktreeTaskOwner(
                session_id=lead_session_id,
                task_id=agent_id,
                origin="team",
                team_id=team_id,
                agent_id=agent_id,
            )
        )

    async def lease_current_for_task(
        self,
        task: WorktreeTaskOwner,
    ) -> WorkspaceAssignment:
        """冻结当前主 Agent 工作区，并在受管目录上增加任务租约。

        Args:
            task: 准备共用当前目录的子任务归属。

        Returns:
            主仓库返回共享分配；当前处于受管 Worktree 时返回带新任务租约的
            独立分配。两种情况都重新读取当前本地 HEAD。

        Raises:
            ValueError: ``task`` 类型无效。
            WorktreeManagerError: 当前受管记录缺失或 Git 无法读取 HEAD。
        """

        self._ensure_started()
        if not isinstance(task, WorktreeTaskOwner):
            raise ValueError("lease_current_for_task task 类型无效")
        current = self.binding.snapshot()
        try:
            head = await asyncio.to_thread(
                self.git.resolve_local_head,
                cwd=current.root,
            )
        except WorktreeGitError as exc:
            raise WorktreeManagerError(str(exc)) from exc
        if current.worktree_name is None:
            return WorkspaceAssignment(
                root=current.root,
                isolation=WorkspaceIsolationMode.SHARED,
                worktree_name=None,
                branch=head.branch,
                base_commit=head.commit,
            )
        async with self._index_lock:
            if current.worktree_name not in self._records:
                raise WorktreeManagerError("当前 Worktree 不在受管状态中")
            lease_id = uuid4().hex
            self._leases[lease_id] = (current.worktree_name, task.task_id)
        return WorkspaceAssignment(
            root=current.root,
            isolation=WorkspaceIsolationMode.WORKTREE,
            worktree_name=current.worktree_name,
            branch=head.branch,
            base_commit=head.commit,
            lease_id=lease_id,
        )

    async def mark_task_running(self, assignment: WorkspaceAssignment) -> None:
        """把临时 Worktree 的持久化任务状态从 queued 改为 running。

        Args:
            assignment: ``create_for_task`` 返回、仍持有有效租约的分配。

        Returns:
            共享目录不需要持久化时直接返回；临时记录成功更新后不返回数据。

        Raises:
            ValueError: 分配类型无效。
            WorktreeManagerError: 租约失效或记录不存在。
        """

        if not isinstance(assignment, WorkspaceAssignment):
            raise ValueError("mark_task_running assignment 类型无效")
        if assignment.worktree_name is None:
            return
        await self._require_lease(assignment)

        def update() -> None:
            """把当前任务 Worktree 标为正在运行。

            Returns:
                状态更新完成后不返回数据。

            Raises:
                WorktreeManagerError: 分配引用的 Worktree 记录不存在。
            """

            record = self._records.get(assignment.worktree_name or "")
            if record is None:
                raise WorktreeManagerError("任务 Worktree 记录不存在")
            if record.kind is WorktreeKind.TASK:
                self._records[record.name] = replace(
                    record,
                    task_state=WorktreeTaskState.RUNNING,
                    last_used_at=datetime.now(UTC),
                )

        await self._write_state(update)

    async def release_team_member_lease(self, name: str) -> None:
        """释放长期团队成员的运行租约，同时保留它的目录和分支。

        Args:
            name: 成员记录中的受管 Worktree 名称。

        Returns:
            所有指向该 Worktree 的内存租约都已移除时不返回数据；原本没有
            租约也视为完成。

        Raises:
            WorktreeManagerError: Manager 尚未启动，或该名称不是团队成员目录。
        """

        self._ensure_started()

        def clear_owner_process() -> None:
            """清空停止 Host 留下的进程占用标记。

            Returns:
                状态更新完成后不返回数据。

            Raises:
                WorktreeManagerError: 指定名称不是受管团队成员 Worktree。
            """

            record = self._records.get(name)
            if (
                record is None
                or record.owner is None
                or record.owner.origin != "team"
            ):
                raise WorktreeManagerError(f"团队成员 Worktree 不存在：{name}")
            self._records[name] = replace(
                record,
                owner_pid=None,
                last_used_at=datetime.now(UTC),
            )

        # owner_pid 是跨进程删除门禁，必须先持久化清空。进程若在下一步
        # 移除内存租约前退出，重启后租约本就不会恢复，续清理仍然安全。
        await self._write_state(clear_owner_process)
        async with self._index_lock:
            stale = [
                lease_id
                for lease_id, (lease_name, _) in self._leases.items()
                if lease_name == name
            ]
            for lease_id in stale:
                self._leases.pop(lease_id, None)

    async def finish_task(
        self,
        assignment: WorkspaceAssignment,
        status: WorktreeTaskOutcome,
    ) -> WorktreeFinishReport:
        """释放任务租约，并删除干净同基线目录或保留有成果的目录。

        Args:
            assignment: 子 Agent 运行前冻结的工作区分配。
            status: 子 Agent 完成、失败、取消或中断终态。

        Returns:
            包含 Git 变更摘要、实际处置动作和保留原因的收尾报告。

        Raises:
            ValueError: 参数类型无效。
            WorktreeManagerError: 分配的租约已失效或受管记录不存在。
        """

        if not isinstance(assignment, WorkspaceAssignment):
            raise ValueError("finish_task assignment 类型无效")
        if not isinstance(status, WorktreeTaskOutcome):
            raise ValueError("finish_task status 类型无效")
        if assignment.worktree_name is None:
            await self._release_lease(assignment.lease_id)
            return WorktreeFinishReport(
                workspace=assignment,
                action=WorktreeFinishAction.SHARED_RELEASED,
                terminal_status=status,
                changes=None,
                reason="共享工作区没有临时目录需要清理",
            )
        await self._require_lease(assignment)
        operation_lock = await self._operation_lock(assignment.worktree_name)
        async with operation_lock:
            async with self._index_lock:
                record = self._records.get(assignment.worktree_name)
            if record is None:
                await self._release_lease(assignment.lease_id)
                raise WorktreeManagerError("任务 Worktree 记录不存在")
            if record.kind is not WorktreeKind.TASK:
                await self._release_lease(assignment.lease_id)
                return WorktreeFinishReport(
                    workspace=assignment,
                    action=WorktreeFinishAction.SHARED_RELEASED,
                    terminal_status=status,
                    changes=None,
                    reason="子任务共享了手工 Worktree，只释放任务租约",
                )
            try:
                changes = await asyncio.to_thread(self.git.inspect_changes, record)
            except (WorktreeGitError, OSError) as exc:
                changes = None
                reason = f"无法可靠检查 Worktree 变更，已保留：{exc}"
                await self._retain_task(record, assignment.lease_id, interrupted=status is WorktreeTaskOutcome.INTERRUPTED)
                return WorktreeFinishReport(
                    workspace=assignment,
                    action=WorktreeFinishAction.RETAINED,
                    terminal_status=status,
                    changes=None,
                    reason=reason,
                )
            if not self._can_delete_automatically(changes):
                reason = "Worktree 包含文件变化、新提交或无法确认的提交状态"
                await self._retain_task(record, assignment.lease_id, interrupted=status is WorktreeTaskOutcome.INTERRUPTED)
                return WorktreeFinishReport(
                    workspace=assignment,
                    action=WorktreeFinishAction.RETAINED,
                    terminal_status=status,
                    changes=changes,
                    reason=reason,
                )
            await self._set_lifecycle(record.name, WorktreeLifecycle.REMOVING)
            try:
                await asyncio.to_thread(self.git.remove, record.path)
                await asyncio.to_thread(self.git.delete_branch, record.branch)
            except WorktreeGitError as exc:
                await self._set_lifecycle(record.name, WorktreeLifecycle.ERROR)
                await self._release_lease(assignment.lease_id)
                return WorktreeFinishReport(
                    workspace=assignment,
                    action=WorktreeFinishAction.RETAINED,
                    terminal_status=status,
                    changes=changes,
                    reason=f"清理 Git Worktree 失败，已保留状态记录：{exc}",
                )
            await self._remove_record(record.name)
            await self._release_lease(assignment.lease_id)
            return WorktreeFinishReport(
                workspace=assignment,
                action=WorktreeFinishAction.DELETED,
                terminal_status=status,
                changes=changes,
                reason="临时 Worktree 干净且 HEAD 等于创建基线",
            )

    async def bind_session(
        self,
        session_id: str,
        name: str,
    ) -> WorkspaceAssignment:
        """验证手工 Worktree，并预写主会话到该目录的持久化映射。

        Args:
            session_id: 当前主会话 ID。
            name: 等待进入的受管 Worktree slug。

        Returns:
            可交给 :meth:`activate_session` 原子切换主绑定的分配。

        Raises:
            ValueError: 会话或名称为空。
            WorktreeManagerError: 记录不可用、Git 验证失败或状态不可信。
        """

        self._ensure_trusted()
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("bind_session session_id 必须是非空字符串")
        validate_worktree_slug(name, self.repo_root)
        operation_lock = await self._operation_lock(name)
        async with operation_lock:
            async with self._index_lock:
                record = self._records.get(name)
            if record is None or record.lifecycle not in {
                WorktreeLifecycle.READY,
                WorktreeLifecycle.RETAINED,
                WorktreeLifecycle.INTERRUPTED,
            }:
                raise WorktreeManagerError(f"Worktree 不可进入：{name}")
            try:
                entry = await asyncio.to_thread(self.git.validate_existing, record)
            except WorktreeGitError as exc:
                raise WorktreeManagerError(str(exc)) from exc

            def update() -> None:
                """登记会话绑定并刷新复用记录的最后使用时间。

                Returns:
                    内存索引更新完成后不返回数据。
                """

                self._session_bindings[session_id] = name
                self._records[name] = replace(
                    record,
                    last_used_at=datetime.now(UTC),
                )

            await self._write_state(update)
            return WorkspaceAssignment(
                root=record.path,
                isolation=WorkspaceIsolationMode.WORKTREE,
                worktree_name=record.name,
                branch=entry.branch,
                base_commit=entry.head_commit,
                lease_id=f"session:{session_id}",
            )

    async def resolve_session_binding(self, session_id: str) -> WorkspaceResolution:
        """解析恢复会话最后绑定的 Worktree，失败时明确回退主仓库。

        Args:
            session_id: 正在执行 ``--resume`` 的主会话 ID。

        Returns:
            可用受管目录或主仓库分配，以及恢复降级警告。

        Raises:
            ValueError: 会话 ID 为空。
        """

        self._ensure_started()
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("resolve_session_binding session_id 必须是非空字符串")
        if not self._state_trusted:
            return WorkspaceResolution(
                assignment=await self._main_assignment(),
                warnings=("Worktree 状态不可信，会话已回退主仓库",),
            )
        async with self._index_lock:
            name = self._session_bindings.get(session_id)
            record = self._records.get(name) if name is not None else None
        if name is None:
            return WorkspaceResolution(assignment=await self._main_assignment())
        if record is None or record.lifecycle is WorktreeLifecycle.PRUNED:
            return WorkspaceResolution(
                assignment=await self._main_assignment(),
                warnings=(f"会话绑定的 Worktree {name} 已不存在，已回退主仓库",),
            )
        try:
            entry = await asyncio.to_thread(self.git.validate_existing, record)
        except WorktreeGitError as exc:
            return WorkspaceResolution(
                assignment=await self._main_assignment(),
                warnings=(f"Worktree {name} 恢复验证失败，已回退主仓库：{exc}",),
            )
        return WorkspaceResolution(
            assignment=WorkspaceAssignment(
                root=record.path,
                isolation=WorkspaceIsolationMode.WORKTREE,
                worktree_name=record.name,
                branch=entry.branch,
                base_commit=entry.head_commit,
                lease_id=f"session:{session_id}",
            )
        )

    async def activate_session(
        self,
        session_id: str,
        assignment: WorkspaceAssignment,
    ) -> None:
        """在上下文重载成功后，把主 Agent 绑定原子切换到预览分配。

        Args:
            session_id: 当前活动主会话 ID。
            assignment: ``bind_session`` 或 ``resolve_session_binding`` 返回的分配。

        Returns:
            活动会话和主绑定切换完成后不返回数据。

        Raises:
            ValueError: 参数类型无效。
            WorktreeManagerError: 独立分配与持久化会话映射不一致。
        """

        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("activate_session session_id 必须是非空字符串")
        if not isinstance(assignment, WorkspaceAssignment):
            raise ValueError("activate_session assignment 类型无效")
        if assignment.worktree_name is not None:
            async with self._index_lock:
                if self._session_bindings.get(session_id) != assignment.worktree_name:
                    raise WorktreeManagerError("会话映射已变化，拒绝切换到过期预览")
                self._active_session_id = session_id
        else:
            async with self._index_lock:
                self._active_session_id = session_id
        self.binding.bind(assignment)

    async def exit_session(self, session_id: str) -> WorkspaceAssignment:
        """清除主会话 Worktree 映射并把主绑定恢复到主仓库。

        Args:
            session_id: 当前活动主会话 ID。

        Returns:
            恢复后指向主仓库本地 HEAD 的共享工作区分配。

        Raises:
            ValueError: 会话 ID 为空。
            WorktreeManagerError: 状态不可信或 Git 无法读取主仓库 HEAD。
        """

        self._ensure_trusted()
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("exit_session session_id 必须是非空字符串")

        def update() -> None:
            """移除退出会话的绑定和当前会话标记。

            Returns:
                内存索引更新完成后不返回数据。
            """

            self._session_bindings.pop(session_id, None)
            if self._active_session_id == session_id:
                self._active_session_id = None

        await self._write_state(update)
        assignment = await self._main_assignment()
        self.binding.bind(assignment)
        return assignment

    async def list(self) -> tuple[WorktreeSnapshot, ...]:
        """读取所有受管 Worktree 的只读快照。

        Returns:
            按名称排序的记录、租约占用和会话绑定视图。
        """

        self._ensure_started()
        async with self._index_lock:
            leased_names = {
                name for name, _ in self._leases.values() if name is not None
            }
            return tuple(
                WorktreeSnapshot(
                    record=record,
                    leased=name in leased_names,
                    session_ids=tuple(
                        session_id
                        for session_id, bound_name in self._session_bindings.items()
                        if bound_name == name
                    ),
                )
                for name, record in sorted(self._records.items())
            )

    async def status(self, name: str) -> WorktreeSnapshot:
        """读取一个受管 Worktree 的当前记录和占用情况。

        Args:
            name: 要查询的原始 slug。

        Returns:
            与 :meth:`list` 相同格式的单条快照。

        Raises:
            WorktreeManagerError: 名称不存在。
        """

        for snapshot in await self.list():
            if snapshot.record.name == name:
                return snapshot
        raise WorktreeManagerError(f"Worktree 不存在：{name}")

    async def inspect_changes(self, name: str) -> WorktreeChangeSummary:
        """读取一个仍有目录的受管 Worktree 的文件和提交变化。

        Args:
            name: 要检查的原始 Worktree slug。

        Returns:
            Git 后端基于创建基线、当前 HEAD 和 upstream 生成的变更摘要。

        Raises:
            WorktreeManagerError: 记录不存在、目录已移除，或 Git 无法可靠检查。
        """

        snapshot = await self.status(name)
        record = snapshot.record
        if record.lifecycle is WorktreeLifecycle.PRUNED:
            raise WorktreeManagerError("Worktree 目录已经移除，无法检查文件变更")
        try:
            return await asyncio.to_thread(self.git.inspect_changes, record)
        except WorktreeGitError as exc:
            raise WorktreeManagerError(str(exc)) from exc

    async def branch_merged_status(self, name: str) -> bool | None:
        """判断一个已 prune 的受管分支是否已经合入创建基准。

        Args:
            name: 要检查的原始 Worktree slug。

        Returns:
            已合入返回 ``True``，未合入返回 ``False``，Git 无法确认时返回
            ``None``。

        Raises:
            WorktreeManagerError: 记录不存在或目录尚未先移除。
        """

        snapshot = await self.status(name)
        record = snapshot.record
        if record.lifecycle is not WorktreeLifecycle.PRUNED:
            raise WorktreeManagerError("必须先移除 Worktree 目录，再检查分支")
        try:
            return await asyncio.to_thread(
                self.git.is_branch_merged,
                record.branch,
                record.base_ref,
            )
        except WorktreeGitError:
            return None

    async def remove(
        self,
        name: str,
        *,
        discard_changes: bool,
    ) -> WorktreeRemoveReport:
        """移除 Worktree 目录，但保留分支和 ``PRUNED`` 状态记录。

        Args:
            name: 要移除的受管 Worktree slug。
            discard_changes: 是否明确允许丢弃未提交文件和未保存提交。

        Returns:
            目录、分支和记录最终状态的移除报告。

        Raises:
            ValueError: ``discard_changes`` 不是布尔值。
            WorktreeManagerError: 状态不可信、目录在使用、变更需要确认或 Git
                移除失败。
        """

        self._ensure_trusted()
        if not isinstance(discard_changes, bool):
            raise ValueError("discard_changes 必须是布尔值")
        operation_lock = await self._operation_lock(name)
        async with operation_lock:
            async with self._index_lock:
                record = self._records.get(name)
                active_bound = (
                    self._active_session_id is not None
                    and self._session_bindings.get(self._active_session_id) == name
                )
                leased = any(lease_name == name for lease_name, _ in self._leases.values())
            if record is None:
                raise WorktreeManagerError(f"Worktree 不存在：{name}")
            if record.lifecycle is WorktreeLifecycle.PRUNED:
                return WorktreeRemoveReport(
                    name=name,
                    directory_removed=True,
                    branch_removed=False,
                    lifecycle=WorktreeLifecycle.PRUNED,
                    message="Worktree 目录已经移除，分支仍保留",
                )
            if active_bound:
                raise WorktreeManagerError("当前活动会话仍在该 Worktree 中，请先退出")
            if leased:
                raise WorktreeManagerError("仍有子 Agent 使用该 Worktree，不能删除")
            if record.owner_pid is not None:
                owner_status = self._owner_process_status(record.owner_pid)
                if owner_status is not False:
                    raise WorktreeManagerError(
                        "Worktree 可能仍被其他进程使用，拒绝删除"
                    )
            try:
                changes = await asyncio.to_thread(self.git.inspect_changes, record)
            except WorktreeGitError as exc:
                if not discard_changes:
                    raise WorktreeManagerError(
                        f"无法确认 Worktree 是否有成果；需显式确认丢弃：{exc}"
                    ) from exc
                changes = None
            if changes is not None and not self._can_delete_automatically(changes) and not discard_changes:
                raise WorktreeManagerError(
                    "Worktree 有未提交修改、新提交或未推送提交；需显式确认丢弃"
                )
            await self._set_lifecycle(name, WorktreeLifecycle.REMOVING)
            try:
                await asyncio.to_thread(
                    self.git.remove,
                    record.path,
                    force=discard_changes,
                )
            except WorktreeGitError as exc:
                await self._set_lifecycle(name, WorktreeLifecycle.ERROR)
                raise WorktreeManagerError(str(exc)) from exc

            def mark_pruned() -> None:
                """把已移除目录的记录标为 pruned 并清除会话绑定。

                Returns:
                    内存索引更新完成后不返回数据。
                """

                current = self._records[name]
                self._records[name] = replace(
                    current,
                    lifecycle=WorktreeLifecycle.PRUNED,
                    task_state=(
                        WorktreeTaskState.FINISHED
                        if current.kind is WorktreeKind.TASK
                        else current.task_state
                    ),
                    owner_pid=None,
                    last_used_at=datetime.now(UTC),
                )
                stale_sessions = [
                    session_id
                    for session_id, bound_name in self._session_bindings.items()
                    if bound_name == name
                ]
                for session_id in stale_sessions:
                    del self._session_bindings[session_id]

            await self._write_state(mark_pruned)
            return WorktreeRemoveReport(
                name=name,
                directory_removed=True,
                branch_removed=False,
                lifecycle=WorktreeLifecycle.PRUNED,
                message="Worktree 目录已移除，分支保留待单独处理",
            )

    async def delete_branch(
        self,
        name: str,
        *,
        discard_commits: bool,
    ) -> WorktreeRemoveReport:
        """删除 ``PRUNED`` 记录保留的本地分支，并移除最终状态记录。

        Args:
            name: 目录已移除的受管 Worktree slug。
            discard_commits: 是否明确允许用 ``git branch -D`` 丢弃未合入提交。

        Returns:
            分支和记录已删除的报告。

        Raises:
            ValueError: ``discard_commits`` 不是布尔值。
            WorktreeManagerError: 状态不可信、目录尚在、分支未合入或 Git 失败。
        """

        self._ensure_trusted()
        if not isinstance(discard_commits, bool):
            raise ValueError("discard_commits 必须是布尔值")
        operation_lock = await self._operation_lock(name)
        async with operation_lock:
            async with self._index_lock:
                record = self._records.get(name)
            if record is None:
                raise WorktreeManagerError(f"Worktree 不存在：{name}")
            if record.lifecycle is not WorktreeLifecycle.PRUNED:
                raise WorktreeManagerError("必须先移除 Worktree 目录，再单独删除分支")
            merged = await asyncio.to_thread(
                self.git.is_branch_merged,
                record.branch,
                record.base_ref,
            )
            if merged is not True and not discard_commits:
                raise WorktreeManagerError("保留分支未确认合入基准；需显式确认丢弃提交")
            try:
                await asyncio.to_thread(
                    self.git.delete_branch,
                    record.branch,
                    force=discard_commits,
                )
            except WorktreeGitError as exc:
                raise WorktreeManagerError(str(exc)) from exc
            await self._remove_record(name)
            return WorktreeRemoveReport(
                name=name,
                directory_removed=True,
                branch_removed=True,
                lifecycle=None,
                message="Worktree 保留分支和状态记录已删除",
            )

    async def cleanup_stale(
        self,
        now: datetime,
        *,
        stale_after_hours: float,
    ) -> "CleanupReport":
        """清理过期且能确认没有成果的临时 Worktree 目录。

        Args:
            now: 本次扫描使用的带时区当前时间；所有候选共用这一时间点。
            stale_after_hours: ``last_used_at`` 超过多少小时才进入变更检查。

        Returns:
            检查过、已 prune、跳过和失败名称组成的 ``CleanupReport``。清理只
            删除目录，保留分支并把记录改为 ``PRUNED``。

        Raises:
            ValueError: ``now`` 没有时区，或过期小时数不是正数。
            WorktreeManagerError: 状态文件不可信。
        """

        from mycode.models.worktrees import CleanupReport

        self._ensure_trusted()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("cleanup_stale now 必须是带时区的 datetime")
        if (
            isinstance(stale_after_hours, bool)
            or not isinstance(stale_after_hours, (int, float))
            or stale_after_hours <= 0
        ):
            raise ValueError("cleanup_stale stale_after_hours 必须是正数")
        cutoff = now - timedelta(hours=float(stale_after_hours))
        snapshots = await self.list()
        checked: list[str] = []
        pruned: list[str] = []
        skipped: list[tuple[str, str]] = []
        errors: list[tuple[str, str]] = []
        for snapshot in snapshots:
            record = snapshot.record
            if record.kind is not WorktreeKind.TASK:
                skipped.append((record.name, "不是临时任务 Worktree"))
                continue
            if record.lifecycle in {
                WorktreeLifecycle.PRUNED,
                WorktreeLifecycle.REMOVING,
                WorktreeLifecycle.CREATING,
            }:
                skipped.append((record.name, f"生命周期为 {record.lifecycle.value}"))
                continue
            if record.last_used_at > cutoff:
                skipped.append((record.name, "尚未过期"))
                continue
            if snapshot.leased:
                skipped.append((record.name, "仍有子任务租约"))
                continue
            if record.owner_pid is not None:
                owner_status = self._owner_process_status(record.owner_pid)
                if owner_status is not False:
                    skipped.append((record.name, "可能仍被其他进程使用"))
                    continue
            if (
                self._active_session_id is not None
                and self._active_session_id in snapshot.session_ids
            ):
                skipped.append((record.name, "当前活动会话仍在使用"))
                continue
            checked.append(record.name)
            try:
                changes = await asyncio.to_thread(self.git.inspect_changes, record)
            except (WorktreeGitError, OSError) as exc:
                skipped.append((record.name, f"无法可靠检查变更：{exc}"))
                continue
            if not self._can_delete_automatically(changes):
                skipped.append((record.name, "包含变更、提交或无法确认的 Git 状态"))
                continue
            try:
                await self.remove(record.name, discard_changes=False)
            except (WorktreeManagerError, WorktreeStateError) as exc:
                errors.append((record.name, str(exc)))
                continue
            pruned.append(record.name)
        return CleanupReport(
            checked=tuple(checked),
            pruned=tuple(pruned),
            skipped=tuple(skipped),
            errors=tuple(errors),
        )

    async def _create(
        self,
        name: str,
        *,
        kind: WorktreeKind,
        owner: WorktreeTaskOwner | None,
    ) -> tuple[WorktreeRecord, bool]:
        """执行手工和任务 Worktree 共用的名称预留、Git 创建和初始化。

        Args:
            name: 原始 Worktree slug。
            kind: 手工或临时任务种类。
            owner: 临时任务归属；手工创建为 ``None``。

        Returns:
            ``(ready_record, reused)``；第二项说明是否复用了已存在目录。

        Raises:
            WorktreeManagerError: 状态不可信、名称已登记、Git 或初始化失败。
        """

        self._ensure_trusted()
        safe_name = validate_worktree_slug(name, self.repo_root)
        try:
            await asyncio.to_thread(self.git.validate_branch_name, safe_name.branch)
        except WorktreeGitError as exc:
            raise WorktreeManagerError(str(exc)) from exc
        operation_lock = await self._operation_lock(safe_name.original)
        async with operation_lock:
            source = self.binding.snapshot()
            try:
                head = await asyncio.to_thread(
                    self.git.resolve_local_head,
                    cwd=source.root,
                )
                dirty = await asyncio.to_thread(
                    self.git.parent_has_changes,
                    cwd=source.root,
                )
            except WorktreeGitError as exc:
                raise WorktreeManagerError(str(exc)) from exc
            now = datetime.now(UTC)
            warnings = ("parent-had-changes",) if dirty else ()
            record = WorktreeRecord(
                name=safe_name.original,
                path=safe_name.path,
                branch=safe_name.branch,
                base_ref=head.branch or head.commit,
                base_commit=head.commit,
                kind=kind,
                lifecycle=WorktreeLifecycle.CREATING,
                owner=owner,
                owner_pid=os.getpid() if owner is not None else None,
                task_state=(
                    WorktreeTaskState.QUEUED if owner is not None else None
                ),
                created_at=now,
                last_used_at=now,
                initialization_complete=False,
                warnings=warnings,
            )

            def reserve() -> None:
                """在创建目录前预留 Worktree 名称和初始记录。

                Returns:
                    名称预留完成后不返回数据。

                Raises:
                    WorktreeManagerError: 同名记录已经存在。
                """

                if safe_name.original in self._records:
                    raise WorktreeManagerError(
                        f"Worktree 已存在：{safe_name.original}"
                    )
                self._records[safe_name.original] = record

            await self._write_state(reserve)
            created_by_us = False
            reused = False
            try:
                if record.path.exists():
                    precheck = await asyncio.to_thread(self.state_store.precheck, record)
                    if precheck.matched is False:
                        raise WorktreeManagerError(precheck.reason)
                    await asyncio.to_thread(self.git.validate_existing, record)
                    reused = True
                    initialization = None
                else:
                    await asyncio.to_thread(
                        self.git.add,
                        record.path,
                        record.branch,
                        record.base_commit,
                    )
                    created_by_us = True
                    initialization = await asyncio.to_thread(
                        self.initializer.initialize,
                        record,
                    )
                    if not initialization.complete:
                        failed = next(
                            (
                                action
                                for action in initialization.actions
                                if action.status.value == "failed"
                            ),
                            None,
                        )
                        reason = failed.message if failed is not None else "初始化未完成"
                        raise WorktreeManagerError(reason)
                ready = replace(
                    record,
                    lifecycle=WorktreeLifecycle.READY,
                    initialization_complete=True,
                    warnings=(
                        record.warnings
                        if initialization is None
                        else record.warnings + initialization.warnings
                    ),
                    last_used_at=datetime.now(UTC),
                )
                await self._write_state(
                    lambda: self._records.__setitem__(ready.name, ready)
                )
                return ready, reused
            except Exception as exc:
                rolled_back = False
                if created_by_us:
                    try:
                        await asyncio.to_thread(self.git.remove, record.path)
                        await asyncio.to_thread(self.git.delete_branch, record.branch)
                        rolled_back = True
                    except (WorktreeGitError, OSError):
                        rolled_back = False
                if rolled_back:
                    await self._remove_record(record.name)
                else:
                    await self._set_lifecycle(record.name, WorktreeLifecycle.ERROR)
                if isinstance(exc, WorktreeManagerError):
                    raise
                raise WorktreeManagerError(str(exc)) from exc

    async def _main_assignment(self) -> WorkspaceAssignment:
        """读取主仓库当前本地 HEAD 并构造共享分配。

        Returns:
            根目录为 ``repo_root`` 的新 ``WorkspaceAssignment``。

        Raises:
            WorktreeManagerError: Git 无法读取主仓库 HEAD。
        """

        try:
            head = await asyncio.to_thread(self.git.resolve_local_head)
        except WorktreeGitError as exc:
            raise WorktreeManagerError(str(exc)) from exc
        return WorkspaceAssignment(
            root=self.repo_root,
            isolation=WorkspaceIsolationMode.SHARED,
            worktree_name=None,
            branch=head.branch,
            base_commit=head.commit,
        )

    async def _retain_task(
        self,
        record: WorktreeRecord,
        lease_id: str | None,
        *,
        interrupted: bool,
    ) -> None:
        """把有成果或无法确认的任务目录标为保留并释放租约。

        Args:
            record: 当前任务 Worktree 记录。
            lease_id: 等待释放的任务租约 ID。
            interrupted: 是否应使用 ``INTERRUPTED`` 而不是 ``RETAINED`` 状态。

        Returns:
            状态保存且租约释放后不返回数据。
        """

        lifecycle = (
            WorktreeLifecycle.INTERRUPTED
            if interrupted
            else WorktreeLifecycle.RETAINED
        )

        def update() -> None:
            """把任务 Worktree 标为保留或中断，并清空进程占用。

            Returns:
                内存记录更新完成后不返回数据。
            """

            self._records[record.name] = replace(
                record,
                lifecycle=lifecycle,
                task_state=(
                    WorktreeTaskState.INTERRUPTED
                    if interrupted
                    else WorktreeTaskState.FINISHED
                ),
                owner_pid=None,
                last_used_at=datetime.now(UTC),
            )

        await self._write_state(update)
        await self._release_lease(lease_id)

    @staticmethod
    def _can_delete_automatically(changes: WorktreeChangeSummary) -> bool:
        """判断变更摘要是否充分证明目录没有任何任务成果。

        Args:
            changes: Git 后端返回的文件和提交摘要。

        Returns:
            只有文件干净、HEAD 与基线相同、新提交为零、未推送数明确为零时
            返回 ``True``；任何 unknown/``None`` 都返回 ``False``。
        """

        return (
            not changes.has_file_changes
            and changes.relation_to_base is CommitRelation.SAME
            and changes.new_commit_count == 0
            and changes.unpushed_commit_count == 0
        )

    @staticmethod
    def _owner_process_status(pid: int | None) -> bool | None:
        """保守判断持久化记录中的原进程是否仍存在。

        Args:
            pid: 状态文件记录的正整数进程 ID；旧记录或非任务记录可能为
                ``None``。

        Returns:
            能确认进程存在时返回 ``True``，能确认已经结束时返回 ``False``；
            缺少 PID、权限或系统错误导致无法判断时返回 ``None``。调用方必须
            把 ``None`` 当作仍可能被占用，不能执行接管或删除。
        """

        if pid is None:
            return None
        if sys.platform == "win32":
            # Windows 的 os.kill(pid, 0) 会调用 TerminateProcess，不能拿它做
            # 存活探测。OpenProcess 只申请查询权限，不会向目标进程发送信号。
            import ctypes

            process_query_limited_information = 0x1000
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.OpenProcess(
                process_query_limited_information,
                False,
                pid,
            )
            if handle:
                kernel32.CloseHandle(handle)
                return True
            error = ctypes.get_last_error()
            if error == 87:  # ERROR_INVALID_PARAMETER：PID 不存在。
                return False
            if error == 5:  # ERROR_ACCESS_DENIED：进程存在但不能查询。
                return True
            return None
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return None
        return True

    async def _operation_lock(self, name: str) -> asyncio.Lock:
        """取得一个名称专属的长操作锁。

        Args:
            name: 原始 Worktree slug。

        Returns:
            同名操作共享、不同名操作互不共享的 ``asyncio.Lock``。
        """

        async with self._index_lock:
            return self._operation_locks.setdefault(name, asyncio.Lock())

    async def _require_lease(self, assignment: WorkspaceAssignment) -> None:
        """确认一个独立工作区分配仍持有当前进程签发的任务租约。

        Args:
            assignment: 等待运行或收尾的工作区分配。

        Returns:
            租约存在且名称匹配时不返回数据。

        Raises:
            WorktreeManagerError: 租约缺失、已释放或属于其他 Worktree。
        """

        lease_id = assignment.lease_id
        if lease_id is None:
            raise WorktreeManagerError("独立工作区缺少任务租约")
        async with self._index_lock:
            lease = self._leases.get(lease_id)
        if lease is None or lease[0] != assignment.worktree_name:
            raise WorktreeManagerError("任务 Worktree 租约已失效")

    async def _release_lease(self, lease_id: str | None) -> None:
        """幂等释放一个任务租约。

        Args:
            lease_id: ``create_for_task`` 或共享当前目录时签发的租约；``None``
                表示无需释放。

        Returns:
            租约不存在或删除完成后不返回数据。
        """

        if lease_id is None:
            return
        async with self._index_lock:
            self._leases.pop(lease_id, None)

    async def _set_lifecycle(
        self,
        name: str,
        lifecycle: WorktreeLifecycle,
    ) -> None:
        """更新一条仍存在记录的生命周期并保存状态。

        Args:
            name: 受管 Worktree slug。
            lifecycle: 准备写入的新生命周期状态。

        Returns:
            记录不存在时直接返回；更新成功后不返回数据。
        """

        def update() -> None:
            """在记录仍存在时替换生命周期和最后使用时间。

            Returns:
                内存记录更新完成后不返回数据。
            """

            record = self._records.get(name)
            if record is not None:
                self._records[name] = replace(
                    record,
                    lifecycle=lifecycle,
                    last_used_at=datetime.now(UTC),
                )

        await self._write_state(update)

    async def _remove_record(self, name: str) -> None:
        """删除最终结束的记录和指向它的历史会话映射。

        Args:
            name: 已确认目录和分支都不再需要的 Worktree slug。

        Returns:
            状态文件成功保存后不返回数据。
        """

        def update() -> None:
            """移除最终结束的记录及所有指向它的会话绑定。

            Returns:
                内存索引更新完成后不返回数据。
            """

            self._records.pop(name, None)
            stale_sessions = [
                session_id
                for session_id, bound_name in self._session_bindings.items()
                if bound_name == name
            ]
            for session_id in stale_sessions:
                del self._session_bindings[session_id]

        await self._write_state(update)

    async def _write_state(self, mutation: Callable[[], _T]) -> _T:
        """串行应用一次短内存修改，并把对应 revision 原子保存到磁盘。

        Args:
            mutation: 在 ``_index_lock`` 内同步执行的短函数，不得运行 Git、
                文件复制、网络或模型调用。

        Returns:
            ``mutation`` 的返回值。

        Raises:
            WorktreeStateError: 原子保存失败。内存记录会回滚到保存前的副本。
        """

        async with self._state_write_lock:
            async with self._index_lock:
                previous_records = dict(self._records)
                previous_bindings = dict(self._session_bindings)
                previous_revision = self._revision
                previous_active_session_id = self._active_session_id
                result = mutation()
                self._revision += 1
                snapshot = self._snapshot_locked()
            try:
                await asyncio.to_thread(self.state_store.save, snapshot)
            except Exception:
                async with self._index_lock:
                    self._records = previous_records
                    self._session_bindings = previous_bindings
                    self._revision = previous_revision
                    self._active_session_id = previous_active_session_id
                raise
            return result

    def _snapshot_locked(self) -> WorktreeStateSnapshot:
        """在 ``_index_lock`` 内把当前索引冻结成持久化快照。

        Returns:
            按名称和会话 ID 排序的 ``WorktreeStateSnapshot``。
        """

        return WorktreeStateSnapshot(
            revision=self._revision,
            records=tuple(record for _, record in sorted(self._records.items())),
            session_bindings=tuple(sorted(self._session_bindings.items())),
        )

    def _ensure_started(self) -> None:
        """确认 Manager 已经加载过磁盘状态。

        Returns:
            Manager 已启动时不返回数据。

        Raises:
            WorktreeManagerError: 调用方在 :meth:`start` 前使用 Manager。
        """

        if not self._started:
            raise WorktreeManagerError("WorktreeManager 尚未启动")

    def _ensure_trusted(self) -> None:
        """确认状态可信，可以执行创建、绑定或删除等受管操作。

        Returns:
            Manager 已启动且状态可信时不返回数据。

        Raises:
            WorktreeManagerError: Manager 未启动或状态文件损坏。
        """

        self._ensure_started()
        if not self._state_trusted:
            raise WorktreeManagerError(
                "Worktree 状态文件不可信，已停用受管创建和破坏性操作"
            )
