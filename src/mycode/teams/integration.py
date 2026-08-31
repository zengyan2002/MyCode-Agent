"""检查成员提交、记录合并冲突并保存分层验证证据。"""

from __future__ import annotations

import subprocess
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from mycode.models.teams import (
    TeamActorContext,
    TeamDeletionReport,
    TeamIntegrationState,
    TeamTaskStatus,
    TeammateState,
    ValidationReport,
)
from mycode.teams.store import TeamStateStore
from mycode.worktrees.manager import WorktreeManager, WorktreeManagerError


class TeamIntegrationError(RuntimeError):
    """表示提交归属、合并门限或验证前置条件不满足。"""


class TeamIntegrationService:
    """负责读取真实 Git 状态并持久化团队合并与验证结果。

    Attributes:
        workspace_root: Lead 当前主仓库的绝对路径。
        store: 团队任务、成员和 integration 状态 Store。
        worktrees: 查询成员目录修改、分支合并关系和清理状态的管理器。
    """

    def __init__(
        self,
        workspace_root: Path,
        store: TeamStateStore,
        worktrees: WorktreeManager,
    ) -> None:
        """绑定一个仓库和该仓库内唯一的团队 Store。

        Args:
            workspace_root: 主工作区绝对路径。
            store: 当前工作区的团队持久化入口。
            worktrees: 已启动的 WorktreeManager。

        Returns:
            不返回数据。
        """

        self.workspace_root = workspace_root.resolve(strict=True)
        self.store = store
        self.worktrees = worktrees

    async def validate_task_commit(
        self,
        actor: TeamActorContext,
        task_id: str,
    ) -> tuple[str, ...]:
        """确认代码任务的提交位于负责人分支且 Worktree 没有文件修改。

        Args:
            actor: 当前有效 Lead 或任务负责人身份。
            task_id: 已完成代码任务的 ID。

        Returns:
            通过检查的提交 SHA 元组；调查任务返回空元组。

        Raises:
            TeamIntegrationError: 任务未完成、提交缺失、分支不包含提交或目录脏。
        """

        self.store.require_actor(actor)
        snapshot = self.store.load_team(actor.team_id)
        task = next((item for item in snapshot.tasks if item.task_id == task_id), None)
        if task is None:
            raise TeamIntegrationError(f"任务不存在：{task_id}")
        if task.status is not TeamTaskStatus.COMPLETED:
            raise TeamIntegrationError("只有 completed 任务可以进入提交预检")
        if task.task_kind == "research":
            if not task.result:
                raise TeamIntegrationError("调查任务完成时必须保存结构化结果")
            return ()
        if not task.commit_hashes:
            raise TeamIntegrationError("代码任务完成时必须报告提交 SHA")
        member = next(
            (item for item in snapshot.members if item.agent_id == task.owner_id), None
        )
        if member is None:
            raise TeamIntegrationError("代码任务没有有效负责人")
        changes = await self.worktrees.inspect_changes(member.worktree_name)
        if changes.has_file_changes:
            raise TeamIntegrationError("成员 Worktree 仍有未提交文件")
        for commit in task.commit_hashes:
            self._require_commit_on_branch(commit, member.branch)
        return task.commit_hashes

    async def validate_task_completion(
        self,
        actor: TeamActorContext,
        task_id: str,
        *,
        commit_hashes: tuple[str, ...] | None,
        result: str | None,
    ) -> None:
        """在任务状态写成 ``completed`` 前验证本次提交内容。

        Args:
            actor: 当前任务负责人或有效 Lead 身份。
            task_id: 准备完成的任务 ID。
            commit_hashes: 本次更新明确报告的提交 SHA；未提供时使用任务
                已保存的提交列表。
            result: 本次更新明确报告的调查结果；未提供时使用任务已有结果。

        Returns:
            任务结果满足类型要求且代码提交可从负责人分支到达时不返回数据。

        Raises:
            TeamIntegrationError: 任务不存在、没有负责人、Worktree 有未提交
                修改、提交不属于负责人分支，或调查任务缺少结果。
        """

        self.store.require_actor(actor)
        snapshot = self.store.load_team(actor.team_id)
        task = next((item for item in snapshot.tasks if item.task_id == task_id), None)
        if task is None:
            raise TeamIntegrationError(f"任务不存在：{task_id}")
        if task.task_kind == "research":
            if not (result or task.result):
                raise TeamIntegrationError("调查任务完成前必须保存结构化结果")
            return
        effective_commits = commit_hashes if commit_hashes is not None else task.commit_hashes
        if not effective_commits:
            raise TeamIntegrationError("代码任务完成前必须报告提交 SHA")
        member = next(
            (item for item in snapshot.members if item.agent_id == task.owner_id),
            None,
        )
        if member is None:
            raise TeamIntegrationError("代码任务没有有效负责人")
        changes = await self.worktrees.inspect_changes(member.worktree_name)
        if changes.has_file_changes:
            raise TeamIntegrationError("成员 Worktree 仍有未提交文件")
        for commit in effective_commits:
            self._require_commit_on_branch(commit, member.branch)

    async def deletion_preflight(self, team_id: str) -> TeamDeletionReport:
        """只读检查团队是否已满足全量删除条件。

        Args:
            team_id: 准备删除的团队 ID。

        Returns:
            allowed 和全部阻塞原因。该方法不停止 Host、不写状态也不删资源。
        """

        snapshot = self.store.load_team(team_id)
        blockers: list[str] = []
        for member in snapshot.members:
            if member.state not in {
                TeammateState.IDLE,
                TeammateState.FAILED,
                TeammateState.TERMINATED,
            }:
                blockers.append(f"成员 {member.name} 仍处于 {member.state.value}")
            try:
                changes = await self.worktrees.inspect_changes(member.worktree_name)
            except WorktreeManagerError as exc:
                blockers.append(f"无法确认 {member.name} 的 Worktree：{exc}")
                continue
            if changes.has_file_changes:
                blockers.append(f"成员 {member.name} 的 Worktree 有未提交修改")
            if changes.new_commit_count > 0 and changes.merged_into_base is not True:
                blockers.append(f"成员 {member.name} 的提交尚未合并")
        for task in snapshot.tasks:
            if task.status is TeamTaskStatus.WORKING:
                blockers.append(f"任务 {task.task_id} 仍在 working")
        if snapshot.integration.blocked_by_validation:
            blockers.append("最近一次中间验证失败，尚未完成关联修复")
        return TeamDeletionReport(team_id, not blockers, tuple(blockers))

    async def begin_merge(
        self,
        actor: TeamActorContext,
        source_branch: str,
    ) -> TeamIntegrationState:
        """验证来源分支和成员成果，再登记下一次 merge attempt。

        Args:
            actor: 当前有效 Lead 身份。
            source_branch: Lead 准备传给 ``git merge`` 的成员分支。

        Returns:
            merge_attempt 已增加的 integration 状态。

        Raises:
            TeamIntegrationError: 分支不属于团队、没有已完成成果、成员目录
                不干净、主目录存在会被覆盖的修改、验证仍阻塞、已有另一场
                合并或同源分支已经用完两次机会。
        """

        self.store.require_actor(actor)
        if actor.actor_kind != "lead":
            raise TeamIntegrationError("只有 Lead 能开始合并")
        snapshot = self.store.load_team(actor.team_id)
        member = next(
            (item for item in snapshot.members if item.branch == source_branch),
            None,
        )
        if member is None:
            raise TeamIntegrationError("来源分支不属于当前团队成员")
        completed = tuple(
            task
            for task in snapshot.tasks
            if task.owner_id == member.agent_id
            and task.task_kind == "code"
            and task.status is TeamTaskStatus.COMPLETED
            and task.commit_hashes
        )
        if not completed:
            raise TeamIntegrationError("来源分支没有已完成并报告提交的代码任务")
        changes = await self.worktrees.inspect_changes(member.worktree_name)
        if changes.has_file_changes:
            raise TeamIntegrationError("来源成员 Worktree 仍有未提交文件")
        for task in completed:
            for commit in task.commit_hashes:
                self._require_commit_on_branch(commit, member.branch)
        dirty_paths = self._main_dirty_paths()
        changed_paths = self._branch_changed_paths(source_branch)
        overlaps = sorted(dirty_paths & changed_paths)
        if overlaps:
            preview = "、".join(overlaps[:5])
            raise TeamIntegrationError(f"主工作区有可能被合并覆盖的未提交文件：{preview}")

        def mutation(current: TeamIntegrationState) -> TeamIntegrationState:
            """在 Store 锁内检查门限并生成下一版合并状态。

            Args:
                current: 磁盘中刚读取的当前团队合并状态。

            Returns:
                已登记本次来源分支和尝试次数的新状态。
            """

            if current.blocked_by_validation:
                raise TeamIntegrationError("中间验证失败，完成修复前不能继续合并")
            if current.current_source_branch not in {None, source_branch}:
                raise TeamIntegrationError("另一个来源分支仍处于合并流程")
            if current.current_source_branch == source_branch and current.merge_attempt >= 2:
                raise TeamIntegrationError("该来源分支已经用完两次合并机会")
            attempt = current.merge_attempt + 1 if current.current_source_branch == source_branch else 1
            return replace(
                current,
                current_source_branch=source_branch,
                merge_attempt=attempt,
                conflicted_files=(),
                updated_at=_now(),
            )

        return self.store.update_integration(actor, mutation)

    def observe_merge(
        self,
        actor: TeamActorContext,
        *,
        command_succeeded: bool,
    ) -> TeamIntegrationState:
        """从 Git index 判断刚执行的 merge 已成功还是产生冲突。

        Args:
            actor: 执行受控 Git 合并的当前 Lead 身份。
            command_succeeded: 命令工具报告的真实成功状态。

        Returns:
            成功时追加当前 HEAD 并清空活动来源；冲突时记录真实未合并文件；
            非冲突失败保留来源和次数，供 Lead 查看错误或中止。
        """

        conflicts = tuple(
            Path(line)
            for line in self._git(["diff", "--name-only", "--diff-filter=U"]).splitlines()
            if line.strip()
        )
        head = self._git(["rev-parse", "HEAD"]).strip()

        def mutation(current: TeamIntegrationState) -> TeamIntegrationState:
            """根据刚读取的 Git 冲突与命令结果生成下一版状态。

            Args:
                current: 合并命令执行前保存的团队合并状态。

            Returns:
                记录冲突文件、失败结果或新合并提交的新状态。
            """

            if conflicts:
                return replace(current, conflicted_files=conflicts, updated_at=_now())
            if not command_succeeded:
                return replace(current, conflicted_files=(), updated_at=_now())
            merged = current.merged_commits
            if head and head not in merged:
                merged = (*merged, head)
            return replace(
                current,
                merged_commits=merged,
                current_source_branch=None,
                merge_attempt=0,
                conflicted_files=(),
                updated_at=_now(),
            )

        return self.store.update_integration(actor, mutation)

    def observe_abort(self, actor: TeamActorContext) -> TeamIntegrationState:
        """记录 Lead 已中止当前 merge，但保留同源 attempt 次数。

        Args:
            actor: 当前有效 Lead 身份。

        Returns:
            冲突文件已清空、来源分支和 attempt 保留的状态。
        """

        return self.store.update_integration(
            actor,
            lambda current: replace(current, conflicted_files=(), updated_at=_now()),
        )

    def record_validation(
        self,
        actor: TeamActorContext,
        *,
        command: str,
        scope: str,
        exit_code: int,
    ) -> TeamIntegrationState:
        """保存一条实际执行过的中间或最终验证证据。

        Args:
            actor: 当前有效 Lead 身份。
            command: 实际执行的测试、编译或静态检查命令。
            scope: ``focused`` 表示合并后的轻量验证，``final`` 表示最终全量验证。
            exit_code: 进程真实退出码；非零会设置验证阻塞。

        Returns:
            已追加报告的 integration 状态。失败会建立验证阻塞；普通成功
            不会绕过修复任务直接解除已有阻塞。
        """

        if scope not in {"focused", "final"}:
            raise TeamIntegrationError("验证范围只能是 focused 或 final")
        report = ValidationReport(
            command=command,
            scope=scope,  # type: ignore[arg-type]
            exit_code=exit_code,
            head=self._git(["rev-parse", "HEAD"]).strip(),
            ran_at=_now(),
        )
        return self.store.update_integration(
            actor,
            lambda current: replace(
                current,
                validation_reports=(*current.validation_reports, report),
                blocked_by_validation=(
                    True if exit_code != 0 else current.blocked_by_validation
                ),
                validation_repair_task_id=(
                    None if exit_code != 0 else current.validation_repair_task_id
                ),
                updated_at=_now(),
            ),
        )

    def register_validation_repair(
        self,
        actor: TeamActorContext,
        task_id: str,
    ) -> TeamIntegrationState:
        """把验证失败后创建的第一项代码任务登记为修复任务。

        Args:
            actor: 创建修复任务的当前 Lead 身份。
            task_id: 已真实写入共享看板的新代码任务 ID。

        Returns:
            已关联修复任务的 integration 状态；当前没有验证阻塞时原样返回。
        """

        return self.store.update_integration(
            actor,
            lambda current: (
                replace(
                    current,
                    validation_repair_task_id=task_id,
                    updated_at=_now(),
                )
                if current.blocked_by_validation
                and current.validation_repair_task_id is None
                else current
            ),
        )

    def resolve_validation_repair(
        self,
        actor: TeamActorContext,
        task_id: str,
    ) -> TeamIntegrationState:
        """在已登记修复任务完成后解除后续合并阻塞。

        Args:
            actor: 完成任务的有效成员或 Lead 身份。
            task_id: 刚刚成功写成 completed 的任务 ID。

        Returns:
            任务与登记 ID 匹配时清除阻塞；不匹配时保持原状态。
        """

        return self.store.update_integration(
            actor,
            lambda current: (
                replace(
                    current,
                    blocked_by_validation=False,
                    validation_repair_task_id=None,
                    updated_at=_now(),
                )
                if current.validation_repair_task_id == task_id
                else current
            ),
        )

    def _require_commit_on_branch(self, commit: str, branch: str) -> None:
        """确认一个提交存在且能从成员分支到达。

        Args:
            commit: 任务报告的完整或可解析提交 SHA。
            branch: 任务负责人 Worktree 的本地分支。

        Returns:
            提交存在且属于该分支时不返回数据。

        Raises:
            TeamIntegrationError: 提交不存在或不是该分支祖先。
        """

        try:
            self._git(["cat-file", "-e", f"{commit}^{{commit}}"])
            self._git(["merge-base", "--is-ancestor", commit, branch])
        except TeamIntegrationError as exc:
            raise TeamIntegrationError(f"提交 {commit} 不属于成员分支 {branch}") from exc

    def _main_dirty_paths(self) -> set[str]:
        """读取 Lead 主工作区当前已跟踪和未跟踪的修改路径。

        Returns:
            以正斜杠表示的仓库相对路径集合。
        """

        paths: set[str] = set()
        for line in self._git(["status", "--porcelain", "--untracked-files=all"]).splitlines():
            if len(line) < 4:
                continue
            raw = line[3:]
            if " -> " in raw:
                raw = raw.split(" -> ", 1)[1]
            paths.add(Path(raw.strip('"')).as_posix())
        return paths

    def _branch_changed_paths(self, branch: str) -> set[str]:
        """读取来源分支相对当前 HEAD 会带入的文件路径。

        Args:
            branch: 已确认属于团队成员的本地分支。

        Returns:
            合并可能新增、删除或修改的仓库相对路径集合。
        """

        return {
            Path(line).as_posix()
            for line in self._git(["diff", "--name-only", f"HEAD...{branch}"]).splitlines()
            if line.strip()
        }

    def _git(self, args: list[str]) -> str:
        """在主仓库运行一条无 Shell、无凭据交互的 Git 命令。

        Args:
            args: ``git`` 后面的参数列表，每个元素保留参数边界。

        Returns:
            命令成功时的标准输出文本。

        Raises:
            TeamIntegrationError: Git 无法启动、超时或返回非零退出码。
        """

        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self.workspace_root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TeamIntegrationError(f"Git 命令无法完成：{exc}") from exc
        if result.returncode != 0:
            detail = result.stderr.strip()[:500]
            raise TeamIntegrationError(f"Git 命令失败（{result.returncode}）：{detail}")
        return result.stdout


def _now() -> datetime:
    """返回 integration 快照使用的带时区当前时间。

    Returns:
        当前本地时区的 ``datetime``。
    """

    return datetime.now().astimezone()
