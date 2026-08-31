"""为新建 Worktree 复制本地配置、链接依赖并设置专属 Hooks。"""

from __future__ import annotations

import filecmp
import os
import shutil
from pathlib import Path, PurePosixPath

from mycode.models.config import (
    WorktreeIgnoredRule,
    WorktreePathRule,
    WorktreeSettings,
)
from mycode.models.worktrees import (
    InitializationAction,
    InitializationActionStatus,
    InitializationReport,
    WorktreeRecord,
)
from mycode.worktrees.git import GitWorktreeBackend, WorktreeGitError


class WorktreeInitializer:
    """执行配置中批准的新 Worktree 创建后动作。

    Attributes:
        repo_root: 复制和软链接来源所在的主仓库绝对路径。
        settings: 已完成分层合并和边界校验的 Worktree 配置。
        git: 只用于列出 ignored 文件和写 Worktree 专属 Hooks 的本地 Git 后端。

    初始化按“明确文件、依赖链接、ignored 内容、Hooks”顺序执行。必需项失败会
    立即返回 ``complete=False``；可选项失败只产生不含文件正文的警告。
    """

    def __init__(
        self,
        repo_root: Path,
        settings: WorktreeSettings,
        git: GitWorktreeBackend,
    ) -> None:
        """创建一个仓库专用初始化器。

        Args:
            repo_root: 主仓库绝对路径。
            settings: Worktree 初始化配置。
            git: 与同一主仓库关联的 Git 后端。

        Returns:
            新的初始化器。

        Raises:
            ValueError: 路径、配置或 Git 后端类型无效，或后端指向其他仓库。
        """

        if not isinstance(repo_root, Path) or not repo_root.is_absolute():
            raise ValueError("WorktreeInitializer.repo_root 必须是绝对 Path")
        if not isinstance(settings, WorktreeSettings):
            raise ValueError("WorktreeInitializer.settings 类型无效")
        if not isinstance(git, GitWorktreeBackend):
            raise ValueError("WorktreeInitializer.git 类型无效")
        resolved = repo_root.resolve()
        if git.repo_root != resolved:
            raise ValueError("WorktreeInitializer.git 必须指向同一个主仓库")
        self.repo_root = resolved
        self.settings = settings
        self.git = git

    def initialize(self, worktree: WorktreeRecord) -> InitializationReport:
        """对一个已经由 Git 创建的 Worktree 执行全部配置动作。

        Args:
            worktree: 路径已存在、仍处于创建阶段的受管 Worktree 记录。

        Returns:
            按执行顺序列出动作、完成标志和可选警告的 ``InitializationReport``。
            报告只包含路径和原因，不读取或展示文件正文。

        Raises:
            ValueError: ``worktree`` 类型无效或路径越过当前仓库受管目录。
        """

        if not isinstance(worktree, WorktreeRecord):
            raise ValueError("initialize worktree 类型无效")
        managed_root = (self.repo_root / ".mycode" / "worktrees").resolve()
        try:
            worktree.path.resolve().relative_to(managed_root)
        except ValueError as exc:
            raise ValueError("初始化目标越过 .mycode/worktrees 目录") from exc

        actions: list[InitializationAction] = []
        warnings: list[str] = []
        for rule in self.settings.copy_files:
            if not self._copy_explicit(rule, worktree.path, actions, warnings):
                return InitializationReport(tuple(actions), False, tuple(warnings))
        for rule in self.settings.symlink_directories:
            if not self._link_directory(rule, worktree.path, actions, warnings):
                return InitializationReport(tuple(actions), False, tuple(warnings))
        for rule in self.settings.copy_ignored:
            if not self._copy_ignored(rule, worktree.path, actions, warnings):
                return InitializationReport(tuple(actions), False, tuple(warnings))
        if self.settings.hooks_path is not None:
            try:
                self.git.configure_hooks(worktree.path, self.settings.hooks_path)
            except (WorktreeGitError, OSError) as exc:
                actions.append(
                    InitializationAction(
                        "hooks",
                        self.settings.hooks_path,
                        InitializationActionStatus.FAILED,
                        str(exc),
                    )
                )
                return InitializationReport(tuple(actions), False, tuple(warnings))
            actions.append(
                InitializationAction(
                    "hooks",
                    self.settings.hooks_path,
                    InitializationActionStatus.COMPLETED,
                    "已写入 Worktree 专属 core.hooksPath",
                )
            )
        return InitializationReport(tuple(actions), True, tuple(warnings))

    def _copy_explicit(
        self,
        rule: WorktreePathRule,
        worktree_root: Path,
        actions: list[InitializationAction],
        warnings: list[str],
    ) -> bool:
        """复制一条 ``copy_files`` 规则指定的文件。

        Args:
            rule: 已校验的相对路径和必需标志。
            worktree_root: 新 Worktree 绝对根目录。
            actions: 本次初始化按顺序累积动作的列表。
            warnings: 本次初始化累积可选失败说明的列表。

        Returns:
            动作成功、幂等或可选失败时返回 ``True``；必需项失败返回 ``False``。
        """

        source = self.repo_root / Path(rule.path)
        target = worktree_root / Path(rule.path)
        if not source.is_file():
            return self._record_problem(
                "copy_file",
                rule.path,
                "主仓库来源文件不存在或不是文件",
                rule.required,
                actions,
                warnings,
            )
        try:
            if target.exists():
                if target.is_file() and filecmp.cmp(source, target, shallow=False):
                    actions.append(
                        InitializationAction(
                            "copy_file",
                            rule.path,
                            InitializationActionStatus.SKIPPED,
                            "目标已存在且内容相同",
                        )
                    )
                    return True
                return self._record_problem(
                    "copy_file",
                    rule.path,
                    "目标已存在且与来源不同",
                    rule.required,
                    actions,
                    warnings,
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        except OSError as exc:
            return self._record_problem(
                "copy_file",
                rule.path,
                f"复制失败：{exc}",
                rule.required,
                actions,
                warnings,
            )
        actions.append(
            InitializationAction(
                "copy_file",
                rule.path,
                InitializationActionStatus.COMPLETED,
                "已从主仓库复制",
            )
        )
        return True

    def _link_directory(
        self,
        rule: WorktreePathRule,
        worktree_root: Path,
        actions: list[InitializationAction],
        warnings: list[str],
    ) -> bool:
        """把一条大型依赖目录规则链接到主仓库对应目录。

        Args:
            rule: 已校验的目录路径和必需标志。
            worktree_root: 新 Worktree 绝对根目录。
            actions: 本次初始化累积动作的列表。
            warnings: 本次初始化累积可选失败说明的列表。

        Returns:
            链接成功、已指向同一来源或可选失败时返回 ``True``；必需项失败返回
            ``False``。链接失败不会退化为复制整个目录。
        """

        source = (self.repo_root / Path(rule.path)).resolve()
        target = worktree_root / Path(rule.path)
        if not source.is_dir():
            return self._record_problem(
                "symlink",
                rule.path,
                "主仓库来源目录不存在或不是目录",
                rule.required,
                actions,
                warnings,
            )
        try:
            if target.is_symlink():
                if target.resolve() == source:
                    actions.append(
                        InitializationAction(
                            "symlink",
                            rule.path,
                            InitializationActionStatus.SKIPPED,
                            "目标软链接已指向主仓库目录",
                        )
                    )
                    return True
                return self._record_problem(
                    "symlink",
                    rule.path,
                    "目标软链接指向其他位置",
                    rule.required,
                    actions,
                    warnings,
                )
            if target.exists():
                return self._record_problem(
                    "symlink",
                    rule.path,
                    "目标已存在且不是预期软链接",
                    rule.required,
                    actions,
                    warnings,
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source, target_is_directory=True)
        except OSError as exc:
            return self._record_problem(
                "symlink",
                rule.path,
                f"创建软链接失败：{exc}",
                rule.required,
                actions,
                warnings,
            )
        actions.append(
            InitializationAction(
                "symlink",
                rule.path,
                InitializationActionStatus.COMPLETED,
                "已链接到主仓库依赖目录",
            )
        )
        return True

    def _copy_ignored(
        self,
        rule: WorktreeIgnoredRule,
        worktree_root: Path,
        actions: list[InitializationAction],
        warnings: list[str],
    ) -> bool:
        """匹配 Git 已确认忽略的实际文件并逐个复制。

        Args:
            rule: 已校验的 shell 风格模式和必需标志。
            worktree_root: 新 Worktree 绝对根目录。
            actions: 本次初始化累积动作的列表。
            warnings: 本次初始化累积可选失败说明的列表。

        Returns:
            所有匹配项复制成功、幂等或可选失败时返回 ``True``；必需规则没有
            匹配或任一匹配项失败时返回 ``False``。
        """

        try:
            ignored = self.git.list_ignored_paths()
        except WorktreeGitError as exc:
            return self._record_problem(
                "copy_ignored",
                rule.pattern,
                str(exc),
                rule.required,
                actions,
                warnings,
            )
        matched = [
            item
            for item in ignored
            if PurePosixPath(item).match(rule.pattern)
        ]
        if not matched:
            return self._record_problem(
                "copy_ignored",
                rule.pattern,
                "没有实际 ignored 文件匹配该模式",
                rule.required,
                actions,
                warnings,
            )
        for relative in matched:
            source = self.repo_root / Path(relative)
            target = worktree_root / Path(relative)
            if not source.is_file():
                if not self._record_problem(
                    "copy_ignored",
                    relative,
                    "匹配项不存在或不是文件",
                    rule.required,
                    actions,
                    warnings,
                ):
                    return False
                continue
            try:
                if target.exists():
                    if target.is_file() and filecmp.cmp(source, target, shallow=False):
                        actions.append(
                            InitializationAction(
                                "copy_ignored",
                                relative,
                                InitializationActionStatus.SKIPPED,
                                "目标已存在且内容相同",
                            )
                        )
                        continue
                    if not self._record_problem(
                        "copy_ignored",
                        relative,
                        "目标已存在且与来源不同",
                        rule.required,
                        actions,
                        warnings,
                    ):
                        return False
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            except OSError as exc:
                if not self._record_problem(
                    "copy_ignored",
                    relative,
                    f"复制失败：{exc}",
                    rule.required,
                    actions,
                    warnings,
                ):
                    return False
                continue
            actions.append(
                InitializationAction(
                    "copy_ignored",
                    relative,
                    InitializationActionStatus.COMPLETED,
                    f"匹配模式 {rule.pattern} 并复制",
                )
            )
        return True

    @staticmethod
    def _record_problem(
        operation: str,
        target: str,
        message: str,
        required: bool,
        actions: list[InitializationAction],
        warnings: list[str],
    ) -> bool:
        """把必需或可选初始化失败写入报告。

        Args:
            operation: 失败动作类别。
            target: 失败的相对路径或模式。
            message: 不含文件正文的具体失败原因。
            required: 配置是否要求该动作必须成功。
            actions: 本次初始化累积动作的列表。
            warnings: 本次初始化累积可选失败说明的列表。

        Returns:
            可选失败返回 ``True`` 让初始化继续；必需失败返回 ``False``。
        """

        status = (
            InitializationActionStatus.FAILED
            if required
            else InitializationActionStatus.WARNING
        )
        actions.append(InitializationAction(operation, target, status, message))
        if not required:
            warnings.append(f"{operation} {target}：{message}")
        return not required
