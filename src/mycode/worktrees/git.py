"""执行本地、非交互的 Git Worktree 命令并返回结构化事实。"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from mycode.constants import GIT_STATUS_TIMEOUT_SECONDS
from mycode.errors import redact_secrets
from mycode.models.worktrees import (
    CommitRelation,
    GitHead,
    GitWorktreeEntry,
    WorktreeChangeSummary,
    WorktreeRecord,
)


_URL_CREDENTIALS = re.compile(r"(?i)(https?://)([^/@\s:]+):([^/@\s]+)@")


class WorktreeGitError(RuntimeError):
    """说明一个本地 Git 操作失败，并保存可直接展示的错误信息。

    Attributes:
        purpose: 上层发起该命令的用途，例如“创建 Worktree”。
        returncode: Git 进程退出码；启动进程失败时为 ``None``。
        stderr: 已去除常见认证值并限制长度的 Git 错误文本。
    """

    def __init__(
        self,
        purpose: str,
        returncode: int | None,
        stderr: str,
    ) -> None:
        """保存失败用途、退出码和已处理的错误文本。

        Args:
            purpose: 命令在产品流程中的用途，不是完整命令行。
            returncode: Git 退出码；无法启动时为 ``None``。
            stderr: 已经脱敏、适合展示的简短错误文本。

        Returns:
            新的 ``WorktreeGitError`` 异常。
        """

        code = "无法启动" if returncode is None else f"退出码 {returncode}"
        message = f"{purpose}失败（{code}）"
        if stderr:
            message += f"：{stderr}"
        super().__init__(message)
        self.purpose = purpose
        self.returncode = returncode
        self.stderr = stderr


class GitWorktreeBackend:
    """封装 MyCode 需要的全部本地 Git Worktree 操作。

    Attributes:
        repo_root: 主仓库绝对路径。所有仓库级命令固定从这里执行。
        timeout_seconds: 单条本地 Git 命令允许执行的最长秒数。

    该类直接使用生产环境唯一的 ``subprocess.run`` 实现，不额外增加只供测试
    注入的接口。测试通过真实临时 Git 仓库验证行为。
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        timeout_seconds: float = GIT_STATUS_TIMEOUT_SECONDS,
    ) -> None:
        """创建只操作一个本地仓库的 Git 后端。

        Args:
            repo_root: 主仓库绝对路径。
            timeout_seconds: 每条 Git 命令的正数超时秒数。

        Returns:
            新的 Git 后端。

        Raises:
            ValueError: 仓库路径不是绝对 ``Path``，或超时不是正数。
        """

        if not isinstance(repo_root, Path) or not repo_root.is_absolute():
            raise ValueError("GitWorktreeBackend.repo_root 必须是绝对 Path")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("GitWorktreeBackend.timeout_seconds 必须是正数")
        self.repo_root = repo_root.resolve()
        self.timeout_seconds = float(timeout_seconds)

    def _run_result(
        self,
        args: list[str],
        *,
        purpose: str,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """执行一条不经过 Shell、不会等待凭据输入的 Git 命令。

        Args:
            args: ``git`` 后面的独立参数，每个列表元素保持自己的参数边界。
            purpose: 命令失败时显示的业务用途。
            cwd: 命令工作目录；未传时固定使用主仓库。

        Returns:
            无论退出码为何都返回 ``CompletedProcess``，供需要解释退出码 1 的
            调用方判断；启动或超时失败会直接抛异常。

        Raises:
            WorktreeGitError: Git 不存在、进程无法启动或命令超时。
        """

        work_dir = self.repo_root if cwd is None else cwd.resolve()
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_ASKPASS"] = ""
        try:
            return subprocess.run(
                ["git", *args],
                cwd=work_dir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
            )
        except (FileNotFoundError, OSError) as exc:
            raise WorktreeGitError(purpose, None, str(exc)) from exc
        except subprocess.TimeoutExpired as exc:
            raise WorktreeGitError(
                purpose,
                None,
                f"超过 {self.timeout_seconds:g} 秒",
            ) from exc

    def _run(
        self,
        args: list[str],
        *,
        purpose: str,
        cwd: Path | None = None,
    ) -> str:
        """执行一条必须成功的 Git 命令并返回标准输出。

        Args:
            args: ``git`` 后面的独立参数。
            purpose: 错误消息中显示的业务用途。
            cwd: 命令工作目录；未传时使用主仓库。

        Returns:
            Git 的完整标准输出，不包含末尾换行。

        Raises:
            WorktreeGitError: Git 无法运行、超时或返回非零退出码。
        """

        result = self._run_result(args, purpose=purpose, cwd=cwd)
        if result.returncode != 0:
            raise WorktreeGitError(
                purpose,
                result.returncode,
                self._safe_stderr(result.stderr),
            )
        return result.stdout.rstrip("\r\n")

    @staticmethod
    def _safe_stderr(stderr: str) -> str:
        """把 Git 错误文本处理成可以显示的单行摘要。

        Args:
            stderr: Git 进程返回的原始标准错误。

        Returns:
            已清理常见认证头和 URL 用户密码、合并换行并限制为 1000 字符的文本。
        """

        redacted = redact_secrets(stderr)
        redacted = _URL_CREDENTIALS.sub(r"\1***:***@", redacted)
        return " ".join(redacted.split())[:1000]

    def resolve_local_head(self, *, cwd: Path | None = None) -> GitHead:
        """读取一个工作目录当前的本地分支和 HEAD commit。

        Args:
            cwd: 要读取的 Git 工作目录；未传时读取主仓库。

        Returns:
            包含完整 commit SHA 和可选符号分支名的 ``GitHead``。

        Raises:
            WorktreeGitError: 目录不是仓库，HEAD 无法解析或 Git 调用失败。
        """

        commit = self._run(
            ["rev-parse", "--verify", "HEAD^{commit}"],
            purpose="读取本地 HEAD",
            cwd=cwd,
        ).strip()
        branch_result = self._run_result(
            ["symbolic-ref", "--quiet", "--short", "HEAD"],
            purpose="读取本地分支",
            cwd=cwd,
        )
        if branch_result.returncode not in {0, 1}:
            raise WorktreeGitError(
                "读取本地分支",
                branch_result.returncode,
                self._safe_stderr(branch_result.stderr),
            )
        branch = branch_result.stdout.strip() or None
        return GitHead(branch=branch, commit=commit)

    def parent_has_changes(self, *, cwd: Path | None = None) -> bool:
        """检查父工作区是否有暂存、未暂存或未追踪文件。

        Args:
            cwd: 要检查的工作目录；未传时检查主仓库。

        Returns:
            ``git status --porcelain`` 有任意记录时返回 ``True``。

        Raises:
            WorktreeGitError: Git 状态无法读取。
        """

        output = self._run(
            ["status", "--porcelain=v1", "--untracked-files=all"],
            purpose="检查父工作区变更",
            cwd=cwd,
        )
        return bool(output)

    def validate_branch_name(self, branch: str) -> None:
        """让 Git 校验最终生成的本地分支名。

        Args:
            branch: 已加 ``mycode-worktree-`` 前缀的候选分支名。

        Returns:
            Git 接受该分支名时不返回数据。

        Raises:
            ValueError: 分支名不是非空字符串。
            WorktreeGitError: ``git check-ref-format --branch`` 拒绝该名称。
        """

        if not isinstance(branch, str) or not branch.strip():
            raise ValueError("branch 必须是非空字符串")
        self._run(
            ["check-ref-format", "--branch", branch],
            purpose="校验 Worktree 分支名",
        )

    def list_worktrees(self) -> tuple[GitWorktreeEntry, ...]:
        """读取 Git 登记的全部工作目录。

        Returns:
            按 Git 输出顺序排列的 ``GitWorktreeEntry`` 元组。

        Raises:
            WorktreeGitError: 列表命令失败或输出缺少路径/HEAD 等必要字段。
        """

        output = self._run(
            ["worktree", "list", "--porcelain"],
            purpose="列出 Git Worktree",
        )
        entries: list[GitWorktreeEntry] = []
        block: dict[str, str | bool] = {}

        def finish() -> None:
            """把当前解析块转换为 Worktree 条目并清空临时字段。

            Returns:
                当前块为空或转换完成后不返回数据。

            Raises:
                WorktreeGitError: 当前块缺少 Worktree 路径或 HEAD。
            """

            if not block:
                return
            path_value = block.get("worktree")
            head_value = block.get("HEAD")
            if not isinstance(path_value, str) or not isinstance(head_value, str):
                raise WorktreeGitError("解析 Git Worktree 列表", None, "条目缺少路径或 HEAD")
            branch_value = block.get("branch")
            branch = None
            if isinstance(branch_value, str):
                prefix = "refs/heads/"
                branch = (
                    branch_value[len(prefix) :]
                    if branch_value.startswith(prefix)
                    else branch_value
                )
            entries.append(
                GitWorktreeEntry(
                    path=Path(path_value).resolve(),
                    head_commit=head_value,
                    branch=branch,
                    bare=bool(block.get("bare", False)),
                    detached=bool(block.get("detached", False)),
                    prunable=bool(block.get("prunable", False)),
                )
            )
            block.clear()

        for line in output.splitlines():
            if not line:
                finish()
                continue
            key, _, value = line.partition(" ")
            block[key] = value if value else True
        finish()
        return tuple(entries)

    def add(self, path: Path, branch: str, base_commit: str) -> None:
        """从一个已解析的本地 commit 创建新分支和 Worktree。

        Args:
            path: 新 Worktree 的绝对目标目录。
            branch: 新建的本地分支名。
            base_commit: 父工作区创建时的本地 HEAD commit SHA。

        Returns:
            ``git worktree add`` 成功时不返回数据。

        Raises:
            ValueError: 路径或文本参数无效。
            WorktreeGitError: 目录、分支已占用或 Git 创建失败。
        """

        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("Worktree add path 必须是绝对 Path")
        if not branch.strip() or not base_commit.strip():
            raise ValueError("Worktree add branch 和 base_commit 不能为空")
        self._run(
            ["worktree", "add", "-b", branch, os.fspath(path), base_commit],
            purpose="创建 Git Worktree",
        )

    def validate_existing(self, record: WorktreeRecord) -> GitWorktreeEntry:
        """确认已有目录仍属于本仓库、登记路径和分支均与状态记录一致。

        Args:
            record: 状态文件中等待快速复用的受管记录。

        Returns:
            Git 当前登记的匹配条目。

        Raises:
            WorktreeGitError: 路径未登记、分支不符，或目录 HEAD 无法解析。
        """

        expected_path = record.path.resolve()
        entry = next(
            (item for item in self.list_worktrees() if item.path == expected_path),
            None,
        )
        if entry is None:
            raise WorktreeGitError("验证已有 Worktree", None, "目录未在当前仓库登记")
        if entry.branch != record.branch:
            raise WorktreeGitError("验证已有 Worktree", None, "登记分支与状态记录不一致")
        head = self.resolve_local_head(cwd=record.path)
        if head.branch != record.branch or head.commit != entry.head_commit:
            raise WorktreeGitError("验证已有 Worktree", None, "目录 HEAD 与 Git 登记不一致")
        return entry

    def inspect_changes(self, record: WorktreeRecord) -> WorktreeChangeSummary:
        """检查文件变化、提交关系、未推送提交和是否合入基准。

        Args:
            record: 包含 Worktree 路径、创建基线和基准引用的受管记录。

        Returns:
            可用于自动删除保护判断的 ``WorktreeChangeSummary``。无法可靠确认的
            提交事实使用 ``unknown`` 或 ``None``，不会当成“没有变化”。

        Raises:
            WorktreeGitError: 文件状态或当前 HEAD 这些基础事实无法读取。
        """

        head = self.resolve_local_head(cwd=record.path)
        status = self._run(
            ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
            purpose="检查 Worktree 文件变更",
            cwd=record.path,
        )
        staged: list[str] = []
        unstaged: list[str] = []
        untracked: list[str] = []
        items = status.split("\0")
        index = 0
        while index < len(items):
            item = items[index]
            index += 1
            if not item:
                continue
            if len(item) < 3:
                raise WorktreeGitError("解析 Worktree 文件变更", None, "porcelain 条目过短")
            x, y = item[0], item[1]
            path = item[3:]
            if x == "?" and y == "?":
                untracked.append(path)
                continue
            if x not in {" ", "?"}:
                staged.append(path)
            if y not in {" ", "?"}:
                unstaged.append(path)
            if x in {"R", "C"} and index < len(items):
                index += 1

        relation = self._commit_relation(
            record.base_commit,
            head.commit,
            cwd=record.path,
        )
        new_count = self._count_revisions(
            f"{record.base_commit}..{head.commit}",
            cwd=record.path,
        )
        upstream_result = self._run_result(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
            purpose="读取 Worktree upstream",
            cwd=record.path,
        )
        if upstream_result.returncode == 0:
            upstream = upstream_result.stdout.strip()
            unpushed = self._count_revisions(
                f"{upstream}..{head.commit}",
                cwd=record.path,
            )
        elif upstream_result.returncode in {1, 128}:
            unpushed = new_count if new_count > 0 else 0
        else:
            unpushed = None

        merged_result = self._run_result(
            ["merge-base", "--is-ancestor", head.commit, record.base_ref],
            purpose="检查 Worktree 分支是否已合入基准",
            cwd=record.path,
        )
        merged = (
            True
            if merged_result.returncode == 0
            else False if merged_result.returncode == 1 else None
        )
        return WorktreeChangeSummary(
            staged=tuple(staged),
            unstaged=tuple(unstaged),
            untracked=tuple(untracked),
            head_commit=head.commit,
            relation_to_base=relation,
            new_commit_count=new_count,
            unpushed_commit_count=unpushed,
            merged_into_base=merged,
        )

    def _commit_relation(
        self,
        base_commit: str,
        head_commit: str,
        *,
        cwd: Path,
    ) -> CommitRelation:
        """比较两个 commit 的相等和双向祖先关系。

        Args:
            base_commit: Worktree 创建时的本地基线 SHA。
            head_commit: 当前 Worktree HEAD SHA。
            cwd: 包含这两个 commit 的 Worktree 目录。

        Returns:
            ``same``、``ahead``、``behind``、``diverged`` 或 ``unknown``。
        """

        if base_commit == head_commit:
            return CommitRelation.SAME
        base_is_ancestor = self._run_result(
            ["merge-base", "--is-ancestor", base_commit, head_commit],
            purpose="比较 Worktree 提交关系",
            cwd=cwd,
        )
        head_is_ancestor = self._run_result(
            ["merge-base", "--is-ancestor", head_commit, base_commit],
            purpose="比较 Worktree 提交关系",
            cwd=cwd,
        )
        if base_is_ancestor.returncode not in {0, 1} or head_is_ancestor.returncode not in {0, 1}:
            return CommitRelation.UNKNOWN
        if base_is_ancestor.returncode == 0:
            return CommitRelation.AHEAD
        if head_is_ancestor.returncode == 0:
            return CommitRelation.BEHIND
        return CommitRelation.DIVERGED

    def _count_revisions(self, revision_range: str, *, cwd: Path) -> int:
        """计算一个 Git revision range 中的提交数量。

        Args:
            revision_range: 例如 ``base..HEAD`` 或 ``upstream..HEAD``。
            cwd: 执行计数的 Worktree 目录。

        Returns:
            Git 返回的非负提交数。

        Raises:
            WorktreeGitError: revision range 无法解析或输出不是整数。
        """

        output = self._run(
            ["rev-list", "--count", revision_range],
            purpose="统计 Worktree 提交",
            cwd=cwd,
        )
        try:
            count = int(output.strip())
        except ValueError as exc:
            raise WorktreeGitError("统计 Worktree 提交", None, "Git 返回了非整数") from exc
        if count < 0:
            raise WorktreeGitError("统计 Worktree 提交", None, "Git 返回了负数")
        return count

    def remove(self, path: Path, *, force: bool = False) -> None:
        """从 Git 登记和磁盘中移除一个 Worktree 目录。

        Args:
            path: 要移除的绝对 Worktree 路径。
            force: 是否显式允许 Git 丢弃未提交内容。

        Returns:
            Git 成功移除目录时不返回数据。

        Raises:
            ValueError: 路径或 ``force`` 类型无效。
            WorktreeGitError: Git 拒绝移除或命令失败。
        """

        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("Worktree remove path 必须是绝对 Path")
        if not isinstance(force, bool):
            raise ValueError("Worktree remove force 必须是布尔值")
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(os.fspath(path))
        self._run(args, purpose="移除 Git Worktree")

    def delete_branch(self, branch: str, *, force: bool = False) -> None:
        """删除一个已经不被 Worktree 使用的本地分支。

        Args:
            branch: 要删除的完整本地分支名。
            force: 是否使用 ``-D`` 显式丢弃未合并提交；默认使用 ``-d``。

        Returns:
            Git 成功删除分支时不返回数据。

        Raises:
            ValueError: 分支名为空或 ``force`` 类型无效。
            WorktreeGitError: 分支仍在使用、未合并或 Git 命令失败。
        """

        if not isinstance(branch, str) or not branch.strip():
            raise ValueError("delete_branch branch 必须是非空字符串")
        if not isinstance(force, bool):
            raise ValueError("delete_branch force 必须是布尔值")
        self._run(
            ["branch", "-D" if force else "-d", branch],
            purpose="删除 Worktree 分支",
        )

    def is_branch_merged(self, branch: str, base_ref: str) -> bool | None:
        """检查一个保留分支的 tip 是否已被基准引用包含。

        Args:
            branch: 等待删除的本地分支名。
            base_ref: 创建记录保存的基准分支或 commit。

        Returns:
            已合入返回 ``True``，确认未合入返回 ``False``，Git 无法可靠判断时
            返回 ``None``。

        Raises:
            ValueError: 分支或基准引用为空。
        """

        if not isinstance(branch, str) or not branch.strip():
            raise ValueError("is_branch_merged branch 必须是非空字符串")
        if not isinstance(base_ref, str) or not base_ref.strip():
            raise ValueError("is_branch_merged base_ref 必须是非空字符串")
        result = self._run_result(
            ["merge-base", "--is-ancestor", branch, base_ref],
            purpose="检查保留分支是否已合入基准",
        )
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        return None

    def configure_hooks(self, worktree_path: Path, hooks_path: str) -> None:
        """只为目标 Worktree 写入显式 ``core.hooksPath``。

        Args:
            worktree_path: 已创建并登记的 Worktree 绝对目录。
            hooks_path: 配置中批准的项目内相对 Hooks 路径。

        Returns:
            Worktree 专属配置成功写入时不返回数据。

        Raises:
            ValueError: 路径或 Hooks 文本无效。
            WorktreeGitError: 本机 Git 不支持 worktreeConfig 或写配置失败。
        """

        if not isinstance(worktree_path, Path) or not worktree_path.is_absolute():
            raise ValueError("configure_hooks worktree_path 必须是绝对 Path")
        if not isinstance(hooks_path, str) or not hooks_path.strip():
            raise ValueError("configure_hooks hooks_path 必须是非空字符串")
        self._run(
            ["config", "extensions.worktreeConfig", "true"],
            purpose="启用 Git Worktree 专属配置",
        )
        self._run(
            ["config", "--worktree", "core.hooksPath", hooks_path],
            purpose="配置 Worktree Hooks",
            cwd=worktree_path,
        )

    def list_ignored_paths(self) -> tuple[str, ...]:
        """列出主仓库中实际存在、被标准 Git ignore 规则忽略的路径。

        Returns:
            使用正斜杠、按 Git 输出顺序排列的项目内相对路径元组。NUL 分隔
            保证含空格的文件名不会被拆开。

        Raises:
            WorktreeGitError: ``git ls-files`` 无法读取 ignored 路径。
        """

        output = self._run(
            ["ls-files", "--others", "--ignored", "--exclude-standard", "-z"],
            purpose="列出主仓库 ignored 路径",
        )
        return tuple(path for path in output.split("\0") if path)
