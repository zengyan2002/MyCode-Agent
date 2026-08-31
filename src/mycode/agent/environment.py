"""采集可安全注入模型的工作区、操作系统、时间和 Git 摘要。

给模型 Git 状态，是为了提醒它当前工作区是否已有修改、是否发生了外部变化，
从而更谨慎地处理代码；它只是上下文提示，不能代替真正的文件安全机制。
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from mycode.constants import GIT_STATUS_TIMEOUT_SECONDS
from mycode.worktrees.binding import WorkspaceBinding, shared_workspace_binding

class GitAvailability(str, Enum):
    """
    描述Git的状态
    """
    # Git可用，而且当前目录是 Git 仓库
    AVAILABLE = "available"
    # 系统中没有找到 Git 命令
    COMMAND_MISSING = "command_missing"
    # 当前工作区不是 Git 仓库
    NOT_REPOSITORY = "not_repository"
    # 执行 git status 超时
    TIMEOUT = "timeout"
    # Git 状态读取失败
    ERROR = "error"


@dataclass(frozen=True)
class GitSnapshot:
    """
    某一时刻采集到的 Git 状态快照
    """
    # 表示Git当前是否可用
    availability: GitAvailability
    # 当前的分支名称
    branch: str | None = None
    # git status --porcelain 返回的变更记录
    entries: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.availability is not GitAvailability.AVAILABLE and (
            self.branch is not None or self.entries
        ):
            raise ValueError("Git 不可用快照不能携带仓库状态")

    @property
    def summary(self) -> str:
        """
        把结构化的 Git 状态转换为一段简短的人类可读文本。
        """
        if self.availability is not GitAvailability.AVAILABLE:
            labels = {
                GitAvailability.COMMAND_MISSING: "Git 命令不可用",
                GitAvailability.NOT_REPOSITORY: "当前目录不是 Git 仓库",
                GitAvailability.TIMEOUT: "Git 状态读取超时",
                GitAvailability.ERROR: "Git 状态读取失败",
            }
            return labels[self.availability]
        if not self.entries:
            return "工作区干净"
        categories: dict[str, int] = {}
        for entry in self.entries:
            code = entry[:2].strip() or "?"
            categories[code] = categories.get(code, 0) + 1
        details = "、".join(f"{code}={count}" for code, count in sorted(categories.items()))
        return f"工作区有 {len(self.entries)} 个变更（{details}）"

def read_git_snapshot(workspace_root: Path) -> GitSnapshot:
    """
    在指定工作区执行一次只读的 git status，读取当前分支和文件变更，并把成功或失败情况统一封装成 GitSnapshot。
    """
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                os.fspath(workspace_root),
                "status",
                "--porcelain=v1",
                "--branch",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GIT_STATUS_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        return GitSnapshot(GitAvailability.COMMAND_MISSING)
    except subprocess.TimeoutExpired:
        return GitSnapshot(GitAvailability.TIMEOUT)
    except OSError:
        return GitSnapshot(GitAvailability.ERROR)
    #git命令查找失败
    if result.returncode != 0:
        #判断当前目录是否是git仓库
        if "not a git repository" in result.stderr.lower():
            return GitSnapshot(GitAvailability.NOT_REPOSITORY)
        return GitSnapshot(GitAvailability.ERROR)
    lines = result.stdout.splitlines()
    branch: str | None = None
    entries: list[str] = []
    for line in lines:
        if line.startswith("## "):
            raw = line[3:]
            branch = raw.split("...", 1)[0].strip() or None
        #如果不是空行且不是描述分支行，则是一条文件变更
        elif line:
            entries.append(line)
    #包装成GitSnapshot并返回
    return GitSnapshot(GitAvailability.AVAILABLE, branch, tuple(entries))


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """
    某一时刻采集到的环境快照
    """
    #工作区
    workspace: str
    #操作系统
    operating_system: str
    #当前时间
    collected_at: str
    #git快照
    git: GitSnapshot

    def render(self) -> str:
        branch = self.git.branch or "不可用"
        return (
            f"工作目录：{self.workspace}\n"
            f"操作系统：{self.operating_system}\n"
            f"当前时间：{self.collected_at}\n"
            f"Git 分支：{branch}\n"
            f"Git 状态：{self.git.summary}"
        )


@dataclass(frozen=True)
class EnvironmentChange:
    """
    保存环境发生变化的字段和最新值，并负责把这些变化转换成可追加到模型提示词中的多行文本。
    """
    #保存变化后的字段
    fields: tuple[tuple[str, str], ...] = ()

    #标志环境是否发生变化
    @property
    def changed(self) -> bool:
        return bool(self.fields)

    #变化字段列表转换成适合放进提示词的文本
    def render(self) -> str:
        return "\n".join(f"{name}：{value}" for name, value in self.fields)


def compare_environment(
    previous: EnvironmentSnapshot,
    current: EnvironmentSnapshot,
) -> EnvironmentChange:
    """
    比较两次环境快照，返回发生变化的字段及其最新值
    """
    fields: list[tuple[str, str]] = []
    if previous.workspace != current.workspace:
        fields.append(("工作目录", current.workspace))
    if previous.operating_system != current.operating_system:
        fields.append(("操作系统", current.operating_system))
    if previous.git.availability != current.git.availability:
        fields.append(("Git 可用状态", current.git.availability.value))
    if previous.git.branch != current.git.branch:
        fields.append(("Git 分支", current.git.branch or "不可用"))
    if previous.git.entries != current.git.entries:
        fields.append(("Git 状态", current.git.summary))
    # 先变成不可变元组，再包装成EnvironmentChange类的对象
    return EnvironmentChange(tuple(fields))


class EnvironmentCollector:
    """从工作区绑定采集模型下一次请求需要看到的环境快照。

    Attributes:
        _workspace: 主 Agent 的可变绑定或子 Agent 的固定绑定。每次采集只读取
            一次快照，工作区路径和 Git 状态因此来自同一个目录。
    """

    def __init__(
        self,
        workspace: WorkspaceBinding | Path,
    ) -> None:
        """创建环境采集器。

        Args:
            workspace: 当前 Agent 的绑定。为兼容现有调用，也接受绝对路径并
                立即转换成共享绑定。

        Returns:
            新的环境采集器。

        Raises:
            ValueError: ``workspace`` 不是绑定或绝对路径。
        """

        if isinstance(workspace, Path):
            if not workspace.is_absolute():
                raise ValueError("环境工作区根目录必须是绝对路径")
            self._workspace = shared_workspace_binding(workspace)
        elif isinstance(workspace, WorkspaceBinding):
            self._workspace = workspace
        else:
            raise ValueError("环境 workspace 必须是 WorkspaceBinding 或绝对 Path")

    def collect(self) -> EnvironmentSnapshot:
        """采集当前绑定的一次工作目录、系统、时间和 Git 状态。

        Returns:
            所有工作区相关字段都来自同一绑定快照的 ``EnvironmentSnapshot``。
        """

        assignment = self._workspace.snapshot()
        workspace_root = assignment.root
        collected = datetime.now(timezone.utc).astimezone()
        return EnvironmentSnapshot(
            workspace=os.fspath(workspace_root),
            operating_system=f"{platform.system()} {platform.release()}".strip(),
            collected_at=collected.isoformat(timespec="seconds"),
            git=read_git_snapshot(workspace_root),
        )
