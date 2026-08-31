"""在不修改进程 cwd 的前提下保存 Agent 当前使用的工作目录。"""

from __future__ import annotations

from threading import RLock

from pathlib import Path

from mycode.models.worktrees import (
    WorkspaceAssignment,
    WorkspaceIsolationMode,
)


def shared_workspace_binding(
    root: Path,
    *,
    branch: str | None = None,
    base_commit: str = "unresolved",
) -> WorkspaceBinding:
    """为主仓库或兼容调用方创建一个可切换的共享绑定。

    Args:
        root: 工具和环境采集使用的工作目录，必须能解析成绝对路径。
        branch: 创建绑定时已知的本地分支；尚未读取 Git 时为 ``None``。
        base_commit: 创建绑定时已解析的本地 HEAD SHA。只构造工具测试上下文、
            尚未读取 Git 时可以使用默认的 ``unresolved`` 标记。

    Returns:
        初始分配指向 ``root`` 的可变 ``WorkspaceBinding``。

    Raises:
        ValueError: ``root`` 不是 ``Path``，或分支、基线字段无效。
    """

    if not isinstance(root, Path):
        raise ValueError("共享工作区 root 必须是 Path")
    assignment = WorkspaceAssignment(
        root=root.resolve(),
        isolation=WorkspaceIsolationMode.SHARED,
        worktree_name=None,
        branch=branch,
        base_commit=base_commit,
    )
    return WorkspaceBinding(assignment)


class WorkspaceBinding:
    """保存主 Agent 的可切换工作区，或子 Agent 的固定工作区。

    工具每次执行前调用 :meth:`snapshot`，拿到一个不可变的
    :class:`WorkspaceAssignment`。主 Agent 可以在会话命令完成后原子替换分配；
    子 Agent 使用固定绑定，防止运行途中被其他并发任务切换目录。

    Attributes:
        _assignment: 当前所有新工具调用应使用的不可变工作区分配。
        _initial: 调用 :meth:`reset` 时恢复的主仓库分配。
        _fixed: ``True`` 表示这是子 Agent 的固定绑定，不允许切换或重置。
        _lock: 只保护分配引用的短同步锁，不覆盖 Git 或文件系统操作。
    """

    def __init__(
        self,
        assignment: WorkspaceAssignment,
        *,
        fixed: bool = False,
    ) -> None:
        """创建工作区绑定。

        Args:
            assignment: 创建后立即生效的工作区分配，也作为可变绑定的重置目标。
            fixed: 是否禁止后续 ``bind`` 和 ``reset``。子 Agent 应传 ``True``。

        Returns:
            新的工作区绑定对象。

        Raises:
            ValueError: ``assignment`` 不是 ``WorkspaceAssignment``，或 ``fixed``
                不是布尔值。
        """

        if not isinstance(assignment, WorkspaceAssignment):
            raise ValueError("WorkspaceBinding.assignment 类型无效")
        if not isinstance(fixed, bool):
            raise ValueError("WorkspaceBinding.fixed 必须是布尔值")
        self._assignment = assignment
        self._initial = assignment
        self._fixed = fixed
        self._lock = RLock()

    @classmethod
    def fixed(cls, assignment: WorkspaceAssignment) -> WorkspaceBinding:
        """为一次子 Agent 运行创建不可切换的绑定。

        Args:
            assignment: 子 Agent 在完整运行期间使用的工作区分配。

        Returns:
            拒绝 ``bind`` 和 ``reset`` 的新绑定。

        Raises:
            ValueError: ``assignment`` 类型无效。
        """

        return cls(assignment, fixed=True)

    @property
    def is_fixed(self) -> bool:
        """说明当前绑定是否禁止切换。

        Returns:
            子 Agent 固定绑定返回 ``True``，主 Agent 可变绑定返回 ``False``。
        """

        return self._fixed

    def snapshot(self) -> WorkspaceAssignment:
        """读取一次工具调用应使用的不可变工作区分配。

        Returns:
            进入短锁时当前生效的 ``WorkspaceAssignment``。后续切换不会改写
            已经返回的对象。
        """

        with self._lock:
            return self._assignment

    def bind(self, assignment: WorkspaceAssignment) -> WorkspaceAssignment:
        """把主 Agent 后续工具调用切换到新的工作区。

        Args:
            assignment: 已由 WorktreeManager 验证并准备完成的新分配。

        Returns:
            切换前的工作区分配，调用方可以用它生成日志或执行回滚。

        Raises:
            ValueError: 新分配类型无效。
            RuntimeError: 当前绑定是子 Agent 的固定绑定。
        """

        if not isinstance(assignment, WorkspaceAssignment):
            raise ValueError("WorkspaceBinding.bind assignment 类型无效")
        with self._lock:
            if self._fixed:
                raise RuntimeError("固定工作区绑定不能切换")
            previous = self._assignment
            self._assignment = assignment
            return previous

    def reset(self) -> WorkspaceAssignment:
        """把主 Agent 后续工具调用恢复到创建绑定时的主仓库。

        Returns:
            重置前的工作区分配。

        Raises:
            RuntimeError: 当前绑定是子 Agent 的固定绑定。
        """

        with self._lock:
            if self._fixed:
                raise RuntimeError("固定工作区绑定不能重置")
            previous = self._assignment
            self._assignment = self._initial
            return previous
