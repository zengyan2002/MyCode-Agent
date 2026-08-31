"""定义 Git Worktree 隔离功能在各模块之间传递的数据。

本模块只保存事实，不执行 Git 命令，也不决定是否删除目录。这样状态存储、
生命周期管理器和子 Agent 运行器可以共享同一组值对象，而不形成循环依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path


class WorkspaceIsolationMode(str, Enum):
    """说明子 Agent 使用独立 Worktree，还是与调用方共用当前工作目录。"""

    WORKTREE = "worktree"
    SHARED = "shared"


class WorktreeKind(str, Enum):
    """说明 Worktree 是用户长期管理的目录，还是一次子任务的临时目录。"""

    MANUAL = "manual"
    TASK = "task"


class WorktreeLifecycle(str, Enum):
    """说明一条受管 Worktree 记录目前处于哪个生命周期阶段。"""

    CREATING = "creating"
    READY = "ready"
    RETAINED = "retained"
    INTERRUPTED = "interrupted"
    PRUNED = "pruned"
    REMOVING = "removing"
    ERROR = "error"


class WorktreeTaskState(str, Enum):
    """说明拥有临时 Worktree 的子任务是否仍可能使用该目录。"""

    QUEUED = "queued"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    FINISHED = "finished"


class CommitRelation(str, Enum):
    """说明 Worktree 当前 HEAD 与创建基线之间的提交关系。"""

    SAME = "same"
    AHEAD = "ahead"
    BEHIND = "behind"
    DIVERGED = "diverged"
    UNKNOWN = "unknown"


class WorktreeTaskOutcome(str, Enum):
    """记录子 Agent 结束时的业务终态，避免 Worktree 模型依赖 Agent 模型。"""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class WorktreeFinishAction(str, Enum):
    """说明一次子任务结束后，系统实际如何处理它的工作目录。"""

    SHARED_RELEASED = "shared_released"
    DELETED = "deleted"
    RETAINED = "retained"
    PRUNED = "pruned"
    SKIPPED = "skipped"


class InitializationActionStatus(str, Enum):
    """说明一项创建后初始化动作是完成了、被跳过了，还是失败了。"""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    WARNING = "warning"
    FAILED = "failed"


def _require_text(value: str, field_name: str) -> None:
    """检查持久化模型中的必填文本。

    Args:
        value: 调用方准备写入字段的字符串。
        field_name: 出错时显示给开发者看的字段名。

    Returns:
        文本非空时不返回数据。

    Raises:
        ValueError: ``value`` 不是字符串，或去掉首尾空白后为空。
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 必须是非空字符串")


def _require_absolute_path(value: Path, field_name: str) -> None:
    """检查模型中的路径已经由调用方解析成绝对路径。

    Args:
        value: 要保存到模型中的路径。
        field_name: 出错时显示给开发者看的字段名。

    Returns:
        路径是绝对 ``Path`` 时不返回数据。

    Raises:
        ValueError: 路径类型错误或仍是相对路径。
    """

    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError(f"{field_name} 必须是绝对 Path")


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    """检查状态时间带有时区，避免重启后误判过期时间。

    Args:
        value: 要持久化的时间。
        field_name: 出错时显示给开发者看的字段名。

    Returns:
        时间包含有效时区偏移时不返回数据。

    Raises:
        ValueError: 值不是 ``datetime`` 或没有时区信息。
    """

    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} 必须是带时区的 datetime")


@dataclass(frozen=True)
class WorktreeTaskOwner:
    """记录一个临时 Worktree 属于哪个主会话和哪次子任务。

    Attributes:
        session_id: 发起子任务的主会话 ID，恢复时用它查找遗留任务。
        task_id: TaskManager 分配的任务 ID；任务尚未入队时可以为空。
        origin: 创建任务的入口，例如 ``agent``、``fork``、``skill`` 或 ``team``。
        team_id: 长期团队成员所属团队；普通子任务为 ``None``。
        agent_id: 长期团队成员 ID；普通子任务为 ``None``。
    """

    session_id: str
    task_id: str | None
    origin: str
    team_id: str | None = None
    agent_id: str | None = None

    def __post_init__(self) -> None:
        """校验任务归属字段。

        Returns:
            会话、可选任务 ID 和来源均合法时不返回数据。

        Raises:
            ValueError: 必填文本为空，或可选任务 ID 不是非空字符串。
        """

        _require_text(self.session_id, "WorktreeTaskOwner.session_id")
        _require_text(self.origin, "WorktreeTaskOwner.origin")
        if self.task_id is not None:
            _require_text(self.task_id, "WorktreeTaskOwner.task_id")
        if (self.team_id is None) != (self.agent_id is None):
            raise ValueError("团队 Worktree owner 必须同时提供 team_id 和 agent_id")
        if self.team_id is not None:
            _require_text(self.team_id, "WorktreeTaskOwner.team_id")
            _require_text(self.agent_id or "", "WorktreeTaskOwner.agent_id")


@dataclass(frozen=True)
class WorktreeRecord:
    """表示磁盘状态文件中一条由 MyCode 管理的 Worktree 记录。

    Attributes:
        name: 用户或系统使用的原始 slug，例如 ``team-refactor/alice``。
        path: Worktree 实际所在的绝对目录。
        branch: 该 Worktree 检出的本地分支名。
        base_ref: 创建时调用方选择的基准引用，通常是父工作区分支名。
        base_commit: 创建时解析出的本地基线 commit SHA。
        kind: 手工目录或临时任务目录。
        lifecycle: 目录当前处于创建、可用、保留、移除等哪个阶段。
        owner: 临时目录的主会话和任务归属；手工目录为 ``None``。
        owner_pid: 创建或使用目录的 MyCode 进程 ID；未知时为 ``None``。
        task_state: 临时任务的排队、运行或结束状态；手工目录为 ``None``。
        created_at: 首次预留名称的时间，必须带时区。
        last_used_at: 最近一次绑定或检查目录的时间，必须带时区。
        initialization_complete: 创建后配置、依赖和 Hooks 初始化是否全部结束。
        warnings: 可选初始化动作或恢复检查产生的简短警告，不含文件正文。
    """

    name: str
    path: Path
    branch: str
    base_ref: str
    base_commit: str
    kind: WorktreeKind
    lifecycle: WorktreeLifecycle
    owner: WorktreeTaskOwner | None
    owner_pid: int | None
    task_state: WorktreeTaskState | None
    created_at: datetime
    last_used_at: datetime
    initialization_complete: bool
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """校验一条记录可以安全地写入状态文件。

        Returns:
            字段类型、路径和时间都合法时不返回数据。

        Raises:
            ValueError: 发现空字段、相对路径、无时区时间或错误枚举类型。
        """

        for field_name, value in (
            ("name", self.name),
            ("branch", self.branch),
            ("base_ref", self.base_ref),
            ("base_commit", self.base_commit),
        ):
            _require_text(value, f"WorktreeRecord.{field_name}")
        _require_absolute_path(self.path, "WorktreeRecord.path")
        if not isinstance(self.kind, WorktreeKind):
            raise ValueError("WorktreeRecord.kind 类型无效")
        if not isinstance(self.lifecycle, WorktreeLifecycle):
            raise ValueError("WorktreeRecord.lifecycle 类型无效")
        if self.owner is not None and not isinstance(self.owner, WorktreeTaskOwner):
            raise ValueError("WorktreeRecord.owner 类型无效")
        if self.owner_pid is not None and (
            isinstance(self.owner_pid, bool)
            or not isinstance(self.owner_pid, int)
            or self.owner_pid <= 0
        ):
            raise ValueError("WorktreeRecord.owner_pid 必须是正整数或 None")
        if self.task_state is not None and not isinstance(
            self.task_state, WorktreeTaskState
        ):
            raise ValueError("WorktreeRecord.task_state 类型无效")
        _require_aware_datetime(self.created_at, "WorktreeRecord.created_at")
        _require_aware_datetime(self.last_used_at, "WorktreeRecord.last_used_at")
        if self.last_used_at < self.created_at:
            raise ValueError("WorktreeRecord.last_used_at 不能早于 created_at")
        if not isinstance(self.initialization_complete, bool):
            raise ValueError("WorktreeRecord.initialization_complete 必须是布尔值")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(item, str) and item.strip() for item in self.warnings
        ):
            raise ValueError("WorktreeRecord.warnings 必须是非空字符串元组")


@dataclass(frozen=True)
class WorkspaceAssignment:
    """表示一次 Agent 运行实际应该使用的固定工作目录。

    Attributes:
        root: 所有相对文件操作和命令执行使用的绝对根目录。
        isolation: 这次运行使用独立 Worktree 还是共享调用方目录。
        worktree_name: 独立模式下对应的受管 slug；共享模式下为 ``None``。
        branch: 生成环境说明和结果报告时显示的本地分支名。
        base_commit: 冻结该分配时的本地 HEAD commit SHA。
        lease_id: Manager 为独立目录签发的占用凭据；共享模式下为 ``None``。
        parent_had_changes: 创建独立目录时，父工作区是否存在未提交内容。
    """

    root: Path
    isolation: WorkspaceIsolationMode
    worktree_name: str | None
    branch: str | None
    base_commit: str
    lease_id: str | None = None
    parent_had_changes: bool = False

    def __post_init__(self) -> None:
        """校验工作区分配中的路径、模式和关联字段。

        Returns:
            分配可交给工具或运行器使用时不返回数据。

        Raises:
            ValueError: 路径、模式、基线或独立/共享关联字段不合法。
        """

        _require_absolute_path(self.root, "WorkspaceAssignment.root")
        if not isinstance(self.isolation, WorkspaceIsolationMode):
            raise ValueError("WorkspaceAssignment.isolation 类型无效")
        _require_text(self.base_commit, "WorkspaceAssignment.base_commit")
        if self.branch is not None:
            _require_text(self.branch, "WorkspaceAssignment.branch")
        if not isinstance(self.parent_had_changes, bool):
            raise ValueError("WorkspaceAssignment.parent_had_changes 必须是布尔值")
        if self.isolation is WorkspaceIsolationMode.WORKTREE:
            if self.worktree_name is None or self.lease_id is None:
                raise ValueError("独立工作区必须包含 worktree_name 和 lease_id")
            _require_text(self.worktree_name, "WorkspaceAssignment.worktree_name")
            _require_text(self.lease_id, "WorkspaceAssignment.lease_id")
        elif self.worktree_name is not None or self.lease_id is not None:
            raise ValueError("共享工作区不能包含 worktree_name 或 lease_id")


@dataclass(frozen=True)
class WorkspaceResolution:
    """返回恢复或准备工作区后的分配，以及需要展示给用户的警告。

    Attributes:
        assignment: 工具和 Agent 后续实际使用的工作区分配。
        warnings: 恢复降级、父目录有修改等需要显示的简短说明。
    """

    assignment: WorkspaceAssignment
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """校验工作区解析结果。

        Returns:
            分配和警告字段合法时不返回数据。

        Raises:
            ValueError: 分配类型错误或警告不是非空字符串元组。
        """

        if not isinstance(self.assignment, WorkspaceAssignment):
            raise ValueError("WorkspaceResolution.assignment 类型无效")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(item, str) and item.strip() for item in self.warnings
        ):
            raise ValueError("WorkspaceResolution.warnings 必须是非空字符串元组")


@dataclass(frozen=True)
class WorktreeChangeSummary:
    """汇总 Worktree 中尚未提交的文件和相对创建基线的提交变化。

    Attributes:
        staged: 已加入暂存区的项目内相对路径。
        unstaged: 已修改但未加入暂存区的项目内相对路径。
        untracked: Git 尚未追踪的项目内相对路径。
        head_commit: 检查时 Worktree HEAD 的 commit SHA。
        relation_to_base: HEAD 与创建基线是相同、领先、落后、分叉还是未知。
        new_commit_count: 能确认的 ``base..HEAD`` 提交数；未知关系时仍可为零。
        unpushed_commit_count: 尚未进入 upstream 的提交数；无法确认时为 ``None``。
        merged_into_base: 当前分支是否已被基准引用包含；无法确认时为 ``None``。
    """

    staged: tuple[str, ...]
    unstaged: tuple[str, ...]
    untracked: tuple[str, ...]
    head_commit: str
    relation_to_base: CommitRelation
    new_commit_count: int
    unpushed_commit_count: int | None
    merged_into_base: bool | None

    def __post_init__(self) -> None:
        """校验 Git 变更摘要中的路径、计数和三态判断。

        Returns:
            摘要可以用于删除保护判断时不返回数据。

        Raises:
            ValueError: 路径、SHA、枚举、计数或三态字段类型错误。
        """

        for field_name in ("staged", "unstaged", "untracked"):
            value = getattr(self, field_name)
            if not isinstance(value, tuple) or not all(
                isinstance(item, str) and item for item in value
            ):
                raise ValueError(f"WorktreeChangeSummary.{field_name} 类型无效")
        _require_text(self.head_commit, "WorktreeChangeSummary.head_commit")
        if not isinstance(self.relation_to_base, CommitRelation):
            raise ValueError("WorktreeChangeSummary.relation_to_base 类型无效")
        for field_name in ("new_commit_count", "unpushed_commit_count"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"WorktreeChangeSummary.{field_name} 必须是非负整数或 None")
        if self.merged_into_base is not None and not isinstance(
            self.merged_into_base, bool
        ):
            raise ValueError("WorktreeChangeSummary.merged_into_base 必须是布尔值或 None")

    @property
    def has_file_changes(self) -> bool:
        """判断是否存在暂存、未暂存或未追踪文件。

        Returns:
            三类路径中任意一类非空时返回 ``True``，否则返回 ``False``。
        """

        return bool(self.staged or self.unstaged or self.untracked)


@dataclass(frozen=True)
class WorktreeFinishReport:
    """记录子 Agent 结束后工作区的检查结果和实际处置动作。

    Attributes:
        workspace: 子 Agent 运行时使用的固定工作区分配。
        action: 系统最终释放共享绑定、删除、保留或跳过了什么。
        terminal_status: 子 Agent 完成、失败、取消或中断的终态。
        changes: Git 变更检查结果；共享目录或检查失败时可以为 ``None``。
        reason: 选择该处置动作的可读原因。
        warnings: 收尾过程中出现但未覆盖主要结果的警告。
    """

    workspace: WorkspaceAssignment
    action: WorktreeFinishAction
    terminal_status: WorktreeTaskOutcome
    changes: WorktreeChangeSummary | None
    reason: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """校验工作区收尾报告可以直接显示或持久化。

        Returns:
            字段类型和文本合法时不返回数据。

        Raises:
            ValueError: 分配、动作、终态、摘要或文字字段不合法。
        """

        if not isinstance(self.workspace, WorkspaceAssignment):
            raise ValueError("WorktreeFinishReport.workspace 类型无效")
        if not isinstance(self.action, WorktreeFinishAction):
            raise ValueError("WorktreeFinishReport.action 类型无效")
        if not isinstance(self.terminal_status, WorktreeTaskOutcome):
            raise ValueError("WorktreeFinishReport.terminal_status 类型无效")
        if self.changes is not None and not isinstance(
            self.changes, WorktreeChangeSummary
        ):
            raise ValueError("WorktreeFinishReport.changes 类型无效")
        _require_text(self.reason, "WorktreeFinishReport.reason")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(item, str) and item.strip() for item in self.warnings
        ):
            raise ValueError("WorktreeFinishReport.warnings 必须是非空字符串元组")


@dataclass(frozen=True)
class WorktreeStateSnapshot:
    """表示一次完整、可信、可原子写入磁盘的 Worktree 状态。

    Attributes:
        version: 状态文件格式版本，目前固定为 1。
        revision: Manager 每次成功修改状态后递增的修订号。
        records: 所有仍由 MyCode 管理的 Worktree 记录。
        session_bindings: ``(session_id, worktree_name)`` 对；没有条目表示主仓库。
    """

    version: int = 1
    revision: int = 0
    records: tuple[WorktreeRecord, ...] = ()
    session_bindings: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """校验状态版本、修订号和名称唯一性。

        Returns:
            快照可以序列化时不返回数据。

        Raises:
            ValueError: 版本、修订号、记录或会话映射不合法或重复。
        """

        if self.version != 1:
            raise ValueError("Worktree 状态版本必须是 1")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise ValueError("Worktree 状态 revision 必须是非负整数")
        if not isinstance(self.records, tuple) or not all(
            isinstance(item, WorktreeRecord) for item in self.records
        ):
            raise ValueError("Worktree 状态 records 类型无效")
        names = [item.name for item in self.records]
        if len(names) != len(set(names)):
            raise ValueError("Worktree 状态包含重复名称")
        if not isinstance(self.session_bindings, tuple):
            raise ValueError("Worktree 状态 session_bindings 必须是元组")
        session_ids: list[str] = []
        known_names = set(names)
        for pair in self.session_bindings:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("Worktree 会话绑定必须是二元组")
            session_id, worktree_name = pair
            _require_text(session_id, "Worktree session_id")
            _require_text(worktree_name, "Worktree binding name")
            if worktree_name not in known_names:
                raise ValueError("Worktree 会话绑定引用了不存在的记录")
            session_ids.append(session_id)
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("Worktree 状态包含重复会话绑定")


@dataclass(frozen=True)
class WorktreeStateLoadResult:
    """返回状态文件加载结果，并明确区分空状态和损坏状态。

    Attributes:
        snapshot: 成功读取的快照；文件损坏或版本未知时为 ``None``。
        trusted: Manager 是否可以依据该状态执行删除等破坏性动作。
        error: 读取失败的具体原因；成功时为 ``None``。
    """

    snapshot: WorktreeStateSnapshot | None
    trusted: bool
    error: str | None = None

    def __post_init__(self) -> None:
        """校验加载结果不会把损坏状态伪装成空状态。

        Returns:
            成功和失败字段互相一致时不返回数据。

        Raises:
            ValueError: 可信结果没有快照，或失败结果没有错误说明。
        """

        if self.trusted:
            if not isinstance(self.snapshot, WorktreeStateSnapshot) or self.error is not None:
                raise ValueError("可信 Worktree 状态必须只有 snapshot")
        elif self.snapshot is not None or not self.error:
            raise ValueError("不可信 Worktree 状态必须只有 error")


@dataclass(frozen=True)
class WorktreeHeadPrecheck:
    """保存纯文件系统读取 ``.git`` 指针和 admin ``HEAD`` 的快速结果。

    Attributes:
        matched: 指针和分支明显匹配时为 ``True``，明显冲突时为 ``False``，
            文件缺失等无法判断时为 ``None``。
        admin_dir: 指针解析出的 Git Worktree admin 绝对目录；无法解析时为
            ``None``。
        head_ref: admin ``HEAD`` 中读取的完整符号引用；detached 或不可读时为
            ``None``。
        reason: 结果的简短说明。预检通过后仍需 Git 后端做最终验证。
    """

    matched: bool | None
    admin_dir: Path | None
    head_ref: str | None
    reason: str

    def __post_init__(self) -> None:
        """校验快速预检结果的三态和可选路径。

        Returns:
            结果字段合法时不返回数据。

        Raises:
            ValueError: 三态、绝对路径、引用或原因字段无效。
        """

        if self.matched is not None and not isinstance(self.matched, bool):
            raise ValueError("WorktreeHeadPrecheck.matched 类型无效")
        if self.admin_dir is not None:
            _require_absolute_path(self.admin_dir, "WorktreeHeadPrecheck.admin_dir")
        if self.head_ref is not None:
            _require_text(self.head_ref, "WorktreeHeadPrecheck.head_ref")
        _require_text(self.reason, "WorktreeHeadPrecheck.reason")


@dataclass(frozen=True)
class GitHead:
    """保存主仓库一次本地 HEAD 解析得到的分支和提交。

    Attributes:
        branch: 当前符号分支名；detached HEAD 时为 ``None``。
        commit: 当前 HEAD 的完整 commit SHA。
    """

    branch: str | None
    commit: str

    def __post_init__(self) -> None:
        """校验本地 HEAD 信息。

        Returns:
            提交非空且可选分支合法时不返回数据。

        Raises:
            ValueError: 提交为空或分支是空字符串。
        """

        _require_text(self.commit, "GitHead.commit")
        if self.branch is not None:
            _require_text(self.branch, "GitHead.branch")


@dataclass(frozen=True)
class GitWorktreeEntry:
    """表示 ``git worktree list --porcelain`` 中一个工作目录条目。

    Attributes:
        path: Git 登记的 Worktree 绝对路径。
        head_commit: 该目录当前 HEAD commit SHA。
        branch: 去掉 ``refs/heads/`` 后的本地分支；detached 时为 ``None``。
        bare: 该条目是否为 bare 仓库。
        detached: 该条目是否处于 detached HEAD。
        prunable: Git 是否报告该条目可以 prune。
    """

    path: Path
    head_commit: str
    branch: str | None
    bare: bool = False
    detached: bool = False
    prunable: bool = False

    def __post_init__(self) -> None:
        """校验 Git Worktree 条目。

        Returns:
            路径、提交、可选分支和标志位合法时不返回数据。

        Raises:
            ValueError: 条目包含相对路径、空文本或非布尔标志。
        """

        _require_absolute_path(self.path, "GitWorktreeEntry.path")
        _require_text(self.head_commit, "GitWorktreeEntry.head_commit")
        if self.branch is not None:
            _require_text(self.branch, "GitWorktreeEntry.branch")
        if not all(isinstance(value, bool) for value in (self.bare, self.detached, self.prunable)):
            raise ValueError("GitWorktreeEntry 标志必须是布尔值")


@dataclass(frozen=True)
class WorktreeName:
    """保存通过 slug 校验后得到的目录和分支名称。

    Attributes:
        original: 用户或系统提供的原始 slug。
        flat: 把斜杠替换成加号后的文件系统名称。
        path: ``.mycode/worktrees`` 下经过边界检查的绝对目录。
        branch: 使用固定前缀构造并通过 Git 校验前的本地分支名。
    """

    original: str
    flat: str
    path: Path
    branch: str

    def __post_init__(self) -> None:
        """校验名称转换结果包含完整文本和绝对目录。

        Returns:
            四个字段合法时不返回数据。

        Raises:
            ValueError: 任一文本为空或路径不是绝对路径。
        """

        for field_name in ("original", "flat", "branch"):
            _require_text(getattr(self, field_name), f"WorktreeName.{field_name}")
        _require_absolute_path(self.path, "WorktreeName.path")


@dataclass(frozen=True)
class InitializationAction:
    """记录创建后初始化中的一项可观察动作。

    Attributes:
        operation: 动作类别，例如 ``copy_file``、``symlink`` 或 ``hooks``。
        target: 相对 Worktree 根目录的目标路径或配置名。
        status: 动作完成、跳过、警告或失败状态。
        message: 对状态的简短说明，不包含文件正文或密钥值。
    """

    operation: str
    target: str
    status: InitializationActionStatus
    message: str

    def __post_init__(self) -> None:
        """校验初始化动作可以写入日志和报告。

        Returns:
            动作、目标、状态和说明合法时不返回数据。

        Raises:
            ValueError: 文本为空或状态枚举无效。
        """

        _require_text(self.operation, "InitializationAction.operation")
        _require_text(self.target, "InitializationAction.target")
        if not isinstance(self.status, InitializationActionStatus):
            raise ValueError("InitializationAction.status 类型无效")
        _require_text(self.message, "InitializationAction.message")


@dataclass(frozen=True)
class InitializationReport:
    """汇总一个新 Worktree 的全部创建后初始化动作。

    Attributes:
        actions: 按实际执行顺序排列的动作记录。
        complete: 是否所有必需动作均成功，可把目录交给 Agent。
        warnings: 可选动作失败等不阻断运行的简短说明。
    """

    actions: tuple[InitializationAction, ...]
    complete: bool
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """校验初始化报告中的动作和警告。

        Returns:
            报告字段合法时不返回数据。

        Raises:
            ValueError: 动作、完成标志或警告字段类型错误。
        """

        if not isinstance(self.actions, tuple) or not all(
            isinstance(item, InitializationAction) for item in self.actions
        ):
            raise ValueError("InitializationReport.actions 类型无效")
        if not isinstance(self.complete, bool):
            raise ValueError("InitializationReport.complete 必须是布尔值")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(item, str) and item.strip() for item in self.warnings
        ):
            raise ValueError("InitializationReport.warnings 必须是非空字符串元组")


@dataclass(frozen=True)
class WorktreeCreateReport:
    """返回创建或复用 Worktree 后的记录和初始化事实。

    Attributes:
        record: 已经进入 READY 状态的受管记录。
        reused: 是否复用了磁盘上已存在且完整匹配的 Worktree。
        initialization: 新建时的初始化报告；纯复用时为 ``None``。
    """

    record: WorktreeRecord
    reused: bool
    initialization: InitializationReport | None


@dataclass(frozen=True)
class WorktreeRemoveReport:
    """返回一次目录移除或分支删除实际产生的状态变化。

    Attributes:
        name: 被操作的受管 Worktree slug。
        directory_removed: Worktree 目录是否已从 Git 登记和磁盘中移除。
        branch_removed: 对应本地分支是否也已删除。
        lifecycle: 操作后仍有记录时的新状态；记录已删除时为 ``None``。
        message: 显示给用户的结果说明。
    """

    name: str
    directory_removed: bool
    branch_removed: bool
    lifecycle: WorktreeLifecycle | None
    message: str


@dataclass(frozen=True)
class WorktreeRecoveryReport:
    """汇总 Manager 启动时加载状态和接管遗留任务的结果。

    Attributes:
        state_trusted: 状态文件是否完整可信，可以执行受管删除操作。
        interrupted_tasks: 上次进程退出时仍排队或运行的子任务摘要。
        warnings: 状态损坏、目录失配或恢复降级等需要展示的说明。
    """

    state_trusted: bool
    interrupted_tasks: tuple[InterruptedTaskSummary, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CleanupReport:
    """汇总一次启动扫描或周期清理检查了哪些临时 Worktree。

    Attributes:
        checked: 通过临时种类和时间过滤、实际接受变更检查的名称。
        pruned: 目录已删除但分支和记录保留的名称。
        skipped: ``(name, reason)`` 对，说明为何保留目录。
        errors: ``(name, error)`` 对，说明检查或移除失败原因。
    """

    checked: tuple[str, ...] = ()
    pruned: tuple[str, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()
    errors: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class InterruptedTaskSummary:
    """记录进程重启后从状态文件导入的一次未完成子任务。

    Attributes:
        task_id: 原任务 ID；创建阶段没有任务 ID 时为 ``None``。
        session_id: 原任务所属主会话 ID。
        worktree_name: 保留下来的受管 Worktree slug。
        path: 用户可以继续检查成果的绝对目录。
        branch: 保存成果的本地分支名。
        base_commit: 创建该临时 Worktree 时使用的本地基线提交。
        reason: 为什么任务被标为 interrupted 的说明。
    """

    task_id: str | None
    session_id: str
    worktree_name: str
    path: Path
    branch: str
    base_commit: str
    reason: str


@dataclass(frozen=True)
class WorktreeSnapshot:
    """向命令和状态界面暴露的一条只读 Worktree 视图。

    Attributes:
        record: 生成快照时观察到的持久化记录。
        leased: 当前进程是否仍有运行者持有该目录租约。
        session_ids: 当前状态中绑定到该目录的主会话 ID。
    """

    record: WorktreeRecord
    leased: bool
    session_ids: tuple[str, ...] = ()
