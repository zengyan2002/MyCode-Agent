"""Agent Team 的团队、任务、消息和运行状态模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from mycode.models.json_types import JsonObject


class TeamLifecycle(str, Enum):
    """说明一个尚未删除的团队当前能否继续接收写操作。"""

    ACTIVE = "active"
    CLEANING = "cleaning"
    CLEANUP_FAILED = "cleanup_failed"


class TeammateState(str, Enum):
    """说明成员 Host 当前处于启动、工作、等待还是终态。"""

    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    SUSPENDED = "suspended"
    FAILED = "failed"
    TERMINATED = "terminated"


class BackendPreference(str, Enum):
    """保存创建成员时用户要求的后端选择方式。"""

    AUTO = "auto"
    TMUX = "tmux"
    ITERM2 = "iterm2"
    IN_PROCESS = "in-process"


class TeammateBackend(str, Enum):
    """保存检测结束后成员实际使用的运行后端。"""

    TMUX = "tmux"
    ITERM2 = "iterm2"
    IN_PROCESS = "in-process"


class TeamTaskStatus(str, Enum):
    """保存共享任务的待办、执行和终态。"""

    TODO = "todo"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TeamTaskPriority(str, Enum):
    """决定成员查看可领取任务时的排序。"""

    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class TeamMessageKind(str, Enum):
    """区分普通沟通和计划、退出两类协议消息。"""

    TEXT = "text"
    PLAN_REQUEST = "plan_request"
    PLAN_RESPONSE = "plan_response"
    SHUTDOWN_REQUEST = "shutdown_request"
    SHUTDOWN_RESPONSE = "shutdown_response"


class PlanDecision(str, Enum):
    """保存 Lead 对某一版成员计划的决定。"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


def _text(value: str, name: str) -> None:
    """检查持久化记录中的必填文字不是空字符串。

    Args:
        value: 准备写入记录的文字。
        name: 出错时展示的字段名。

    Returns:
        字段合法时不返回数据。

    Raises:
        ValueError: 字段不是非空字符串。
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")


def _aware(value: datetime, name: str) -> None:
    """检查记录时间带时区，避免跨进程比较本地裸时间。

    Args:
        value: 需要检查的时间。
        name: 出错时展示的字段名。

    Returns:
        时间带时区时不返回数据。
    """

    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} 必须是带时区时间")


@dataclass(frozen=True, slots=True)
class TeamRecord:
    """代表一个存续团队的身份、负责人和成员花名册。

    Attributes:
        team_id: 创建时生成且永不复用的团队标识。
        name: Lead 和用户用于寻址的团队名称。
        description: 说明团队本次负责的目标。
        lead_session_id: 当前有权管理团队的主会话 ID。
        lead_generation: 接管时递增，用来拒绝旧 Lead 写入。
        lifecycle: 团队当前可写、清理中或清理失败状态。
        member_ids: 按加入顺序保存的成员 ID。
        created_at: 团队首次创建时间。
        updated_at: 最近一次持久化修改时间。
        revision: 每次原子更新递增的快照版本。
    """

    team_id: str
    name: str
    description: str
    lead_session_id: str
    lead_generation: int
    lifecycle: TeamLifecycle
    member_ids: tuple[str, ...]
    created_at: datetime
    updated_at: datetime
    revision: int

    def __post_init__(self) -> None:
        """校验团队身份、时间和并发控制字段可以用于持久化。

        Returns:
            字段合法时不返回数据。

        Raises:
            ValueError: 必填文字为空、时间无时区，或 generation/revision 越界。
        """

        for name in ("team_id", "name", "lead_session_id"):
            _text(getattr(self, name), name)
        _aware(self.created_at, "created_at")
        _aware(self.updated_at, "updated_at")
        if self.lead_generation <= 0 or self.revision < 0:
            raise ValueError("Lead generation 必须为正数且 revision 不能为负数")


@dataclass(frozen=True, slots=True)
class TeammateRecord:
    """代表一个团队成员的身份、工作目录和当前运行位置。

    Attributes:
        agent_id: 成员在团队生命周期内不变的内部标识。
        team_id: 成员所属团队 ID。
        name: 团队内唯一的可读名称。
        role_name: 创建成员时选择的 Agent 定义名称。
        model_override: 成员覆盖的模型；None 表示继承当前配置。
        session_id: 保存成员完整对话的会话 ID。
        worktree_name: WorktreeManager 中的受管名称。
        worktree_path: 成员工具实际使用的绝对工作目录。
        branch: 成员提交代码的 Git 分支。
        backend: 成员实际使用的后端。
        backend_ref: pane、session 或同进程 task 的运行标识。
        state: 成员当前生命周期状态。
        runtime_generation: 每次 Host 启动时递增的写入栅栏。
        owner_pid: 外部 Host 进程；同进程成员可以为空。
        lease_token_hash: 当前租约摘要，原文不写磁盘。
        plan_mode_required: 修改工作区前是否必须取得 Lead 批准。
        current_task_id: 当前 working 任务；没有时为空。
        created_at: 成员首次加入团队的时间。
        updated_at: 最近一次状态变化时间。
    """

    agent_id: str
    team_id: str
    name: str
    role_name: str
    model_override: str | None
    session_id: str
    worktree_name: str
    worktree_path: Path
    branch: str
    backend: TeammateBackend
    backend_ref: str | None
    state: TeammateState
    runtime_generation: int
    owner_pid: int | None
    lease_token_hash: str | None
    plan_mode_required: bool
    current_task_id: str | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        """校验成员标识、绝对工作目录、generation 和时间字段。

        Returns:
            字段合法时不返回数据。

        Raises:
            ValueError: 成员记录无法作为 Host 和 Worktree 的可信持久化状态。
        """

        for name in (
            "agent_id", "team_id", "name", "role_name", "session_id",
            "worktree_name", "branch",
        ):
            _text(getattr(self, name), name)
        if not self.worktree_path.is_absolute():
            raise ValueError("成员 Worktree 路径必须是绝对路径")
        if self.runtime_generation <= 0:
            raise ValueError("成员 runtime generation 必须为正数")
        _aware(self.created_at, "created_at")
        _aware(self.updated_at, "updated_at")


@dataclass(frozen=True, slots=True)
class TeamTaskAttempt:
    """保存共享任务的一次正式执行及暂停、失败信息。

    Attributes:
        number: 从 1 开始，最多为 2。
        owner_id: 本次执行当前或最后的负责人 ID。
        started_at: 本次执行第一次进入 working 的时间。
        paused_at: 最近一次暂停时间。
        ended_at: 完成或失败时间。
        failure_reason: 本次执行失败时的具体原因。
    """

    number: int
    owner_id: str
    started_at: datetime
    paused_at: datetime | None = None
    ended_at: datetime | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        """校验执行次数、负责人和各阶段时间。

        Returns:
            attempt 可追加到任务记录时不返回数据。

        Raises:
            ValueError: 次数不在 1 至 2、负责人为空或时间不带时区。
        """

        if self.number not in {1, 2}:
            raise ValueError("团队任务只允许第 1 或第 2 次执行")
        _text(self.owner_id, "owner_id")
        _aware(self.started_at, "started_at")
        for name in ("paused_at", "ended_at"):
            value = getattr(self, name)
            if value is not None:
                _aware(value, name)


@dataclass(frozen=True, slots=True)
class TeamTaskRecord:
    """代表共享看板中的一项任务及其依赖、进展和提交。

    Attributes:
        task_id: 团队内唯一的任务标识。
        team_id: 任务所属团队 ID。
        title: 看板列表使用的短标题。
        description: 成员执行时读取的完整工作说明。
        task_kind: code 要求提交；research 只要求结构化结果。
        priority: 可领取任务的排序优先级。
        status: todo、working 或一个终态。
        owner_id: 指派或认领的成员；未分配时为空。
        blocked_by: 必须先完成的同团队任务 ID。
        progress: 暂停、交接和状态查询使用的阶段结果。
        result: 完成或失败时保存的最终说明。
        commit_hashes: 代码任务完成时报告的提交。
        attempts: 最多两次正式执行记录。
        created_at: 任务创建时间。
        updated_at: 最近一次更新的时间。
        completed_at: completed 时的时间，其他状态为空。
    """

    task_id: str
    team_id: str
    title: str
    description: str
    task_kind: Literal["code", "research"] = "code"
    priority: TeamTaskPriority = TeamTaskPriority.NORMAL
    status: TeamTaskStatus = TeamTaskStatus.TODO
    owner_id: str | None = None
    blocked_by: tuple[str, ...] = ()
    progress: str | None = None
    result: str | None = None
    commit_hashes: tuple[str, ...] = ()
    attempts: tuple[TeamTaskAttempt, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    updated_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        """校验任务身份、类型、执行次数和持久化时间。

        Returns:
            任务可写入共享看板时不返回数据。

        Raises:
            ValueError: 必填字段为空、任务类型未知、attempt 超过两次或时间无时区。
        """

        for name in ("task_id", "team_id", "title", "description"):
            _text(getattr(self, name), name)
        if self.task_kind not in {"code", "research"}:
            raise ValueError("任务类型只能是 code 或 research")
        if len(self.attempts) > 2:
            raise ValueError("团队任务最多执行两次")
        _aware(self.created_at, "created_at")
        _aware(self.updated_at, "updated_at")
        if self.completed_at is not None:
            _aware(self.completed_at, "completed_at")


@dataclass(frozen=True, slots=True)
class TeamTaskCreateRequest:
    """保存 Lead 创建任务时提供的行为字段。

    Attributes:
        title: 看板中显示的简短任务名。
        description: 成员开始工作前读取的完整要求和完成条件。
        task_kind: ``code`` 要求提交，``research`` 只要求结构化结论。
        priority: 成员列举可领取任务时使用的排序等级。
        blocked_by: 必须先完成的同团队任务 ID。
    """

    title: str
    description: str
    task_kind: Literal["code", "research"] = "code"
    priority: TeamTaskPriority = TeamTaskPriority.NORMAL
    blocked_by: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TeamTaskUpdateRequest:
    """保存一次显式字段更新，避免任意 patch 绕过字段权限。

    Attributes:
        task_id: 需要更新的团队任务 ID。
        status: 新状态；为空表示不改状态。
        owner: Lead 指派或重新分配的成员名称/ID；为空表示不改负责人。
        priority: Lead 指定的新优先级。
        progress: 成员暂停或交接时留下的阶段结果。
        result: 完成、失败或调查任务的最终说明。
        commit_hashes: 代码任务本次报告的 Git 提交。
        add_blocked_by: 本次新增的直接依赖任务 ID。
        remove_blocked_by: 本次移除的直接依赖任务 ID。
        failure_reason: attempt 失败时记录的可诊断原因。
    """

    task_id: str
    status: TeamTaskStatus | None = None
    owner: str | None = None
    priority: TeamTaskPriority | None = None
    progress: str | None = None
    result: str | None = None
    commit_hashes: tuple[str, ...] | None = None
    add_blocked_by: tuple[str, ...] = ()
    remove_blocked_by: tuple[str, ...] = ()
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class TeamTaskQuery:
    """保存任务列表的可选状态、负责人和可领取过滤条件。

    Attributes:
        status: 只返回该状态；为空表示不过滤状态。
        owner_id: 只返回该负责人任务；为空表示不过滤负责人。
        claimable_only: 是否只返回依赖已完成且当前可认领的 todo 任务。
    """

    status: TeamTaskStatus | None = None
    owner_id: str | None = None
    claimable_only: bool = False


@dataclass(frozen=True, slots=True)
class TeamTaskView:
    """把持久化任务和读取时计算的依赖状态一起返回给工具。

    Attributes:
        task: 磁盘中保存的原始任务记录。
        blocked: 是否仍有未完成的直接依赖。
        blocks: 当前任务完成后会解除阻塞的任务 ID。
        assigned: 是否已经有负责人。
        claimable: 当前成员是否可以原子认领该任务。
    """

    task: TeamTaskRecord
    blocked: bool
    blocks: tuple[str, ...]
    assigned: bool
    claimable: bool


@dataclass(frozen=True, slots=True)
class ClaimScanRound:
    """记录一批被成功唤醒的成员是否查看并认领新开放的任务。

    Attributes:
        round_id: 本轮检查的唯一 ID。
        team_id: 检查所属团队 ID。
        task_ids: 本轮要求成员查看的新开放任务。
        expected_member_ids: 后端确认已唤醒、因此需要回报的成员。
        finished_member_ids: 已经完成本轮查看的成员。
        claimed_task_ids: 本轮实际有人认领的任务。
        created_at: 本轮开始时间，不用于时间超时。
    """

    round_id: str
    team_id: str
    task_ids: tuple[str, ...]
    expected_member_ids: tuple[str, ...]
    finished_member_ids: tuple[str, ...]
    claimed_task_ids: tuple[str, ...]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class MailboxMessage:
    """代表已经持久化到一个收件箱中的团队消息。

    Attributes:
        message_id: 跨进程去重和 cursor 确认使用的消息 ID。
        team_id: 消息所属团队 ID。
        sender_id: 运行时确认的发送方 Agent ID 或 ``lead``。
        recipient_id: 实际落盘的单个收件人 ID。
        kind: 普通文本、计划审批或退出协议类型。
        summary: 消息列表展示的短摘要。
        body: 下一轮注入收件人上下文的正文。
        wake: 发送方是否明确要求唤醒空闲收件人。
        payload: 协议消息携带的结构化字段。
        created_at: 消息成功追加到邮箱的时间。
    """

    message_id: str
    team_id: str
    sender_id: str
    recipient_id: str
    kind: TeamMessageKind
    summary: str
    body: str
    wake: bool
    payload: JsonObject
    created_at: datetime


@dataclass(frozen=True, slots=True)
class MailboxCursor:
    """保存一个收件人下次读取 JSONL 的字节位置。

    Attributes:
        byte_offset: 已确认消费内容末尾的 UTF-8 字节偏移。
        last_message_id: 最近确认的消息 ID；空邮箱时为空。
    """

    byte_offset: int = 0
    last_message_id: str | None = None


@dataclass(frozen=True, slots=True)
class SendMessageRequest:
    """保存 SendMessage 工具已经通过 Schema 校验的输入。

    Attributes:
        to: 成员名称、Agent ID 或广播地址 ``*``。
        kind: 文本或受支持的结构化协议类型。
        summary: 纯文本消息必须提供的列表摘要。
        message: 收件人下一轮读取的正文。
        wake: 是否显式唤醒 idle/suspended 收件人。
        payload: 计划和退出协议所需的结构化字段。
    """

    to: str
    kind: TeamMessageKind = TeamMessageKind.TEXT
    summary: str = ""
    message: str = ""
    wake: bool = False
    payload: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DeliveryItem:
    """说明广播中的一个收件人是否完成持久化和唤醒请求。

    Attributes:
        recipient_id: 本条投递对应的实际 Agent ID。
        delivered: 消息是否已经追加到该收件人邮箱。
        error: 投递失败时的具体原因；成功时为空。
    """

    recipient_id: str
    delivered: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DeliveryReport:
    """汇总一次点对点或广播发送的逐收件人结果。

    Attributes:
        deliveries: 每个实际收件人的独立投递结果。
    """

    deliveries: tuple[DeliveryItem, ...]


@dataclass(frozen=True, slots=True)
class MemberPlanApproval:
    """保存 Lead 对某个任务执行版本的计划审批结果。

    Attributes:
        member_id: 提交计划的成员 ID。
        task_id: 计划所属任务 ID。
        attempt_number: 计划对应的正式执行次数。
        plan_revision: 同一 attempt 内成员提交的计划版本号。
        plan_text: Lead 实际审阅的计划正文。
        decision: 待审、批准或拒绝。
        feedback: 拒绝时返回成员的修改意见。
        decided_by_generation: 作出决定的 Lead generation；待审时为空。
        updated_at: 最近提交或审批时间。
    """

    member_id: str
    task_id: str
    attempt_number: int
    plan_revision: int
    plan_text: str
    decision: PlanDecision
    feedback: str | None
    decided_by_generation: int | None
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class TeamBinding:
    """把主会话绑定到一个团队和有效 Lead generation。

    Attributes:
        team_id: 主会话恢复时应重新连接的团队 ID。
        lead_generation: 恢复时必须仍匹配的 Lead 写入代次。
    """

    team_id: str
    lead_generation: int


@dataclass(frozen=True, slots=True)
class TeamActorContext:
    """保存本地运行时确认的团队调用者身份。

    Attributes:
        team_id: Actor 当前所属团队 ID。
        actor_id: ``lead`` 或成员不可变 Agent ID。
        actor_kind: 区分 Lead 和普通团队成员。
        generation: Store 每次写入前重新核对的栅栏代次。
    """

    team_id: str
    actor_id: str
    actor_kind: Literal["lead", "member"]
    generation: int


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """保存 Lead 实际运行的一条中间或最终验证命令。

    Attributes:
        command: Coordinator 实际执行的原始命令。
        scope: ``focused`` 为合并后轻量验证，``final`` 为全量验证。
        exit_code: 命令进程的真实退出码。
        head: 运行验证时主分支 HEAD。
        ran_at: 验证结束并写入记录的时间。
    """

    command: str
    scope: Literal["focused", "final"]
    exit_code: int
    head: str
    ran_at: datetime


@dataclass(frozen=True, slots=True)
class TeamIntegrationState:
    """保存当前团队已经合并的提交、冲突和验证证据。

    Attributes:
        team_id: 状态所属的不可变团队 ID。
        merged_commits: 已成功进入 Lead 当前分支的合并提交。
        current_source_branch: 正在合并或等待成员修正的来源分支。
        merge_attempt: 当前来源分支已经开始过的合并次数，最多为 2。
        conflicted_files: Git 当前报告的未合并文件相对路径。
        blocked_by_validation: 最近一次中间验证是否失败并阻止后续合并。
        validation_repair_task_id: 验证失败后创建的修复任务；该任务完成前
            不解除合并阻塞。
        validation_reports: Lead 实际运行过的轻量或最终验证记录。
        updated_at: 最近一次集成状态写入时间。
    """

    team_id: str
    merged_commits: tuple[str, ...] = ()
    current_source_branch: str | None = None
    merge_attempt: int = 0
    conflicted_files: tuple[Path, ...] = ()
    blocked_by_validation: bool = False
    validation_repair_task_id: str | None = None
    validation_reports: tuple[ValidationReport, ...] = ()
    updated_at: datetime = field(default_factory=lambda: datetime.now().astimezone())


@dataclass(frozen=True, slots=True)
class TeamEvent:
    """保存生命周期变化或无人认领等系统事件。

    Attributes:
        event_id: 事件唯一 ID。
        team_id: 事件所属团队 ID。
        kind: 生命周期或任务扫描事件名。
        actor_id: 触发事件的 Lead/成员 ID；系统事件可以为空。
        payload: 事件类型对应的可序列化字段。
        created_at: 事件追加到 JSONL 的时间。
    """

    event_id: str
    team_id: str
    kind: str
    actor_id: str | None
    payload: JsonObject
    created_at: datetime


@dataclass(frozen=True, slots=True)
class TeamSnapshot:
    """把团队、成员、任务、扫描轮次和集成状态组成一致读取快照。

    Attributes:
        team: 当前团队身份和生命周期记录。
        members: 按花名册顺序排列的成员记录。
        tasks: 当前共享任务记录。
        scans: 尚在磁盘中的自主认领检查轮次。
        integration: 当前合并、冲突和验证状态。
    """

    team: TeamRecord
    members: tuple[TeammateRecord, ...]
    tasks: tuple[TeamTaskRecord, ...]
    scans: tuple[ClaimScanRound, ...]
    integration: TeamIntegrationState


@dataclass(frozen=True, slots=True)
class TeamDeletionReport:
    """说明团队当前能否删除，以及已经清理了哪些资源。

    Attributes:
        team_id: 被检查或清理的团队 ID。
        allowed: 预检允许删除或实际清理成功时为 True。
        blockers: 阻止删除或中断清理的全部可读原因。
        removed_resources: 本次调用已经实际移除的资源，便于失败后续清理。
    """

    team_id: str
    allowed: bool
    blockers: tuple[str, ...] = ()
    removed_resources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SpawnTeammateRequest:
    """保存 Lead 创建长期成员时需要的角色和后端选项。

    Attributes:
        team_name: 目标团队名称，必须与当前 Lead 团队一致。
        name: 团队内唯一的成员名称。
        role_name: 要加载的 Agent 定义名称。
        prompt: 成员启动后首先处理的任务背景或要求。
        model_override: 可选模型覆盖；为空时使用 Agent 定义或默认模型。
        backend: 自动检测或显式指定的运行后端偏好。
        plan_mode_required: 成员修改代码前是否必须提交计划给 Lead 审批。
    """

    team_name: str
    name: str
    role_name: str
    prompt: str
    model_override: str | None = None
    backend: BackendPreference = BackendPreference.AUTO
    plan_mode_required: bool = False


@dataclass(frozen=True, slots=True)
class TeamCreateRequest:
    """保存 TeamCreate 的名称和用户可读说明。

    Attributes:
        team_name: 当前工作区存续期间唯一的团队名称。
        description: 向 Lead 和成员说明团队总体目标的文字。
    """

    team_name: str
    description: str = ""
