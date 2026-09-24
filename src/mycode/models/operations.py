"""保存工具步骤的身份、执行状态和用于恢复的结果。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from mycode.models.messages import AssistantMessage, ToolCall
from mycode.models.json_types import JsonObject
from mycode.models.tools import ToolAccess, ToolErrorCode, ToolExecutionResult


class OperationState(str, Enum):
    """区分还未领取、可能正在执行、已有结果和效果未知的步骤。"""

    PREPARED = "prepared"
    RUNNING = "running"
    COMPLETED = "completed"
    UNKNOWN = "unknown"


class ResolutionVerdict(str, Enum):
    """用户核查实际效果后登记的两种判断。"""

    COMPLETED = "completed"
    NOT_APPLIED = "not-applied"


@dataclass(frozen=True)
class OperationScope:
    """记录任务属于哪个会话、由哪个运行在什么目录执行。"""

    session_id: str
    execution_id: str
    runtime_id: str
    workspace_root: Path
    actor_key: str | None = None


@dataclass(frozen=True)
class OperationRecord:
    """数据库中的一个工具步骤；result 在已保存工具结果时有值。"""

    operation_id: str
    scope: OperationScope
    batch_id: str
    call_index: int
    call: ToolCall
    arguments_hash: str
    access: ToolAccess
    state: OperationState
    attempt: int
    owner_pid: int | None
    owner_token: str | None
    execution_started: bool | None
    result: ToolExecutionResult | None
    reason: str | None
    resolution_source: str | None


@dataclass(frozen=True)
class ToolBatchRecord:
    """保存一条助手消息及它按声明顺序引用的操作。"""

    batch_id: str
    scope: OperationScope
    batch_number: int
    assistant: AssistantMessage
    operation_ids: tuple[str, ...]
    history_committed: bool


class ClaimKind(str, Enum):
    """告诉调度器本次可以执行，还是只能读取已有状态。"""

    EXECUTE = "execute"
    REPLAY = "replay"
    IN_PROGRESS = "in_progress"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class ClaimResult:
    """领取结果与当前操作记录；只有 EXECUTE 允许进入工具。"""

    kind: ClaimKind
    record: OperationRecord


@dataclass(frozen=True)
class RecoveryReport:
    """告诉恢复调用方补回了几批消息，还有哪些操作无法确认。"""

    appended_batches: int
    blocked_operation_ids: tuple[str, ...]


@dataclass(frozen=True)
class FileWriteExpectation:
    """写线程在修改文件前保存的目标字节摘要，以及该次写入是否已结束。"""

    operation_id: str
    owner_token: str
    attempt: int
    target_path: str
    expected_sha256: str
    expected_size: int
    writer_finished: bool = False


@dataclass(frozen=True)
class FileVerificationCandidate:
    """读取前取得的原操作和文件预期；reason 非空时只能记录未确认原因。"""

    record: OperationRecord
    expectation: FileWriteExpectation | None
    reason: str | None = None


@dataclass(frozen=True)
class OperationVerification:
    """一次真实文件读取对原写操作得出的核查结论。"""

    verification_id: str
    operation_id: str
    query_operation_id: str
    owner_token: str
    attempt: int
    verdict: str
    evidence: JsonObject
    previous_result: JsonObject | None
    created_at: str


def operation_failure(call: ToolCall, code: ToolErrorCode, message: str) -> ToolExecutionResult:
    """为未启动或无法确认结果的调用生成明确的工具报告。"""
    return ToolExecutionResult(
        tool_call_id=call.id, tool_name=call.name, success=False, content="",
        error_code=code, error_message=message, timed_out=code is ToolErrorCode.TIMEOUT,
        truncated=False, original_size_bytes=0, duration_ms=0,
    )
