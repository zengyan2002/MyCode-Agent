"""查询中断工具、补回会话结果，并处理用户明确要求的原操作重试。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from mycode.agent.cancellation import CancellationToken
from mycode.hooks.runtime import HookRunScope
from mycode.models.events import AgentRunOptions
from mycode.models.messages import AssistantMessage, ToolResultMessage
from mycode.models.operations import (OperationRecord, OperationScope, OperationState, RecoveryReport,
    ToolBatchRecord, operation_failure, FileWriteExpectation, FileVerificationCandidate)
from mycode.models.tools import ToolAccess, ToolErrorCode, ToolExecutionResult
from mycode.persistence.operations import OperationError, OperationStore, operation_io, _process_alive
from mycode.persistence.sessions import SessionManager
from mycode.tools.scheduler import ToolScheduler


def recovery_result(record: OperationRecord) -> ToolExecutionResult:
    """保留已完成的真实结果，其余状态明确报告没有执行或效果未知。"""
    if record.state is OperationState.COMPLETED and record.result is not None:
        return record.result
    if record.state is OperationState.PREPARED:
        return operation_failure(record.call, ToolErrorCode.CANCELLED, "原任务中断，本步骤尚未执行")
    code = (ToolErrorCode.OPERATION_IN_PROGRESS if record.state is OperationState.RUNNING
            else ToolErrorCode.OPERATION_UNKNOWN)
    return operation_failure(record.call, code,
        f"工具效果尚未确认。操作：{record.operation_id}。{record.reason or '原执行者可能仍在运行'}")


class OperationRecovery:
    """使用项目执行记录恢复历史；不调用模型猜测外部操作是否成功。"""

    def __init__(self, store: OperationStore) -> None:
        self.store = store

    async def file_candidates(self, scope: OperationScope, path: Path) -> tuple[FileVerificationCandidate, ...]:
        """读取前筛选本运行同一路径的未知写入；线程未结束的只记录原因。"""
        records = await operation_io(self.store.list_operations, runtime_id=scope.runtime_id)
        candidates = []
        for record in records:
            if (record.state is not OperationState.UNKNOWN
                    or record.call.name not in ("write_file", "edit_file")
                    or record.scope.session_id != scope.session_id
                    or record.scope.workspace_root != scope.workspace_root
                    or record.scope.actor_key != scope.actor_key
                    or record.owner_token is None):
                continue
            target = (scope.workspace_root / str(record.call.arguments["path"])).resolve()
            if target != path:
                continue
            expected = await operation_io(self.store.file_expectation, record.operation_id, record.owner_token)
            exited = record.owner_pid is not None and await operation_io(_process_alive, record.owner_pid) is False
            if expected is None and record.call.name == "write_file" and exited:
                data = str(record.call.arguments["content"]).encode("utf-8")
                expected = FileWriteExpectation(record.operation_id, record.owner_token, record.attempt,
                    str(path), hashlib.sha256(data).hexdigest(), len(data), True)
            reason = None
            if expected is None:
                reason = "missing_expectation"
            elif expected.target_path != str(path):
                reason = "unavailable"
            elif not expected.writer_finished and not exited:
                reason = "writer_active"
            candidates.append(FileVerificationCandidate(record, expected, reason))
        return tuple(candidates)

    async def unresolved(self, runtime_id: str) -> tuple[OperationRecord, ...]:
        """检查原运行的遗留记录，返回尚未结束或效果未知的写操作。"""
        records = await operation_io(self.store.list_operations, runtime_id=runtime_id)
        blocked = []
        for record in records:
            if record.state is OperationState.RUNNING:
                record = await operation_io(self.store.recover_orphan, record.operation_id)
            active_prepared = record.state is OperationState.PREPARED and await operation_io(self.store.execution_active, record.scope.execution_id)
            if record.access is ToolAccess.WRITE and (active_prepared or record.state in (OperationState.RUNNING, OperationState.UNKNOWN)):
                blocked.append(record)
        return tuple(blocked)

    async def reconcile(self, runtime_id: str, sessions: SessionManager) -> RecoveryReport:
        """在读取旧会话链之前补回已保存的批次，不重新执行工具。"""
        blocked = await self.unresolved(runtime_id)
        batches = await operation_io(self.store.pending_batches, runtime_id)
        appended = 0
        for batch in batches:
            records = [await operation_io(self.store.get, oid) for oid in batch.operation_ids]
            if any(record.state is OperationState.PREPARED for record in records) and await operation_io(self.store.execution_active, batch.scope.execution_id):
                continue
            if any(record.state is OperationState.RUNNING for record in records):
                continue
            results = []
            for record in records:
                result = recovery_result(record)
                if record.state is OperationState.PREPARED:
                    await operation_io(self.store.finish_unstarted, record.operation_id, result)
                results.append(result)
            messages = (batch.assistant, *(ToolResultMessage(result.tool_call_id, result.tool_name,
                         result.to_model_json(), not result.success) for result in results))
            appended += await operation_io(sessions.reconcile_tool_batch, batch, messages)
            await operation_io(self.store.mark_history_committed, batch.batch_id)
        return RecoveryReport(appended, tuple(record.operation_id for record in blocked))

    async def retry(self, operation_id: str, *, scope: OperationScope, scheduler: ToolScheduler,
                    options: AgentRunOptions, cancellation: CancellationToken,
                    hook_scope: HookRunScope, visible_tool_names: frozenset[str]) -> ToolExecutionResult:
        """在原运行权限下重试未启动步骤；已执行结果复用，未知写操作拒绝。"""
        record = await operation_io(self.store.recover_orphan, operation_id)
        if record.scope != scope or not scope.workspace_root.is_dir():
            raise OperationError("当前不是原操作的运行、目录或身份，不能代执行")
        if record.state in (OperationState.RUNNING, OperationState.UNKNOWN):
            raise OperationError("操作可能已经生效，必须先核查实际结果，不能直接重试")
        record = await operation_io(self.store.prepare_retry, operation_id, scope)
        if record.state is OperationState.COMPLETED:
            assert record.result is not None
            return record.result
        # 重试原步骤不新建业务操作，不重写原批次的已投影历史。
        batch = ToolBatchRecord(record.batch_id, scope, 1, AssistantMessage((record.call,)),
                                (record.operation_id,), True)
        session = scheduler.schedule((record.call,), batch=batch, model_call_number=1,
            options=options, cancellation=cancellation, hook_scope=hook_scope,
            visible_tool_names=visible_tool_names)
        try:
            async for _ in session.stream():
                pass
            return (await session.finalize(ToolErrorCode.CANCELLED if cancellation.is_cancelled else None))[0]
        finally:
            await session.finalize(ToolErrorCode.CANCELLED)
