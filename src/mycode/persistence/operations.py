"""在项目本地 SQLite 中登记工具步骤，领取执行权并保留实际结果。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from mycode.errors import MyCodeError
from mycode.models.messages import AssistantMessage, ToolCall
from mycode.models.operations import (
    ClaimKind, ClaimResult, OperationRecord, OperationScope, OperationState,
    ResolutionVerdict, ToolBatchRecord, operation_failure,
    FileWriteExpectation, FileVerificationCandidate, OperationVerification,
)
from mycode.models.tools import ToolAccess, ToolErrorCode, ToolExecutionResult
from mycode.persistence.session_codec import SessionCodec, SessionRecord


class OperationError(MyCodeError):
    """执行记录不能读取、更新或与本次调用匹配。"""


async def operation_io(function, *args, **kwargs):
    """在线程中完成短数据库操作；取消时也等事务结束，避免后台遗留领取。"""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        # 取出线程异常，取消仍由外层负责记录和收尾。
        if task.done() and not task.cancelled():
            task.exception()
        raise


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(call: ToolCall, scope: OperationScope) -> str:
    payload = [call.name, call.arguments, str(scope.workspace_root.resolve()), scope.actor_key]
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()


def _process_alive(pid: int) -> bool | None:
    """只读查询本机进程；无法证明已经退出时返回 None，不抢占执行权。"""
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            return False if ctypes.get_last_error() == 87 else None
        try:
            status = kernel.WaitForSingleObject(handle, 0)
            return {0: False, 258: True}.get(status)
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return None
    return True


class OperationStore:
    """多个本机运行共享的工具执行记录；目录在启动时固定，不跟随 Worktree。"""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.path = self.project_root / ".mycode" / "operations.sqlite3"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._db() as db:
                db.execute("PRAGMA journal_mode=WAL")
                version = db.execute("PRAGMA user_version").fetchone()[0]
                if version not in (0, 1, 2):
                    raise OperationError("工具执行数据库版本不受支持")
                db.executescript("""
                    BEGIN IMMEDIATE;
                    CREATE TABLE IF NOT EXISTS executions (
                        execution_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                        runtime_id TEXT NOT NULL, workspace_root TEXT NOT NULL,
                        actor_key TEXT, owner_pid INTEGER NOT NULL, owner_released INTEGER NOT NULL DEFAULT 0,
                        state TEXT NOT NULL DEFAULT 'active', created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS tool_batches (
                        batch_id TEXT PRIMARY KEY, execution_id TEXT NOT NULL REFERENCES executions,
                        batch_number INTEGER NOT NULL, assistant_json TEXT NOT NULL,
                        workspace_root TEXT NOT NULL, actor_key TEXT,
                        history_committed INTEGER NOT NULL DEFAULT 0,
                        UNIQUE(execution_id, batch_number)
                    );
                    CREATE TABLE IF NOT EXISTS tool_operations (
                        operation_id TEXT PRIMARY KEY, execution_id TEXT NOT NULL REFERENCES executions,
                        batch_id TEXT NOT NULL REFERENCES tool_batches, call_index INTEGER NOT NULL,
                        call_id TEXT NOT NULL, tool_name TEXT NOT NULL, arguments_json TEXT NOT NULL,
                        arguments_hash TEXT NOT NULL, access TEXT NOT NULL, state TEXT NOT NULL,
                        attempt INTEGER NOT NULL DEFAULT 0, owner_pid INTEGER, owner_token TEXT,
                        execution_started INTEGER, result_json TEXT, reason TEXT, resolution_source TEXT,
                        created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS batch_steps (
                        batch_id TEXT NOT NULL REFERENCES tool_batches,
                        call_index INTEGER NOT NULL, operation_id TEXT NOT NULL REFERENCES tool_operations,
                        PRIMARY KEY(batch_id, call_index)
                    );
                    CREATE TABLE IF NOT EXISTS tool_call_links (
                        execution_id TEXT NOT NULL REFERENCES executions, tool_call_id TEXT NOT NULL,
                        operation_id TEXT NOT NULL REFERENCES tool_operations,
                        PRIMARY KEY(execution_id, tool_call_id)
                    );
                    CREATE TABLE IF NOT EXISTS operation_resolutions (
                        id INTEGER PRIMARY KEY, operation_id TEXT NOT NULL REFERENCES tool_operations,
                        verdict TEXT NOT NULL, note TEXT NOT NULL, previous_result_json TEXT, created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS execution_runtime ON executions(runtime_id);
                    CREATE TABLE IF NOT EXISTS file_write_expectations (
                        operation_id TEXT NOT NULL REFERENCES tool_operations,
                        owner_token TEXT NOT NULL, attempt INTEGER NOT NULL,
                        target_path TEXT NOT NULL, expected_sha256 TEXT NOT NULL,
                        expected_size INTEGER NOT NULL, writer_finished INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL, finished_at TEXT,
                        PRIMARY KEY(operation_id,owner_token)
                    );
                    CREATE TABLE IF NOT EXISTS operation_verifications (
                        verification_id TEXT PRIMARY KEY,
                        operation_id TEXT NOT NULL REFERENCES tool_operations,
                        query_operation_id TEXT NOT NULL REFERENCES tool_operations,
                        owner_token TEXT NOT NULL, attempt INTEGER NOT NULL,
                        verdict TEXT NOT NULL, evidence_json TEXT NOT NULL,
                        previous_result_json TEXT, created_at TEXT NOT NULL,
                        UNIQUE(operation_id,query_operation_id,owner_token)
                    );
                    PRAGMA user_version=2;
                    COMMIT;
                """)
        except OSError as exc:
            raise OperationError("无法创建工具执行数据库") from exc

    @contextmanager
    def _db(self, *, write=False):
        db = None
        try:
            db = sqlite3.connect(self.path, timeout=1.0)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA busy_timeout=1000")
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except sqlite3.Error as exc:
            if db is not None:
                db.rollback()
            raise OperationError("工具执行数据库无法完成读写") from exc
        finally:
            if db is not None:
                db.close()

    def begin_execution(self, scope: OperationScope) -> None:
        """首次保存逻辑任务；重复进入同一任务必须具有相同归属。"""
        with self._db(write=True) as db:
            values = (scope.execution_id, scope.session_id, scope.runtime_id,
                      str(scope.workspace_root.resolve()), scope.actor_key)
            db.execute("INSERT OR IGNORE INTO executions "
                       "(execution_id,session_id,runtime_id,workspace_root,actor_key,owner_pid,created_at) "
                       "VALUES (?,?,?,?,?,?,?)", (*values, os.getpid(), _now()))
            row = db.execute("SELECT * FROM executions WHERE execution_id=?", (scope.execution_id,)).fetchone()
            if (row["session_id"], row["runtime_id"]) != (scope.session_id, scope.runtime_id):
                raise OperationError("执行任务身份与已保存记录冲突")

    @staticmethod
    def _scope(row) -> OperationScope:
        return OperationScope(row["session_id"], row["execution_id"], row["runtime_id"],
                              Path(row["workspace_root"]), row["actor_key"])

    def _get(self, db, operation_id: str) -> OperationRecord:
        row = db.execute("SELECT o.*, e.session_id,e.runtime_id,b.workspace_root,b.actor_key "
                         "FROM tool_operations o JOIN executions e USING(execution_id) JOIN tool_batches b ON b.batch_id=o.batch_id "
                         "WHERE operation_id=?", (operation_id,)).fetchone()
        if row is None:
            raise OperationError("找不到工具操作")
        result = None
        if row["result_json"] is not None:
            payload = json.loads(row["result_json"])
            payload["error_code"] = ToolErrorCode(payload["error_code"]) if payload["error_code"] else None
            result = ToolExecutionResult(**payload)
        return OperationRecord(
            row["operation_id"], self._scope(row), row["batch_id"], row["call_index"],
            ToolCall(row["call_id"], row["tool_name"], json.loads(row["arguments_json"])),
            row["arguments_hash"], ToolAccess(row["access"]), OperationState(row["state"]),
            row["attempt"], row["owner_pid"], row["owner_token"],
            bool(row["execution_started"]) if row["execution_started"] is not None else None,
            result, row["reason"], row["resolution_source"],
        )

    def get(self, operation_id: str) -> OperationRecord:
        """读取一个已登记工具步骤及它保存的结果。"""
        with self._db() as db:
            return self._get(db, operation_id)

    def save_file_expectation(self, value: FileWriteExpectation) -> None:
        """线程真正修改文件前保存预期内容；领取身份改变后不再允许写入。"""
        with self._db(write=True) as db:
            record = self._get(db, value.operation_id)
            if (record.owner_token != value.owner_token or record.attempt != value.attempt
                    or record.state not in (OperationState.RUNNING, OperationState.UNKNOWN)):
                raise OperationError("文件写入已不属于当前执行者")
            db.execute("INSERT INTO file_write_expectations VALUES (?,?,?,?,?,?,0,?,NULL)",
                       (value.operation_id, value.owner_token, value.attempt, value.target_path,
                        value.expected_sha256, value.expected_size, _now()))

    def finish_file_writer(self, operation_id: str, owner_token: str) -> None:
        """只标记这一次线程的文件操作已结束，不把未知操作改为成功。"""
        with self._db(write=True) as db:
            db.execute("UPDATE file_write_expectations SET writer_finished=1,finished_at=? "
                       "WHERE operation_id=? AND owner_token=?", (_now(), operation_id, owner_token))

    def file_expectation(self, operation_id: str, owner_token: str) -> FileWriteExpectation | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM file_write_expectations WHERE operation_id=? AND owner_token=?",
                             (operation_id, owner_token)).fetchone()
        if row is None:
            return None
        return FileWriteExpectation(row["operation_id"], row["owner_token"], row["attempt"],
            row["target_path"], row["expected_sha256"], row["expected_size"], bool(row["writer_finished"]))

    def complete_verification_read(self, query_operation_id: str, owner_token: str,
                                   result: ToolExecutionResult,
                                   candidates: tuple[FileVerificationCandidate, ...]) -> ToolExecutionResult:
        """一起保存真实读取结果与自动确认；人工已处理的原操作不会被覆盖。"""
        with self._db(write=True) as db:
            query = self._get(db, query_operation_id)
            if query.state is OperationState.COMPLETED and query.owner_token == owner_token:
                return query.result
            if query.state is not OperationState.RUNNING or query.owner_token != owner_token:
                raise OperationError("当前调用不再持有核查读取执行权")
            summaries = []
            for candidate in candidates:
                old = candidate.record
                if (old.scope.runtime_id != query.scope.runtime_id
                        or old.scope.session_id != query.scope.session_id
                        or old.scope.workspace_root != query.scope.workspace_root
                        or old.scope.actor_key != query.scope.actor_key):
                    raise OperationError("核查读取与原操作不属于同一运行、目录或身份")
                current = self._get(db, old.operation_id)
                if (current.state is not OperationState.UNKNOWN or current.owner_token != old.owner_token
                        or current.attempt != old.attempt or current.arguments_hash != old.arguments_hash
                        or current.scope != old.scope):
                    continue
                expected = candidate.expectation
                evidence = {"rule": "full_file_sha256", "path": expected.target_path if expected else None,
                    "expected_sha256": expected.expected_sha256 if expected else None,
                    "expected_size": expected.expected_size if expected else None,
                    "actual_sha256": result.metadata.get("file_sha256"),
                    "actual_size": result.metadata.get("total_bytes"),
                    "reason": candidate.reason or result.error_message}
                verdict = candidate.reason or "unavailable"
                if candidate.reason is None and expected is not None and result.success:
                    fresh = result.metadata.get("fresh_file_read") is True
                    matching = (fresh and result.metadata.get("resolved_path") == expected.target_path
                        and evidence["actual_sha256"] == expected.expected_sha256
                        and evidence["actual_size"] == expected.expected_size)
                    verdict = "matched" if matching else "different"
                vid = "verify-" + uuid4().hex
                previous = _json(asdict(current.result)) if current.result else None
                db.execute("INSERT INTO operation_verifications VALUES (?,?,?,?,?,?,?,?,?)",
                    (vid, old.operation_id, query_operation_id, old.owner_token, old.attempt,
                     verdict, _json(evidence), previous, _now()))
                summary = {"verification_id": vid, "operation_id": old.operation_id,
                           "verdict": verdict, "source": "file_verification"}
                summaries.append(summary)
                if verdict == "matched":
                    confirmed = replace(result, tool_call_id=current.call.id, tool_name=current.call.name,
                        content="程序核查确认目标文件的完整内容已满足原操作要求；未重新执行写入。",
                        metadata={**summary, "query_operation_id": query_operation_id, "evidence": evidence},
                        timed_out=False, truncated=False, original_size_bytes=0, duration_ms=0)
                    db.execute("UPDATE tool_operations SET state='completed',result_json=?,"
                               "resolution_source='file_verification',updated_at=? WHERE operation_id=?",
                               (_json(asdict(confirmed)), _now(), old.operation_id))
            result = replace(result, metadata={**result.metadata, "operation_verifications": summaries})
            db.execute("UPDATE tool_operations SET state='completed',result_json=?,execution_started=1,"
                       "resolution_source='tool',updated_at=? WHERE operation_id=?",
                       (_json(asdict(result)), _now(), query_operation_id))
            for execution_id in {c.record.scope.execution_id for c in candidates}:
                db.execute("UPDATE executions SET state=CASE WHEN owner_released=1 THEN 'finished' ELSE 'active' END "
                           "WHERE execution_id=? AND NOT EXISTS (SELECT 1 FROM tool_operations "
                           "WHERE execution_id=? AND state!='completed')", (execution_id, execution_id))
            return result

    @staticmethod
    def _verification(row) -> OperationVerification:
        return OperationVerification(row["verification_id"], row["operation_id"], row["query_operation_id"],
            row["owner_token"], row["attempt"], row["verdict"], json.loads(row["evidence_json"]),
            json.loads(row["previous_result_json"]) if row["previous_result_json"] else None, row["created_at"])

    def verifications(self, operation_id: str) -> tuple[OperationVerification, ...]:
        """返回这次操作的自动核查依据，保留原来的失败结果。"""
        with self._db() as db:
            return tuple(self._verification(r) for r in db.execute(
                "SELECT * FROM operation_verifications WHERE operation_id=? ORDER BY created_at,verification_id",
                (operation_id,)))

    def verification_summary(self, runtime_id: str) -> tuple[OperationVerification, ...]:
        """返回本运行已自动确认且仍有效的依据，供恢复后的模型了解当前状态。"""
        with self._db() as db:
            return tuple(self._verification(r) for r in db.execute(
                "SELECT v.* FROM operation_verifications v JOIN tool_operations o USING(operation_id) "
                "JOIN executions e USING(execution_id) WHERE e.runtime_id=? AND v.verdict='matched' "
                "AND o.state='completed' AND o.resolution_source='file_verification' "
                "AND v.owner_token=o.owner_token ORDER BY v.created_at,v.verification_id", (runtime_id,)))

    def list_operations(self, *, session_id=None, runtime_id=None) -> tuple[OperationRecord, ...]:
        """查询当前项目的记录，可按归属会话或具体运行过滤。"""
        with self._db() as db:
            rows = db.execute("SELECT operation_id FROM tool_operations o JOIN executions e USING(execution_id) "
                              "WHERE (? IS NULL OR e.session_id=?) AND (? IS NULL OR e.runtime_id=?) "
                              "ORDER BY o.created_at,o.operation_id",
                              (session_id, session_id, runtime_id, runtime_id)).fetchall()
            return tuple(self._get(db, row[0]) for row in rows)

    def _batch(self, db, batch_id: str) -> ToolBatchRecord:
        row = db.execute("SELECT b.*,e.session_id,e.runtime_id "
                         "FROM tool_batches b JOIN executions e USING(execution_id) WHERE batch_id=?",
                         (batch_id,)).fetchone()
        if row is None:
            raise OperationError("找不到工具批次")
        assistant = SessionCodec().decode(row["assistant_json"]).message
        assert isinstance(assistant, AssistantMessage)
        ids = tuple(r[0] for r in db.execute("SELECT operation_id FROM batch_steps WHERE batch_id=? "
                                            "ORDER BY call_index", (batch_id,)))
        return ToolBatchRecord(batch_id, self._scope(row), row["batch_number"], assistant,
                               ids, bool(row["history_committed"]))

    def prepare_batch(self, scope: OperationScope, batch_number: int, assistant: AssistantMessage,
                      accesses: tuple[ToolAccess, ...]) -> ToolBatchRecord:
        """原子保存整批调用；同一任务重新收到原调用 ID 时引用原操作。"""
        calls = assistant.tool_calls
        if not calls or len(calls) != len(accesses) or len({c.id for c in calls}) != len(calls):
            raise OperationError("工具批次为空、分类不匹配或包含重复调用 ID")
        self.begin_execution(scope)
        with self._db(write=True) as db:
            row = db.execute("SELECT batch_id FROM tool_batches WHERE execution_id=? AND batch_number=?",
                             (scope.execution_id, batch_number)).fetchone()
            if row:
                batch = self._batch(db, row[0])
                if batch.assistant != assistant or batch.scope != scope or tuple(self._get(db, oid).access for oid in batch.operation_ids) != accesses:
                    raise OperationError("工具批次内容与原记录冲突")
                return batch
            batch_id = "batch-" + uuid4().hex
            encoded = SessionCodec().encode(SessionRecord(datetime.now(timezone.utc), assistant))
            db.execute("INSERT INTO tool_batches(batch_id,execution_id,batch_number,assistant_json,workspace_root,actor_key) VALUES(?,?,?,?,?,?)",
                       (batch_id, scope.execution_id, batch_number, encoded, str(scope.workspace_root), scope.actor_key))
            for index, (call, access) in enumerate(zip(calls, accesses, strict=True)):
                old = db.execute("SELECT operation_id FROM tool_call_links WHERE execution_id=? AND tool_call_id=?",
                                 (scope.execution_id, call.id)).fetchone()
                if old:
                    operation_id = old[0]
                    record = self._get(db, operation_id)
                    if record.arguments_hash != _fingerprint(call, scope) or record.access != access:
                        raise OperationError("重复工具调用的参数或工具身份发生变化")
                else:
                    operation_id = "op-" + uuid4().hex
                    db.execute("INSERT INTO tool_operations(operation_id,execution_id,batch_id,call_index,call_id,"
                               "tool_name,arguments_json,arguments_hash,access,state,created_at,updated_at) "
                               "VALUES(?,?,?,?,?,?,?,?,?,'prepared',?,?)",
                               (operation_id, scope.execution_id, batch_id, index, call.id, call.name,
                                _json(call.arguments), _fingerprint(call, scope), access.value, _now(), _now()))
                    db.execute("INSERT INTO tool_call_links VALUES(?,?,?)", (scope.execution_id, call.id, operation_id))
                db.execute("INSERT INTO batch_steps VALUES(?,?,?)", (batch_id, index, operation_id))
            return self._batch(db, batch_id)

    def claim(self, operation_id: str, call: ToolCall, scope: OperationScope,
              owner_pid: int, owner_token: str) -> ClaimResult:
        """只有领取成功者可以启动工具；其它调用读取原记录或明确状态。"""
        with self._db(write=True) as db:
            record = self._get(db, operation_id)
            if record.scope != scope or record.arguments_hash != _fingerprint(call, scope) or record.call.id != call.id:
                return ClaimResult(ClaimKind.CONFLICT, record)
            kind = {OperationState.COMPLETED: ClaimKind.REPLAY, OperationState.RUNNING: ClaimKind.IN_PROGRESS,
                    OperationState.UNKNOWN: ClaimKind.UNKNOWN}.get(record.state)
            if kind:
                return ClaimResult(kind, record)
            # 同一 runtime 的未知写操作不能被新的 execution/call ID 绕过。
            blocked = db.execute("SELECT 1 FROM tool_operations o JOIN executions e USING(execution_id) "
                                 "WHERE e.runtime_id=? AND o.access='write' AND o.state='unknown' LIMIT 1",
                                 (scope.runtime_id,)).fetchone()
            if blocked and record.access is ToolAccess.WRITE:
                return ClaimResult(ClaimKind.UNKNOWN, record)
            db.execute("UPDATE tool_operations SET state='running',attempt=attempt+1,owner_pid=?,"
                       "owner_token=?,execution_started=NULL,updated_at=? WHERE operation_id=? AND state='prepared'",
                       (owner_pid, owner_token, _now(), operation_id))
            return ClaimResult(ClaimKind.EXECUTE, self._get(db, operation_id))

    def complete(self, operation_id: str, owner_token: str, result: ToolExecutionResult, *, started: bool) -> None:
        """持有者在返回外层前保存实际结果，失败结果也保留。"""
        with self._db(write=True) as db:
            changed = db.execute("UPDATE tool_operations SET state='completed',result_json=?,execution_started=?,resolution_source='tool',"
                                 "updated_at=? WHERE operation_id=? AND state='running' AND owner_token=?",
                                 (_json(asdict(result)), int(started), _now(), operation_id, owner_token)).rowcount
            if changed != 1:
                raise OperationError("当前调用不再持有工具执行权")

    def mark_unknown(self, operation_id: str, owner_token: str, reason: str,
                     result: ToolExecutionResult | None) -> None:
        """保存已启动操作的不确定结果，阻止同一任务继续写入。"""
        with self._db(write=True) as db:
            changed = db.execute("UPDATE tool_operations SET state='unknown',reason=?,result_json=?,updated_at=? "
                                 "WHERE operation_id=? AND state='running' AND owner_token=?",
                                 (reason, _json(asdict(result)) if result else None, _now(), operation_id, owner_token)).rowcount
            if changed != 1:
                raise OperationError("无法更新不属于当前调用的执行状态")
            db.execute("UPDATE executions SET state='blocked' WHERE execution_id="
                       "(SELECT execution_id FROM tool_operations WHERE operation_id=?)", (operation_id,))

    def finish_unstarted(self, operation_id: str, result: ToolExecutionResult) -> None:
        """取消尚未领取的步骤；不能覆盖另一执行者的结果。"""
        with self._db(write=True) as db:
            db.execute("UPDATE tool_operations SET state='completed',execution_started=0,result_json=?,updated_at=? "
                       "WHERE operation_id=? AND state='prepared'", (_json(asdict(result)), _now(), operation_id))

    def finish_execution(self, execution_id: str) -> None:
        """释放本轮执行者；只有所有步骤都有结果时，才把逻辑任务标记结束。"""
        with self._db(write=True) as db:
            db.execute("UPDATE executions SET owner_released=1 WHERE execution_id=?", (execution_id,))
            db.execute("UPDATE executions SET state='finished' WHERE execution_id=? AND NOT EXISTS "
                       "(SELECT 1 FROM tool_operations WHERE execution_id=? AND state!='completed')",
                       (execution_id, execution_id))

    def execution_active(self, execution_id: str) -> bool:
        """未领取的步骤仍属于原运行；原进程未退出且未释放时不能恢复为取消。"""
        with self._db() as db:
            row = db.execute("SELECT owner_pid,owner_released FROM executions WHERE execution_id=?",
                             (execution_id,)).fetchone()
        return not row["owner_released"] and _process_alive(row["owner_pid"]) is not False

    def pending_batches(self, runtime_id: str) -> tuple[ToolBatchRecord, ...]:
        """读取尚未完整提交到该运行历史的批次。"""
        with self._db() as db:
            rows = db.execute("SELECT batch_id FROM tool_batches b JOIN executions e USING(execution_id) "
                              "WHERE e.runtime_id=? AND history_committed=0 ORDER BY e.created_at,b.batch_number",
                              (runtime_id,)).fetchall()
            return tuple(self._batch(db, row[0]) for row in rows)

    def mark_history_committed(self, batch_id: str) -> None:
        """历史文件同步完成后记录投影标记；恢复仍会先核对文件元数据。"""
        with self._db(write=True) as db:
            db.execute("UPDATE tool_batches SET history_committed=1 WHERE batch_id=?", (batch_id,))

    def recover_orphan(self, operation_id: str) -> OperationRecord:
        """确认原持有者已退出后，把遗留运行中状态改为未知。"""
        record = self.get(operation_id)
        if record.state is OperationState.RUNNING and record.owner_pid and _process_alive(record.owner_pid) is False:
            with self._db(write=True) as db:
                db.execute("UPDATE tool_operations SET state='unknown',reason=?,updated_at=? "
                           "WHERE operation_id=? AND state='running' AND owner_token=?",
                           ("原执行进程已退出，无法确认工具实际效果", _now(), operation_id, record.owner_token))
        return self.get(operation_id)

    def resolutions(self, operation_id: str) -> tuple[dict, ...]:
        """返回用户登记的核查说明，不丢弃之前的判断。"""
        with self._db() as db:
            return tuple(dict(row) for row in db.execute("SELECT verdict,note,previous_result_json,created_at FROM operation_resolutions "
                                                        "WHERE operation_id=? ORDER BY id", (operation_id,)))

    def resolve(self, operation_id: str, verdict: ResolutionVerdict, note: str) -> OperationRecord:
        """用户核查后登记判断；不立即执行工具，不覆盖活跃持有者。"""
        if not note.strip():
            raise OperationError("核查说明不能为空")
        self.recover_orphan(operation_id)
        with self._db(write=True) as db:
            record = self._get(db, operation_id)
            if record.state is OperationState.RUNNING:
                raise OperationError("原执行者尚未确认结束，不能修改操作")
            db.execute("INSERT INTO operation_resolutions(operation_id,verdict,note,previous_result_json,created_at) VALUES(?,?,?,?,?)",
                       (operation_id, verdict.value, note.strip(), _json(asdict(record.result)) if record.result else None, _now()))
            if verdict is ResolutionVerdict.COMPLETED:
                result = ToolExecutionResult(record.call.id, record.call.name, True,
                    "用户核查确认已完成：" + note.strip(), None, None, False, False,
                    len(note.encode("utf-8")), 0, {"source": "user_resolution"})
                db.execute("UPDATE tool_operations SET state='completed',result_json=?,execution_started=NULL,resolution_source='user',"
                           "updated_at=? WHERE operation_id=?", (_json(asdict(result)), _now(), operation_id))
            else:
                db.execute("UPDATE tool_operations SET state='prepared',owner_pid=NULL,owner_token=NULL,"
                           "result_json=NULL,execution_started=NULL,resolution_source='user',updated_at=? "
                           "WHERE operation_id=?", (_now(), operation_id))
            return self._get(db, operation_id)

    def prepare_retry(self, operation_id: str, scope: OperationScope) -> OperationRecord:
        """重新准备已确定未启动的步骤；不会自动重试已执行的失败写操作。"""
        with self._db(write=True) as db:
            record = self._get(db, operation_id)
            if record.scope != scope:
                raise OperationError("重试必须使用原运行、目录和身份")
            if record.state is OperationState.COMPLETED and record.execution_started is False:
                db.execute("UPDATE tool_operations SET state='prepared',owner_pid=NULL,owner_token=NULL,"
                           "result_json=NULL,execution_started=NULL,updated_at=? WHERE operation_id=?",
                           (_now(), operation_id))
            return self._get(db, operation_id)
