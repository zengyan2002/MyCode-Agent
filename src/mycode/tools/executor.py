"""为已注册工具提供有界执行和失败隔离。"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace

from mycode.models.messages import ToolCall
from mycode.models.operations import OperationScope, OperationState, operation_failure
from mycode.models.tools import ToolAccess, ToolErrorCode, ToolExecutionResult
from mycode.persistence.operations import OperationError, OperationStore, operation_io
from mycode.tools.base import ToolContext, ToolFailure, ToolOutput
from mycode.tools.registry import ToolRegistry
from mycode.tools.builtin.files import ReadFileTool, WriteFileTool, EditFileTool
from mycode.tools.builtin.command import ExecuteCommandTool
from mycode.tools.builtin.paths import WorkspacePaths
from mycode.tools.file_journal import FileWriteJournal
from mycode.models.tools import ToolSource

# 负责校验并执行一个工具调用，然后返回包含错误信息和耗时的统一结果
# 多个工具如何并发、排序和取消，由 ToolScheduler 负责
class ToolExecutor:
    """校验并执行单个工具，把异常转换成模型可读的统一结果。"""

    def __init__(
        self,
        registry: ToolRegistry,
        context: ToolContext,
        *,
        store: OperationStore,
        timeout_seconds: float = 30.0,
    ) -> None:
        """创建执行器并保存注册表、工具上下文和默认超时。

        Args:
            registry: 查找工具、参数 Schema 和单工具策略的注册表。
            context: 每次工具调用共享的工作区与资源访问状态。
            timeout_seconds: 非 Skill 工具使用的默认超时秒数。

        Raises:
            ValueError: timeout_seconds 不是正数。
        """

        if timeout_seconds <= 0:
            raise ValueError("工具超时时间必须为正数")
        self._registry = registry
        self._context = context
        self.store = store
        self._timeout_seconds = timeout_seconds

    @property
    def context(self) -> ToolContext:
        """返回该执行器每次调用都会传给工具的上下文。

        Returns:
            包含工作区、文件缓存、Skill 路由和 MCP 激活状态的 ToolContext。
            ToolScheduler 用它让 AgentTurnRequest 与 tool_search 读取同一状态。
        """

        return self._context

    async def execute(self, call: ToolCall, *, operation_id: str,
                      scope: OperationScope, owner_token: str,
                      verification_only: bool = False) -> ToolExecutionResult:
        """执行一次模型工具调用。

        Args:
            call: 模型返回的工具名、调用 ID 和 JSON 参数。

        Returns:
            包含成功正文或固定错误、耗时和工具身份的 ToolExecutionResult。
        """

        # 计时覆盖查找、参数校验和实际执行，使 UI 看到的是本次调用从进入
        # 执行边界到形成结果的总耗时，而不只是工具函数内部耗时。
        started = time.monotonic()
        record = await operation_io(self.store.get, operation_id)
        if (record.state is not OperationState.RUNNING or record.owner_token != owner_token
                or record.scope != scope or record.call != call
                or self._context.workspace_root != scope.workspace_root):
            raise OperationError("工具执行身份、目录或领取记录不匹配")
        tool = self._registry.get(call.name)
        if tool is None:
            result = self._result(
                call,
                ToolOutput.fail(
                    ToolErrorCode.UNKNOWN_TOOL,
                    f"未知工具：{call.name}",
                ),
                started,
            )
            await operation_io(self.store.complete, operation_id, owner_token, result, started=False)
            return result

        validation_error = self._registry.validate_arguments(
            call.name,
            call.arguments,
        )
        if validation_error is not None:
            # Schema 失败必须在调用 tool.execute 前返回，保证无效模型参数
            # 绝不会到达文件系统或 Shell 副作用代码。
            result = self._result(
                call,
                ToolOutput.fail(
                    ToolErrorCode.INVALID_ARGUMENTS,
                    validation_error,
                ),
                started,
            )
            await operation_io(self.store.complete, operation_id, owner_token, result, started=False)
            return result

        policy = self._registry.execution_policy(call.name)
        timeout_seconds = (
            policy.timeout_seconds
            if policy is not None and policy.timeout_seconds is not None
            else self._timeout_seconds
        )
        cancelled = False
        context = self._context
        candidates = ()
        builtin = self._registry.source_for(call.name) is ToolSource.BUILTIN
        if builtin and isinstance(tool, ExecuteCommandTool):
            context = replace(context, operation_record=record)
        if builtin and isinstance(tool, (WriteFileTool, EditFileTool)):
            context = replace(context, file_write_journal=FileWriteJournal(
                self.store, operation_id, owner_token, record.attempt))
        verifying_read = verification_only and builtin and isinstance(tool, ReadFileTool)
        if verifying_read:
            context = replace(context, fresh_file_read=True)
        try:
            # asyncio.timeout 会先向工具协程注入 CancelledError，使命令工具
            # 有机会终止进程树，再由下面的分支转换成普通超时结果。
            async with asyncio.timeout(timeout_seconds):
                if verifying_read:
                    from mycode.tools.recovery import OperationRecovery
                    path, _ = WorkspacePaths(context.workspace_root).readable_file(
                        str(call.arguments["path"]), context.user_memory_root, context.skill_resources)
                    candidates = await OperationRecovery(self.store).file_candidates(scope, path)
                    # 只有本地真实的文件写工具才有可自动确认的语义。
                    candidates = tuple(c for c in candidates if
                        isinstance(self._registry.get(c.record.call.name), (WriteFileTool, EditFileTool))
                        and self._registry.source_for(c.record.call.name) is ToolSource.BUILTIN)
                output = await tool.execute(call.arguments, context)
        except OperationError:
            try:
                await operation_io(self.store.mark_unknown, operation_id, owner_token,
                                   "文件操作或核查记录保存失败", None)
            except OperationError:
                pass  # 数据库仍不可用时保留原领取记录，不允许重新领取。
            return operation_failure(call, ToolErrorCode.OPERATION_STORAGE_ERROR,
                                     f"文件操作或核查记录未能保存，已停止执行。操作：{operation_id}")
        except TimeoutError:
            output = ToolOutput.fail(
                ToolErrorCode.TIMEOUT,
                f"工具执行超过 {timeout_seconds:g} 秒限制",
            )
        except ToolFailure as exc:
            # ToolFailure 表示预期内、可安全展示的领域错误，例如路径越界或
            # 文件不存在；其他异常必须走下面的固定脱敏消息。
            output = ToolOutput.fail(exc.code, str(exc))
        except asyncio.CancelledError:
            cancelled = True
            output = ToolOutput.fail(ToolErrorCode.CANCELLED, "工具执行期间收到取消，效果可能已经发生")
        except Exception:
            # 内部异常可能包含路径、环境变量或依赖库细节，因此模型可见的
            # 消息刻意保持笼统，避免泄露运行环境信息。
            output = ToolOutput.fail(
                ToolErrorCode.INTERNAL_ERROR,
                "工具因未预期的内部错误而失败",
            )
        result = self._result(call, output, started)
        uncertain = record.access is ToolAccess.WRITE and output.error_code in {
            ToolErrorCode.TIMEOUT, ToolErrorCode.CANCELLED, ToolErrorCode.IO_ERROR,
            ToolErrorCode.REMOTE_ERROR, ToolErrorCode.INTERNAL_ERROR,
        }
        try:
            if uncertain:
                await operation_io(self.store.mark_unknown, operation_id, owner_token,
                                   result.error_message or "工具效果无法确认", result)
                return replace(result, error_code=ToolErrorCode.OPERATION_UNKNOWN,
                    error_message=f"工具可能已经生效，结果无法确认。操作：{operation_id}。{result.error_message}")
            if verifying_read and not cancelled:
                return await operation_io(self.store.complete_verification_read,
                                          operation_id, owner_token, result, candidates)
            await operation_io(self.store.complete, operation_id, owner_token, result, started=True)
        except (OperationError, asyncio.CancelledError):
            # 本体已结束，提交线程可能已经成功；先查记录，不能覆盖已提交结果。
            try:
                saved = await operation_io(self.store.get, operation_id)
                if saved.state is OperationState.COMPLETED:
                    return saved.result
                if saved.state is OperationState.RUNNING:
                    await operation_io(self.store.mark_unknown, operation_id, owner_token,
                                       "工具已返回，但结果未能保存", result)
            except OperationError:
                pass  # 保留 RUNNING，后续不能重新领取。
            return operation_failure(call, ToolErrorCode.OPERATION_STORAGE_ERROR,
                                     f"工具结果未能确认保存，已停止执行。操作：{operation_id}")
        if cancelled:
            raise asyncio.CancelledError
        return result

    def _result(
        self,
        call: ToolCall,
        output: ToolOutput,
        started: float,
    ) -> ToolExecutionResult:
        """把工具直接输出补齐为带调用身份和耗时的完整结果。

        这里不再截断正文。Agent Loop 会把同一条 assistant 消息对应的全部
        工具结果交给上下文管理器，由它统一决定哪些结果需要存盘。
        """

        return ToolExecutionResult(
            tool_call_id=call.id,
            tool_name=call.name,
            success=output.success,
            content=output.content,
            error_code=output.error_code,
            error_message=output.error_message,
            timed_out=output.error_code is ToolErrorCode.TIMEOUT,
            truncated=output.truncated,
            original_size_bytes=output.original_size_bytes,
            duration_ms=max(0, round((time.monotonic() - started) * 1000)),
            metadata=output.metadata,
        )
