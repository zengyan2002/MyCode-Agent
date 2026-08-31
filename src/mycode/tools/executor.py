"""为已注册工具提供有界执行和失败隔离。"""

from __future__ import annotations

import asyncio
import time

from mycode.models.messages import ToolCall
from mycode.models.tools import ToolErrorCode, ToolExecutionResult
from mycode.tools.base import ToolContext, ToolFailure, ToolOutput
from mycode.tools.registry import ToolRegistry

# 负责校验并执行一个工具调用，然后返回包含错误信息和耗时的统一结果
# 多个工具如何并发、排序和取消，由 ToolScheduler 负责
class ToolExecutor:
    """校验并执行单个工具，把异常转换成模型可读的统一结果。"""

    def __init__(
        self,
        registry: ToolRegistry,
        context: ToolContext,
        *,
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
        self._timeout_seconds = timeout_seconds

    @property
    def context(self) -> ToolContext:
        """返回该执行器每次调用都会传给工具的上下文。

        Returns:
            包含工作区、文件缓存、Skill 路由和 MCP 激活状态的 ToolContext。
            ToolScheduler 用它让 AgentTurnRequest 与 tool_search 读取同一状态。
        """

        return self._context

    async def execute(self, call: ToolCall) -> ToolExecutionResult:
        """执行一次模型工具调用。

        Args:
            call: 模型返回的工具名、调用 ID 和 JSON 参数。

        Returns:
            包含成功正文或固定错误、耗时和工具身份的 ToolExecutionResult。
        """

        # 计时覆盖查找、参数校验和实际执行，使 UI 看到的是本次调用从进入
        # 执行边界到形成结果的总耗时，而不只是工具函数内部耗时。
        started = time.monotonic()
        tool = self._registry.get(call.name)
        if tool is None:
            return self._result(
                call,
                ToolOutput.fail(
                    ToolErrorCode.UNKNOWN_TOOL,
                    f"未知工具：{call.name}",
                ),
                started,
            )

        validation_error = self._registry.validate_arguments(
            call.name,
            call.arguments,
        )
        if validation_error is not None:
            # Schema 失败必须在调用 tool.execute 前返回，保证无效模型参数
            # 绝不会到达文件系统或 Shell 副作用代码。
            return self._result(
                call,
                ToolOutput.fail(
                    ToolErrorCode.INVALID_ARGUMENTS,
                    validation_error,
                ),
                started,
            )

        policy = self._registry.execution_policy(call.name)
        timeout_seconds = (
            policy.timeout_seconds
            if policy is not None and policy.timeout_seconds is not None
            else self._timeout_seconds
        )
        try:
            # asyncio.timeout 会先向工具协程注入 CancelledError，使命令工具
            # 有机会终止进程树，再由下面的分支转换成普通超时结果。
            async with asyncio.timeout(timeout_seconds):
                output = await tool.execute(call.arguments, self._context)
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
            raise
        except Exception:
            # 内部异常可能包含路径、环境变量或依赖库细节，因此模型可见的
            # 消息刻意保持笼统，避免泄露运行环境信息。
            output = ToolOutput.fail(
                ToolErrorCode.INTERNAL_ERROR,
                "工具因未预期的内部错误而失败",
            )
        return self._result(call, output, started)

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
