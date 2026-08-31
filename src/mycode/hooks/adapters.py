"""把工具调度器的数据转换成执行前和执行后 Hook 事件。"""

from __future__ import annotations

from mycode.hooks.engine import HookEngine
from mycode.models.hooks import HookContext, HookEvent
from mycode.models.tools import ToolErrorCode, ToolExecutionResult
from mycode.tools.interceptors import InterceptionDecision, ToolRunContext


_PATH_ARGUMENTS = ("path", "file_path")


def _file_path(context: ToolRunContext) -> str | None:
    """从文件工具的常见参数中取得第一个字符串路径。

    Args:
        context: 调度器传入的完整工具调用上下文。

    Returns:
        ``path`` 或 ``file_path`` 中第一个字符串值；都没有时返回 None。
    """

    for name in _PATH_ARGUMENTS:
        value = context.invocation.call.arguments.get(name)
        if isinstance(value, str):
            return value
    return None


class PreToolHookInterceptor:
    """在权限审批前执行用户配置的 ``pre_tool_use`` 规则。

    Attributes:
        _engine: 应用装配层创建、主 Agent 与 fork 共用的 Hook 引擎。
    """

    def __init__(self, engine: HookEngine) -> None:
        """保存应用共用的 Hook 引擎。

        Args:
            engine: 包含三层有序规则的应用级 Hook 引擎。

        Returns:
            None。每次调用的 scope 由 `ToolRunContext` 提供。
        """

        self._engine = engine

    async def before_tool(
        self,
        context: ToolRunContext,
    ) -> InterceptionDecision:
        """执行匹配规则，并把 Hook 拒绝转换成普通工具拦截决定。

        Args:
            context: 调度器提供的工具名、完整参数和当前 Hook scope。

        Returns:
            Hook 未拒绝时返回放行，让权限拦截器继续；拒绝时返回 BLOCKED。
        """

        call = context.invocation.call
        result = await self._engine.dispatch(
            HookContext(
                event=HookEvent.PRE_TOOL_USE,
                tool_name=call.name,
                tool_args=call.arguments,
                file_path=_file_path(context),
            ),
            context.hook_scope,
        )
        if not result.rejected:
            return InterceptionDecision.allow()
        return InterceptionDecision.deny(
            ToolErrorCode.BLOCKED,
            result.rejection_reason or "该调用被 Hook 拒绝",
        )


class PostToolHookObserver:
    """在 executor 返回后只读派发 ``post_tool_use``，不修改工具结果。

    Attributes:
        _engine: 应用装配层创建、主 Agent 与 fork 共用的 Hook 引擎。
    """

    def __init__(self, engine: HookEngine) -> None:
        """保存应用共用的 Hook 引擎。

        Args:
            engine: 已装入三层规则、供主 Agent 与 fork 共用的 Hook 引擎。

        Returns:
            无返回值；每次工具调用所属的 scope 由观察器参数提供。
        """

        self._engine = engine

    async def after_tool(
        self,
        context: ToolRunContext,
        result: ToolExecutionResult,
    ) -> None:
        """把真实工具结果摘要交给匹配的执行后 Hook。

        Args:
            context: executor 本次实际处理的工具调用与所属 scope。
            result: 即将原样回灌模型的成功或失败工具结果。

        Returns:
            None。Hook 的任何结果都不会替换 `result`。
        """

        call = context.invocation.call
        message = result.content if result.success else result.error_message
        await self._engine.dispatch(
            HookContext(
                event=HookEvent.POST_TOOL_USE,
                tool_name=call.name,
                tool_args=call.arguments,
                file_path=_file_path(context),
                message=message or "",
                error=None if result.success else result.error_message,
            ),
            context.hook_scope,
        )
