"""按声明顺序执行匹配的 Hook，并隔离动作故障。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from mycode.agent.instructions import RuntimeInstructionManager
from mycode.hooks.actions import HookActionRunner
from mycode.hooks.conditions import group_matches
from mycode.hooks.runtime import HookRunScope
from mycode.models.hooks import (
    AgentHookAction,
    CommandHookAction,
    HookActionResult,
    HookDefinition,
    HookDispatchResult,
    HookContext,
    HookEvent,
    HttpHookAction,
    PromptHookAction,
)


_LOGGER = logging.getLogger(__name__)
_DEFAULT_REJECTION = "该调用被 Hook 拒绝"


def _action_name(hook: HookDefinition) -> str:
    """返回日志使用的固定动作类型，不读取动作中的敏感字段。

    Args:
        hook: 已校验且即将执行的 Hook 定义。

    Returns:
        ``command``、``prompt``、``http`` 或 ``agent`` 之一。
    """

    action = hook.action
    if isinstance(action, CommandHookAction):
        return "command"
    if isinstance(action, PromptHookAction):
        return "prompt"
    if isinstance(action, HttpHookAction):
        return "http"
    assert isinstance(action, AgentHookAction)
    return "agent"


class HookEngine:
    """执行全部已校验规则，并管理每个 Agent 的 Hook scope。

    CLI 创建一个引擎供主会话和 Skill fork 共用。规则定义可以共用，但
    `create_scope` 返回的 once、提示词和后台任务状态彼此隔离。

    Attributes:
        hooks: 按用户、项目、本地和文件声明顺序排列的不可变规则。
        _actions: 执行 command、prompt、http 和 agent 占位动作的应用级运行器。
        _scopes: 尚未结束的主会话与 Skill fork 运行作用域。
        _closed: 引擎及其动作运行器是否已经完成关闭。
    """

    def __init__(
        self,
        hooks: Sequence[HookDefinition],
        action_runner: HookActionRunner,
    ) -> None:
        """保存已校验规则和具体动作运行器。

        Args:
            hooks: 配置加载阶段原子生成的有序规则。
            action_runner: 真正执行命令、提示和 HTTP 的应用级对象。

        Returns:
            None。空规则序列也会创建可直接派发的引擎。
        """

        self.hooks = tuple(hooks)
        self._actions = action_runner
        self._scopes: set[HookRunScope] = set()
        self._closed = False

    def create_scope(
        self,
        instruction_manager: RuntimeInstructionManager,
    ) -> HookRunScope:
        """为一个主会话或 Skill fork 创建独立运行状态。

        Args:
            instruction_manager: 该 Agent 自己的运行时提示管理器。

        Returns:
            需要传给 Agent runner 和工具调度器的 `HookRunScope`。
        """

        if self._closed:
            raise RuntimeError("HookEngine 已关闭")
        scope = HookRunScope(instruction_manager)
        self._scopes.add(scope)
        return scope

    async def _run_one(
        self,
        hook: HookDefinition,
        context: HookContext,
        scope: HookRunScope,
    ) -> HookActionResult:
        """执行一条动作，并把意外异常转换为普通失败结果。

        Args:
            hook: 当前命中的 Hook 定义，用于取得动作和日志来源。
            context: 生命周期接入点生成的事件数据。
            scope: 动作所属主会话或 Skill fork 的运行状态。

        Returns:
            动作的成功状态和有限输出。意外异常只保留异常类型，不会离开 Hook 引擎。
        """

        fields = {
            "hook_source": str(hook.source.path),
            "hook_id": hook.source.hook_id,
            "hook_event": hook.event.value,
            "hook_action": _action_name(hook),
        }
        try:
            result = await self._actions.run(hook.action, context, scope)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result = HookActionResult(
                False,
                error=f"动作执行异常：{type(exc).__name__}",
            )
        if result.success:
            _LOGGER.info("Hook 执行成功", extra=fields)
        else:
            _LOGGER.error(
                "Hook 执行失败：%s",
                result.error or "未知原因",
                extra=fields,
            )
        return result

    async def _run_background(
        self,
        hook: HookDefinition,
        context: HookContext,
        scope: HookRunScope,
    ) -> None:
        """完成一条异步动作，并在结束后更新 once 状态。

        Args:
            hook: 已经匹配并被放到后台执行的 Hook 定义。
            context: 启动后台任务时保存的事件数据。
            scope: 持有后台任务和 once 状态的运行作用域。

        Returns:
            无返回值；普通失败留在日志中，取消会继续向关闭流程传播。
        """

        try:
            result = await self._run_one(hook, context, scope)
            if hook.once:
                await scope.finish_once(
                    hook.source,
                    completed=result.success,
                )
        except asyncio.CancelledError:
            if hook.once:
                await scope.finish_once(hook.source, completed=False)
            raise

    async def dispatch(
        self,
        context: HookContext,
        scope: HookRunScope,
    ) -> HookDispatchResult:
        """执行当前事件中全部匹配的 Hook。

        Args:
            context: 生命周期接入点生成的真实事件数据。
            scope: 触发事件的主会话或 Skill fork 运行状态。

        Returns:
            `pre_tool_use` 命中拒绝时返回原因；其他情况返回未拒绝结果。
        """

        for hook in self.hooks:
            if hook.event is not context.event:
                continue
            if hook.condition is not None and not group_matches(
                hook.condition, context
            ):
                continue
            if hook.once and not await scope.reserve_once(hook.source):
                continue
            if hook.async_mode:
                task = asyncio.create_task(
                    self._run_background(hook, context, scope)
                )
                scope.add_background_task(task)
                continue
            try:
                result = await self._run_one(hook, context, scope)
            except asyncio.CancelledError:
                if hook.once:
                    await scope.finish_once(hook.source, completed=False)
                raise
            if hook.once:
                await scope.finish_once(
                    hook.source,
                    completed=result.success or hook.reject,
                )
            if hook.reject:
                reason = result.output.strip() if result.success else ""
                return HookDispatchResult(
                    rejected=True,
                    rejection_reason=reason or _DEFAULT_REJECTION,
                )
        return HookDispatchResult()

    async def close_scope(self, scope: HookRunScope) -> None:
        """关闭并忘记一个主会话或 fork 的后台 Hook。

        Args:
            scope: 会话切换或 fork 结束时不再使用的运行作用域。

        Returns:
            无返回值；不属于当前引擎或已经关闭的 scope 会被直接忽略。
        """

        if scope not in self._scopes:
            return
        await scope.close()
        self._scopes.discard(scope)

    async def close(self) -> None:
        """关闭所有 scope 和动作运行器持有的网络资源。

        Returns:
            无返回值；重复调用不会再次关闭 scope 或 HTTP 客户端。
        """

        if self._closed:
            return
        self._closed = True
        for scope in tuple(self._scopes):
            await scope.close()
        self._scopes.clear()
        await self._actions.close()
