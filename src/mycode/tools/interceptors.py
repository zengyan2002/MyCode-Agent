"""工具执行前拦截与执行后只读观察。"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from mycode.models.events import AgentRunOptions
from mycode.hooks.runtime import HookRunScope
from mycode.models.tools import (
    ToolAccess,
    ToolErrorCode,
    ToolExecutionResult,
    ToolInvocation,
)


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolRunContext:
    """保存一次工具调用进入拦截链时需要的全部真实数据。

    `ToolScheduler` 为每个调用创建该对象。Plan、Skill 信任、Hook 和权限
    拦截器从中读取同一份调用与运行选项，Hook 适配器另外使用 scope 保存
    当前主会话或 fork 的 once、提示词和后台任务。

    Attributes:
        invocation: 工具名、完整参数、读写分类、轮次和调用位置。
        options: 当前 Agent 的 Plan、并发和轮次设置。
        hook_scope: 发起本次调用的主会话或 Skill fork Hook 状态。
    """

    invocation: ToolInvocation
    options: AgentRunOptions
    hook_scope: HookRunScope

# 拦截器用值对象返回决定，而不是通过异常拒绝；这样被策略阻止也能变成
# 普通结构化工具结果回灌模型，由模型解释并结束当前计划。
@dataclass(frozen=True)
class InterceptionDecision:
    """保存执行前拦截器对一个工具调用作出的决定。

    Attributes:
        allowed: True 表示继续调用后续拦截器或执行器。
        error_code: 拒绝时回灌模型的工具错误分类；放行时为 None。
        message: 拒绝时解释原因的用户可读文本；放行时为 None。
    """

    allowed: bool
    error_code: ToolErrorCode | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        """校验放行决定不带错误、拒绝决定同时带错误码和原因。"""

        if self.allowed and (
            self.error_code is not None or self.message is not None
        ):
            raise ValueError("放行决定不能包含错误信息")
        if not self.allowed and (
            self.error_code is None or not self.message
        ):
            raise ValueError("拒绝决定必须包含错误码和错误消息")

    @classmethod
    def allow(cls) -> InterceptionDecision:
        """创建一个不携带错误信息的放行决定。

        Returns:
            允许工具调用继续进入下一层的决定。
        """

        return cls(True)

    @classmethod
    def deny(
        cls,
        error_code: ToolErrorCode,
        message: str,
    ) -> InterceptionDecision:
        """创建一个带结构化错误码和原因的拒绝决定。

        Args:
            error_code: 模型收到的工具错误分类。
            message: 告诉模型本次调用为何没有执行的原因。

        Returns:
            阻止后续拦截器和执行器运行的决定。
        """

        return cls(False, error_code, message)

# 执行前拦截器可以阻止副作用，调用顺序与注册顺序一致。
class BeforeToolInterceptor(Protocol):
    """定义工具执行前可以放行或拒绝调用的组件协议。"""

    async def before_tool(
        self,
        context: ToolRunContext,
    ) -> InterceptionDecision:
        """检查完整工具上下文，并返回是否允许继续执行。"""

        ...

# 执行后观察器只能读取结果，不能改变即将回灌模型的内容。
class AfterToolObserver(Protocol):
    """定义 executor 返回后只读观察真实工具结果的组件协议。"""

    async def after_tool(
        self,
        context: ToolRunContext,
        result: ToolExecutionResult,
    ) -> None:
        """读取调用和结果执行旁路动作，不得替换结果。"""

        ...

# Plan 模式按工具声明的 access 分类，而不是分析命令字符串。特别是
# execute_command 永远属于 WRITE，避免“看起来只读”的 Shell 绕过策略。
class PlanOnlyInterceptor:
    """在 Plan 模式下阻止所有声明为 WRITE 的工具。"""

    async def before_tool(
        self,
        context: ToolRunContext,
    ) -> InterceptionDecision:
        """根据运行选项和工具访问分类决定是否放行。

        Args:
            context: 包含 Plan 开关与工具读写分类的运行上下文。

        Returns:
            非 Plan 模式或 READ 工具返回放行；其他情况返回 BLOCKED。
        """

        if (
            context.options.plan_only
            and context.invocation.access is ToolAccess.WRITE
        ):
            return InterceptionDecision.deny(
                ToolErrorCode.BLOCKED,
                "当前请求处于 Plan 模式，写工具已被拦截；如需执行写操作，"
                "请先关闭 Plan 模式，再发送新的请求",
            )
        return InterceptionDecision.allow()


async def notify_observers(
    observers: Sequence[AfterToolObserver],
    context: ToolRunContext,
    result: ToolExecutionResult,
) -> None:
    """依次通知执行后观察器，并隔离每个观察器的普通异常。

    Args:
        observers: 按装配顺序排列的只读执行后观察器。
        context: executor 刚刚处理的工具调用上下文。
        result: 即将原样回灌模型的工具执行结果。

    Returns:
        无返回值；某个观察器失败不会阻止后续观察器。
    """

    # 观察器属于旁路能力，失败不能把已经完成的工具改判为失败。逐个隔离
    # 后继续通知其余观察器，同时避免记录可能含敏感参数的原异常。
    for observer in observers:
        try:
            await observer.after_tool(context, result)
        except Exception:
            # 观察器异常可能包含工具参数或密钥，因此只记录固定错误文本。
            _LOGGER.error("工具执行后的观察器运行失败")
