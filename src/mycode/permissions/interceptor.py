"""把五层权限决策接入现有工具执行前拦截协议。"""

from __future__ import annotations

import asyncio

from mycode.errors import ConfigError
from mycode.models.permissions import (
    ApprovalChoice,
    PermissionApprover,
    PermissionOutcome,
    PermissionRequest,
    PermissionStore,
    PermissionTool,
)
from mycode.models.tools import ToolErrorCode, ToolSource
from mycode.permissions.blacklist import match_dangerous_command
from mycode.permissions.policy import PermissionController, PermissionPolicy
from mycode.permissions.rules import format_exact_rule_expression
from mycode.permissions.operations import (
    mcp_permission_operation_for_call,
    permission_operation_for_call,
    skill_permission_operation_for_call,
)
from mycode.tools.base import ToolContext, ToolFailure
from mycode.tools.builtin.paths import preflight_tool_path
from mycode.tools.interceptors import InterceptionDecision, ToolRunContext
from mycode.tools.registry import ToolRegistry


class AgentPermissionApprover:
    """在前台转发真实审批，在后台把所有 ASK 结果改为拒绝。

    Attributes:
        _foreground_approver: 前台运行时真正显示权限请求的终端审批器。
        background: ``True`` 表示当前子 Agent 不允许弹出或等待人工审批。
        _background_event: 正在等待审批时接收“已移交后台”的通知事件。
    """

    def __init__(
        self,
        foreground_approver: PermissionApprover,
        *,
        background: bool = False,
    ) -> None:
        """保存前台审批入口和初始运行方式。

        Args:
            foreground_approver: 前台子 Agent 请求权限时使用的真实 UI 审批器。
            background: 是否从创建起就处于非交互后台。

        Returns:
            不返回数据。
        """

        self._foreground_approver = foreground_approver
        self.background = background
        # 正在等待 UI 的请求同时监听该事件；移交后台会立即唤醒并拒绝。
        self._background_event = asyncio.Event()
        if background:
            self._background_event.set()

    def move_to_background(self) -> None:
        """让后续权限 ASK 直接拒绝，不再调用终端 UI。

        Returns:
            不返回数据；已经完成的批准不会复制成新的 SESSION 规则。
        """

        self.background = True
        self._background_event.set()

    async def request_permission(
        self,
        request: PermissionRequest,
    ) -> ApprovalChoice:
        """根据当前前后台状态处理一次人工审批请求。

        Args:
            request: 权限系统生成的具体操作和可保存规则表达式。

        Returns:
            后台固定返回 ``DENY``；前台返回真实审批器的用户选择。
        """

        if self.background:
            return ApprovalChoice.DENY
        approval = asyncio.create_task(
            self._foreground_approver.request_permission(request)
        )
        moved = asyncio.create_task(self._background_event.wait())
        try:
            done, _ = await asyncio.wait(
                {approval, moved},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if moved in done:
                approval.cancel()
                await asyncio.gather(approval, return_exceptions=True)
                return ApprovalChoice.DENY
            return await approval
        finally:
            moved.cancel()
            await asyncio.gather(moved, return_exceptions=True)


class PermissionInterceptor:
    def __init__(
        self,
        registry: ToolRegistry,
        context: ToolContext,
        policy: PermissionPolicy,
        controller: PermissionController,
        approver: PermissionApprover,
        store: PermissionStore,
    ) -> None:
        self._registry = registry
        self._context = context
        self._policy = policy
        self._controller = controller
        self._approver = approver
        self._store = store

    async def before_tool(
        self,
        context: ToolRunContext,
    ) -> InterceptionDecision:
        """在工具执行前检查参数、路径和权限，并返回是否允许执行

        函数会拒绝无效参数、危险 Shell 命令和不符合路径要求的调用；权限规则
        无法直接决定时，会询问用户，并根据用户选择保存本会话或永久授权

        Args:
            context: 当前工具调用、运行选项和所属 Hook scope。本拦截器
                使用其中的调用数据，权限判断本身不读取 Hook 状态。

        Returns:
            允许执行或拒绝执行的拦截结果。权限检查发生异常时返回拒绝
        """

        invocation = context.invocation
        call = invocation.call

        # 判断当前调用的工具，如果是一个未注册的工具，权限拦截器不处理它，直接通过
        if self._registry.get(call.name) is None:
            return InterceptionDecision.allow()

        #校验参数，参数有问题，直接拒绝
        validation_error = self._registry.validate_arguments(
            call.name,
            call.arguments,
        )
        if validation_error is not None:
            return InterceptionDecision.deny(
                ToolErrorCode.INVALID_ARGUMENTS,
                validation_error,
            )

        try:
            # MCP 工具按完整规范化参数授权；内置工具保持原命令/路径适配。
            source = self._registry.source_for(call.name)
            if source is ToolSource.SYSTEM:
                return InterceptionDecision.allow()
            if source is ToolSource.MCP:
                operation = mcp_permission_operation_for_call(call)
            elif source is ToolSource.SKILL:
                policy = self._registry.execution_policy(call.name)
                if policy is None:
                    return InterceptionDecision.deny(
                        ToolErrorCode.BLOCKED,
                        f"Skill 工具 {call.name} 缺少执行策略",
                    )
                operation = skill_permission_operation_for_call(
                    call,
                    policy,
                    invocation.access,
                )
            else:
                operation = permission_operation_for_call(call)
            if operation is None:
                # 转换结果为空
                return InterceptionDecision.deny(
                    ToolErrorCode.BLOCKED,
                    f"已注册工具 {call.name} 尚未配置待授权操作，已按安全策略拒绝",
                )

            if operation.tool is PermissionTool.SHELL:
                # 如果是Shell命令，命中黑名单就拒绝
                blocked = match_dangerous_command(operation.match_value)
                if blocked is not None:
                    return InterceptionDecision.deny(
                        ToolErrorCode.BLOCKED,
                        f"危险命令黑名单已拒绝：{blocked.reason}",
                    )
            elif operation.path_value is not None:
                # 检查工具调用的路径，是不是跑到工作区外，访问不存在的文件等
                preflight_tool_path(
                    self._context,
                    call.name,
                    operation.path_value,
                )
            # 判断当前调用的工具操作是该拒绝、通过还是人工审核
            decision = self._policy.decide(operation)

            if decision.outcome is PermissionOutcome.ALLOW:
                return InterceptionDecision.allow()
            if decision.outcome is PermissionOutcome.DENY:
                return InterceptionDecision.deny(
                    ToolErrorCode.BLOCKED,
                    decision.message,
                )

            # 得到用户的确认的结果
            choice = await self._approver.request_permission(
                PermissionRequest(
                    operation,
                    format_exact_rule_expression(operation),
                )
            )
            if choice is ApprovalChoice.DENY:
                return InterceptionDecision.deny(
                    ToolErrorCode.BLOCKED,
                    "用户拒绝了本次工具调用",
                )

            # 判断用户的确认结果
            if choice is ApprovalChoice.ALLOW_ONCE:
                # 用户选择仅允许一次，则拦截器放行，不存规则
                return InterceptionDecision.allow()
            if choice is ApprovalChoice.ALLOW_SESSION:
                #用户选择进会话允许
                # 将当前操作生成一条规则，放在self._controller._session_rules中
                self._controller.allow_for_session(operation)
                return InterceptionDecision.allow()
            if choice is ApprovalChoice.ALLOW_PERMANENT:
                # 用户选择永久允许，则保存到对应的yaml文件中，并返回新的Local层权限配置
                try:
                    layer = self._store.allow_permanently(operation)
                except ConfigError as exc:
                    return InterceptionDecision.deny(
                        ToolErrorCode.BLOCKED,
                        f"永久授权写入失败：{exc}",
                    )
                # 拿最新的Local层权限配置进行更新
                self._controller.replace_local_layer(layer)
                return InterceptionDecision.allow()
            return InterceptionDecision.deny(
                ToolErrorCode.BLOCKED,
                "人工确认返回了未知选择，已按拒绝处理",
            )
        except ToolFailure as exc:
            return InterceptionDecision.deny(exc.code, str(exc))
        except Exception:
            return InterceptionDecision.deny(
                ToolErrorCode.BLOCKED,
                "权限检查因内部错误而拒绝了工具调用",
            )
