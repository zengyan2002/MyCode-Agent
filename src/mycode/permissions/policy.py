"""当前会话权限状态与规则、模式决策。"""

from __future__ import annotations

from mycode.models.permissions import (
    LoadedPermissionSettings,
    PermissionDecision,
    PermissionEffect,
    PermissionLayer,
    PermissionMode,
    PermissionOutcome,
    PermissionRule,
    PermissionScope,
    PermissionOperation,
)
from mycode.permissions.rules import (
    PermissionRuleResolver,
    make_exact_allow_rule,
)


class LocalPermissionState:
    """保存当前进程内所有父子 Agent 共用的最新 LOCAL 权限层。

    Attributes:
        layer: 从 permissions.local.yaml 加载或永久授权后更新的规则层。
    """

    def __init__(self, layer: PermissionLayer) -> None:
        """创建共享状态并校验它确实是 LOCAL 层。

        Args:
            layer: 当前工作区的本地权限规则层。

        Raises:
            ValueError: 传入的不是 LOCAL 作用域。
        """

        if layer.scope is not PermissionScope.LOCAL:
            raise ValueError("共享本地权限状态必须使用 LOCAL 层")
        self.layer = layer


class PermissionController:
    """
    保存当前会话中会变化的权限状态：
        当前 PermissionMode
        SESSION 会话规则
        最新 LOCAL 权限层
    """
    def __init__(
        self,
        settings: LoadedPermissionSettings,
        *,
        local_state: LocalPermissionState | None = None,
        initial_mode: PermissionMode | None = None,
    ) -> None:
        """创建一个权限控制器，并为它分配独立的 SESSION 规则列表。

        Args:
            settings: 用户、项目、本地静态规则和默认权限模式。
            local_state: 父子 Agent 需要共同观察永久授权时传入的共享对象；
                未传时用 settings.local 新建。
            initial_mode: 子 Agent 创建时解析好的权限模式；未传时使用配置值。
        """

        self._mode = initial_mode or settings.initial_mode
        self._session_rules: list[PermissionRule] = []
        self._local_state = local_state or LocalPermissionState(settings.local)
        self._settings = settings

    @property
    def mode(self) -> PermissionMode:
        return self._mode

    def set_mode(self, mode: PermissionMode) -> None:
        self._mode = mode

    def session_rules(self) -> tuple[PermissionRule, ...]:
        return tuple(self._session_rules)

    def local_layer(self) -> PermissionLayer:
        return self._local_state.layer

    @property
    def local_state(self) -> LocalPermissionState:
        """返回父子 Agent 共同读取的 LOCAL 规则容器。

        Returns:
            永久授权写入后会原地更新的 LocalPermissionState。
        """

        return self._local_state

    def child(self, mode: PermissionMode) -> "PermissionController":
        """创建共享静态规则但不继承临时批准的子 Agent 控制器。

        Args:
            mode: 创建子 Agent 时已经解析完成的 strict、default 或 allow。

        Returns:
            SESSION 规则为空、LOCAL 状态与父控制器共享的新控制器。
        """

        return PermissionController(
            self._settings,
            local_state=self._local_state,
            initial_mode=mode,
        )


    def allow_for_session(self, operation: PermissionOperation) -> None:
        """
        根据当前具体操作生成一条精确匹配的 SESSION 级允许规则，
        并将其添加到当前会话的权限规则列表中。

        后续出现工具类型和匹配目标完全相同的操作时，将直接允许；
        规则只保存在内存中，当前会话结束后失效。
        """
        rule = make_exact_allow_rule(
            operation,
            scope=PermissionScope.SESSION,
            source="当前会话",
        )
        self._session_rules = [
            existing
            for existing in self._session_rules
            if not (
                existing.tool is rule.tool
                and existing.pattern == rule.pattern
            )
        ]
        self._session_rules.append(rule)

    def replace_local_layer(self, layer: PermissionLayer) -> None:
        """
        用最新的 LOCAL 权限层替换当前会话中的本地配置快照。

        永久授权写入磁盘后调用本方法，使新增规则立即参与后续
        权限决策，无需重启程序。
        """
        if layer.scope is not PermissionScope.LOCAL:
            raise ValueError("只能用 local 权限层替换本地规则快照")
        self._local_state.layer = layer


class PermissionPolicy:
    """
    决定一个工具操作应该直接允许、直接拒绝，还是交给用户确认
    """
    def __init__(
        self,
        resolver: PermissionRuleResolver,
        controller: PermissionController,
    ) -> None:
        self._resolver = resolver
        self._controller = controller

    def decide(self, operation: PermissionOperation) -> PermissionDecision:
        """
        判断当前操作应当直接允许、直接拒绝还是请求人工确认。

        Parameters:
            operation: 当前需要检查的具体权限操作。

        Returns:
            包含最终结果、稳定原因代码、说明消息和可选命中规则的ermissionDecision。
        """
        # 找到当前操作最终命中的权限规则
        rule = self._resolver.resolve(
            operation,
            session_rules=self._controller.session_rules(),
            local=self._controller.local_layer(),
        )
        # 命中权限规则
        if rule is not None:
            if rule.effect is PermissionEffect.ALLOW:
                return PermissionDecision(
                    PermissionOutcome.ALLOW,
                    "rule_allow",
                    f"权限规则允许：{rule.tool.value}({rule.pattern})",
                    rule,
                )
            return PermissionDecision(
                PermissionOutcome.DENY,
                "rule_deny",
                f"权限规则拒绝：{rule.tool.value}({rule.pattern})",
                rule,
            )

        # 未命中权限规则
        mode = self._controller.mode
        # 严格模式，直接拒绝未匹配规则的工具调用
        if mode is PermissionMode.STRICT:
            return PermissionDecision(
                PermissionOutcome.DENY,
                "strict_mode",
                "严格权限模式拒绝了未匹配规则的工具调用",
            )
        # 允许模式，直接放行未匹配规则的工具调用
        if mode is PermissionMode.ALLOW:
            return PermissionDecision(
                PermissionOutcome.ALLOW,
                "allow_mode",
                "放行权限模式允许了未匹配规则的工具调用",
            )

        # 来到这则是默认模式，则询问用户是否可以调用未匹配规则的工具
        return PermissionDecision(
            PermissionOutcome.ASK,
            "default_mode",
            "默认权限模式需要用户确认未匹配规则的工具调用",
        )
