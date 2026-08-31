"""在外部 Skill 的专属工具首次运行前取得用户信任。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mycode.models.tools import (
    ToolErrorCode,
    ToolExecutionPolicy,
    ToolInvocation,
    ToolSource,
)
from mycode.tools.interceptors import InterceptionDecision, ToolRunContext
from mycode.tools.registry import ToolRegistry


class SkillTrustApprover(Protocol):
    """由当前界面显示 Skill 来源信息并返回用户是否信任。"""

    async def confirm(self, message: str) -> bool:
        """显示默认拒绝的确认问题。

        Args:
            message: 包含 Skill 来源、工具名、命令和读写分类的说明。

        Returns:
            用户明确确认时返回 True；取消或拒绝时返回 False。
        """

        ...


@dataclass(frozen=True)
class SkillTrustIdentity:
    """表示本会话已经确认过的一个外部 Skill 工具版本。"""

    # 提供工具的 Skill 名，比较时使用规范化小写形式。
    skill_name: str
    # 注册给模型的全局工具名。
    tool_name: str
    # project 或 user；内置 Skill 不会创建此对象。
    skill_origin: str
    # 用户确认时看到的 Skill 入口真实路径。
    source_path: Path
    # 本次实际会启动的命令数组；reload 改变命令后会形成新身份。
    command: tuple[str, ...]
    # 工具可信定义中的 read 或 write 分类。
    access: str


class SkillTrustStore:
    """保存当前会话已信任的外部 Skill 工具身份。"""

    def __init__(self) -> None:
        """创建空的会话信任集合。"""

        # 只存在当前进程内存中；/clear 和新会话会调用 clear。
        self._trusted: set[SkillTrustIdentity] = set()

    @property
    def identities(self) -> frozenset[SkillTrustIdentity]:
        """返回当前会话全部已信任身份的只读快照。

        Returns:
            调用方不能修改的 SkillTrustIdentity 集合。
        """

        return frozenset(self._trusted)

    def contains(self, identity: SkillTrustIdentity) -> bool:
        """判断当前会话是否已经确认过完全相同的工具版本。

        Args:
            identity: 根据当前注册策略和工具读写分类生成的身份。

        Returns:
            之前确认过相同身份时返回 True，否则返回 False。
        """

        return identity in self._trusted

    def allow(self, identity: SkillTrustIdentity) -> None:
        """把用户刚确认的工具身份加入本会话信任集合。

        Args:
            identity: 已经向用户完整展示并获得确认的工具身份。

        Returns:
            None。
        """

        self._trusted.add(identity)

    def clear(self) -> None:
        """清除本会话保存的全部 Skill 工具信任。

        Returns:
            None。
        """

        self._trusted.clear()


class SkillTrustInterceptor:
    """拦截项目级和用户级 Skill 专属工具的首次调用。"""

    def __init__(
        self,
        registry: ToolRegistry,
        approver: SkillTrustApprover,
        store: SkillTrustStore,
    ) -> None:
        """连接工具注册表、当前界面和会话信任集合。

        Args:
            registry: 提供工具来源及真实执行策略的注册表。
            approver: 显示确认信息并等待用户选择的当前界面。
            store: 只在当前会话保存确认结果的信任集合。

        Returns:
            None。
        """

        # 用来确认调用确实来自 Skill，并取得未暴露给模型的可信策略。
        self._registry = registry
        # 只有外部 Skill 首次调用会触发界面确认。
        self._approver = approver
        # 同一个工具版本再次调用时用此集合避免重复询问。
        self._store = store

    async def before_tool(
        self,
        context: ToolRunContext,
    ) -> InterceptionDecision:
        """在外部 Skill 专属工具首次执行前显示来源并询问用户。

        Args:
            context: 已包含可信读写分类的调用和当前运行选项。Plan 拦截器
                已在本拦截器之前处理计划模式。

        Returns:
            内置或已信任工具直接放行；用户确认后记录并放行；拒绝、策略
            缺失或策略不完整时返回阻止决定。
        """

        invocation = context.invocation
        policy = self._registry.execution_policy(invocation.call.name)
        if policy is None or policy.source is not ToolSource.SKILL:
            return InterceptionDecision.allow()
        if policy.skill_origin == "builtin":
            return InterceptionDecision.allow()
        identity = self._identity(invocation, policy)
        if identity is None:
            return InterceptionDecision.deny(
                ToolErrorCode.BLOCKED,
                f"Skill 工具 {invocation.call.name} 缺少首次信任信息",
            )
        if self._store.contains(identity):
            return InterceptionDecision.allow()
        command = json.dumps(
            list(identity.command),
            ensure_ascii=False,
        )
        message = (
            f"外部 Skill {identity.skill_name} 即将首次运行专属工具。\n"
            f"来源：{identity.source_path}\n"
            f"工具：{identity.tool_name}\n"
            f"分类：{identity.access}\n"
            f"命令：{command}\n"
            "是否信任这个工具在本会话中运行？"
        )
        if not await self._approver.confirm(message):
            return InterceptionDecision.deny(
                ToolErrorCode.BLOCKED,
                "用户拒绝信任这个外部 Skill 工具",
            )
        self._store.allow(identity)
        return InterceptionDecision.allow()

    @staticmethod
    def _identity(
        invocation: ToolInvocation,
        policy: ToolExecutionPolicy,
    ) -> SkillTrustIdentity | None:
        """从可信注册策略创建用于会话缓存的工具身份。

        Args:
            invocation: 提供工具名和本地注册表给出的读写分类。
            policy: 提供所属 Skill、来源、入口路径和实际命令。

        Returns:
            必要字段齐全时返回 SkillTrustIdentity，否则返回 None。
        """

        if (
            not policy.skill_name
            or policy.skill_origin not in {"project", "user"}
            or policy.source_path is None
            or not policy.command
        ):
            return None
        return SkillTrustIdentity(
            skill_name=policy.skill_name,
            tool_name=invocation.call.name,
            skill_origin=policy.skill_origin,
            source_path=policy.source_path,
            command=policy.command,
            access=invocation.access.value,
        )
