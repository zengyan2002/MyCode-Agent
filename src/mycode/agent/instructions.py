"""管理环境、一次性通知和 Plan 提醒的可信运行时状态。"""

from __future__ import annotations

from mycode.agent.environment import (
    EnvironmentCollector,
    EnvironmentSnapshot,
    compare_environment,
)
from mycode.agent.system_prompt import PLAN_COMPACT_REMINDER, PLAN_FULL_INSTRUCTION
from mycode.models.prompts import RuntimeInstruction, RuntimeInstructionKind


def deferred_tools_instruction(
    names: tuple[str, ...],
) -> RuntimeInstruction | None:
    """把尚未激活的 MCP 工具名称生成为本次请求使用的提醒。

    Args:
        names: 当前 ToolActivationState 尚未激活的 MCP 工具名。

    Returns:
        有名称时返回提示模型先调用 tool_search 的指令；为空时返回 ``None``。
    """
    if not names:
        return None
    content = "\n".join(
        (
            "以下 MCP 工具已注册，但完整定义尚未加入工具列表。",
            "需要使用时，先调用 tool_search，按名称或用途关键词搜索并激活：",
            *(f"- {name}" for name in names),
        )
    )
    return RuntimeInstruction(
        RuntimeInstructionKind.RUNTIME_NOTICE,
        content,
    )


class RuntimeInstructionManager:
    """维护环境、目录、活动 SOP 和 Plan 提醒，生成每次请求的运行时指令。

    Attributes:
        _environment: 读取当前工作区、日期和平台信息的收集器。
        _environment_history: 环境发生变化后需要持续保留的指令。
        _notices: 只在下一次真实请求中发送一次的通知。
        _skill_catalog: 当前可用 Skill 的名称和用途目录。
        _agent_catalog: 当前可委派角色的名称和用途目录。
        _active_skills: 当前会话已激活的完整 SOP，值中保存激活顺序。
        _plan_requests: 当前连续 Plan 模式已发送的模型请求次数。
    """

    def __init__(self, environment: EnvironmentCollector) -> None:
        """保存环境收集器并创建空的目录、通知和活动 Skill 状态。

        Args:
            environment: 每次请求前读取工作区环境快照的收集器。

        Returns:
            不返回数据。
        """

        # 环境收集器
        self._environment = environment
        # 上一次的环境快照
        self._last_environment: EnvironmentSnapshot | None = None
        # 保存已经产生过的环境指令历史
        self._environment_history: list[RuntimeInstruction] = []
        # 保存还没有发送给模型的运行时通知
        self._notices: list[RuntimeInstruction] = []
        # 记录Plan_on模式是否开启
        self._plan_active = False
        # 当前可用 Skill 的轻量目录；没有 Skill 时为 None。
        self._skill_catalog: RuntimeInstruction | None = None
        # 当前可委派 Agent 的轻量目录；定义式子 Agent 不设置该字段。
        self._agent_catalog: RuntimeInstruction | None = None
        # 键是规范化 Skill 名，值保存激活顺序和完整 SOP 指令。
        self._active_skills: dict[
            str,
            tuple[int, RuntimeInstruction],
        ] = {}

        # 统计的是连续 Plan 模式下的模型请求次数，而不是用户发消息的次数。
        # 第 1 次：完整 Plan 指令
        # 第 2 次：简短提醒
        # 第 3 次：简短提醒
        # 第 4 次：简短提醒
        # 第 5 次：简短提醒
        # 第 6 次：再次发送完整指令
        self._plan_requests = 0

    def set_skill_catalog(self, content: str | None) -> None:
        """替换每轮请求都能看到的轻量 Skill 目录。

        Args:
            content: 只含 Skill 名和一句说明的文本；None 或空白表示当前
                没有可用 Skill，不生成目录指令。

        Returns:
            None。下一次 preview 和 prepare 会立即使用新目录。
        """

        cleaned = content.strip() if content is not None else ""
        self._skill_catalog = (
            RuntimeInstruction(
                RuntimeInstructionKind.SKILL_CATALOG,
                cleaned,
            )
            if cleaned
            else None
        )

    def set_active_skill(
        self,
        name: str,
        content: str,
        activated_order: int,
    ) -> None:
        """添加或刷新一个会在每轮请求中置顶的完整 Skill SOP。

        Args:
            name: Catalog 中的 Skill 名，比较时忽略大小写。
            content: 已经替换好 $ARGUMENTS 的完整 SOP 正文。
            activated_order: 当前会话分配的正整数激活顺序。

        Returns:
            None。重复名字会替换正文，但保留调用方传入的顺序。

        Raises:
            ValueError: 名字或正文为空，或者激活顺序不是正整数。
        """

        normalized = name.strip().casefold()
        if not normalized:
            raise ValueError("活动 Skill 名不能为空")
        if not content.strip():
            raise ValueError("活动 Skill SOP 不能为空")
        if activated_order <= 0:
            raise ValueError("Skill 激活顺序必须为正数")
        self._active_skills[normalized] = (
            activated_order,
            RuntimeInstruction(
                RuntimeInstructionKind.ACTIVE_SKILL,
                f"Skill: {name}\n\n{content}",
            ),
        )

    def set_agent_catalog(self, content: str | None) -> None:
        """替换主 Agent 每轮请求可见的轻量角色目录。

        Args:
            content: 只含角色名和用途说明的文本；``None`` 或空白会移除
                目录。角色完整系统提示不会通过该入口进入主上下文。

        Returns:
            不返回数据；下一次 preview 和 prepare 立即使用新目录。
        """

        cleaned = content.strip() if content is not None else ""
        self._agent_catalog = (
            RuntimeInstruction(
                RuntimeInstructionKind.AGENT_CATALOG,
                cleaned,
            )
            if cleaned
            else None
        )

    def remove_active_skill(self, name: str) -> bool:
        """移除一个活动 Skill 的完整 SOP 指令。

        Args:
            name: 需要停用的 Skill 名，比较时忽略大小写。

        Returns:
            原来存在活动指令时返回 True，否则返回 False。
        """

        return self._active_skills.pop(name.strip().casefold(), None) is not None

    def clear_active_skills(self) -> None:
        """移除当前会话全部活动 Skill SOP。

        Returns:
            None。
        """

        self._active_skills.clear()

    def _skill_instructions(self) -> tuple[RuntimeInstruction, ...]:
        """按目录在前、活动顺序在后生成 Skill 指令。

        Returns:
            当前目录指令和按 activated_order 排列的活动 SOP。后激活的
            Skill 位于元组更后方，更靠近本轮请求。
        """

        catalog = (
            (self._skill_catalog,)
            if self._skill_catalog is not None
            else ()
        )
        active = tuple(
            instruction
            for _, instruction in sorted(
                self._active_skills.values(),
                key=lambda item: item[0],
            )
        )
        return (*catalog, *active)

    def enqueue_notice(self, content: str) -> None:
        """
        把一条临时通知加入等待队列
        """
        self._notices.append(
            RuntimeInstruction(RuntimeInstructionKind.RUNTIME_NOTICE, content)
        )

    def enqueue_hook_notification(self, content: str) -> None:
        """把 Hook 生成的提醒加入当前 Agent 的下一次模型请求。

        Args:
            content: 已经由 Hook 模板展开的实际提示文本。

        Returns:
            None。内容在下一次 `prepare` 中以 `hook_notification` 标签返回，
            随后从队列移除，不会写入 Conversation 历史。
        """

        self._notices.append(
            RuntimeInstruction(
                RuntimeInstructionKind.HOOK_NOTIFICATION,
                content,
            )
        )

    def preview(self, *, plan_only: bool) -> tuple[RuntimeInstruction, ...]:
        """预览下一次请求会使用的运行时指令，但不消费任何状态。

        ContextManager 在请求前估算 Token 时调用这里。环境快照只用于计算
        可能新增的环境指令；一次性通知仍留在队列中，Plan 请求计数也不会
        提前增加。若中间没有外部状态变化，随后 ``prepare`` 返回相同内容。
        """

        current = self._environment.collect()
        environment_history = list(self._environment_history)
        if self._last_environment is None:
            environment_history.append(
                RuntimeInstruction(
                    RuntimeInstructionKind.ENVIRONMENT_CONTEXT,
                    current.render(),
                )
            )
        else:
            change = compare_environment(self._last_environment, current)
            if change.changed:
                environment_history.append(
                    RuntimeInstruction(
                        RuntimeInstructionKind.ENVIRONMENT_UPDATE,
                        change.render(),
                    )
                )

        mode: tuple[RuntimeInstruction, ...] = ()
        if plan_only:
            next_count = self._plan_requests + 1 if self._plan_active else 1
            full = (next_count - 1) % 5 == 0
            mode = (
                RuntimeInstruction(
                    RuntimeInstructionKind.MODE_INSTRUCTION
                    if full
                    else RuntimeInstructionKind.MODE_REMINDER,
                    PLAN_FULL_INSTRUCTION if full else PLAN_COMPACT_REMINDER,
                ),
            )
        return (
            *environment_history,
            *((self._agent_catalog,) if self._agent_catalog is not None else ()),
            *self._skill_instructions(),
            *tuple(self._notices),
            *mode,
        )

    def prepare(self, *, plan_only: bool) -> tuple[RuntimeInstruction, ...]:
        """
        准备本轮运行时指令
        """
        # 获取环境快照
        current = self._environment.collect()

        # 判断上一次环境快照是否为空
        if self._last_environment is None:
            #上一次环境快照为空，则直接向历史中追加
            self._environment_history.append(
                RuntimeInstruction(
                    RuntimeInstructionKind.ENVIRONMENT_CONTEXT,
                    current.render(),
                )
            )
        else:
            # 上一次环境快照非空
            change = compare_environment(self._last_environment, current)
            if change.changed:
                # 当前的环境快照相较于上一次发生了改变，就向历史环境快照追加
                self._environment_history.append(
                    RuntimeInstruction(
                        RuntimeInstructionKind.ENVIRONMENT_UPDATE,
                        change.render(),
                    )
                )
        # 无论环境有没有变化，都把本次快照记为“上一次环境”。
        self._last_environment = current

        #将列表转换为元组，不用列表赋值是为了防止清理self._notices的同时清理掉notices
        notices = tuple(self._notices)
        self._notices.clear()

        # 初始化模式指令  Plan模式会加提示词告诉LLM现在是Plan模式，普通模式时为空
        mode: tuple[RuntimeInstruction, ...] = ()

        #判断当前请求是否开启了plan模式
        if plan_only:
            if not self._plan_active:
                #第一次进入Plan模式，重置计数
                self._plan_requests = 0
            # 标志Plan模式已经激活
            self._plan_active = True
            # 这里统计的是模型请求次数，不是用户输入次数。一次用户请求可能因为工具循环产生多次模型请求。
            self._plan_requests += 1
            # 判断是否发送完整规则 第1次、第6次、第11次、...提示词内发送完整的Plan指令
            full = (self._plan_requests - 1) % 5 == 0
            mode = (
                RuntimeInstruction(
                    RuntimeInstructionKind.MODE_INSTRUCTION
                    if full
                    else RuntimeInstructionKind.MODE_REMINDER,
                    PLAN_FULL_INSTRUCTION if full else PLAN_COMPACT_REMINDER,
                ),
            )
        else:
            self._plan_active = False
            self._plan_requests = 0

        return (
            *self._environment_history,
            *((self._agent_catalog,) if self._agent_catalog is not None else ()),
            *self._skill_instructions(),
            *notices,
            *mode,
        )

    def reset(self) -> None:
        """
        重置管理器状态
        """
        self._last_environment = None
        self._environment_history.clear()
        self._notices.clear()
        self._plan_active = False
        self._plan_requests = 0
