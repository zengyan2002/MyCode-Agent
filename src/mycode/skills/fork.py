"""在独立临时对话中运行 fork Skill，并控制可复制的历史范围。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import uuid4

from mycode.agent.cancellation import CancellationToken
from mycode.agents.runtime import IndependentAgentRuntimeBuilder
from mycode.agents.tool_policy import build_child_tool_view
from mycode.errors import MyCodeError
from mycode.models.agents import (
    BackgroundTaskStatus,
    IndependentAgentOrigin,
    IndependentAgentSpec,
)
from mycode.models.events import AgentRunOptions
from mycode.models.messages import (
    AssistantMessage,
    ChatMessage,
    ToolResultMessage,
    UserMessage,
)
from mycode.models.prompts import PromptContext
from mycode.permissions.policy import PermissionController
from mycode.models.skills import SkillContextMode, SkillDefinition
from mycode.agents.workspaces import AgentWorkspaceService


class SkillForkError(MyCodeError):
    """表示独立 Skill 在拿到最终回答前已经失败。"""


class SkillForkRunner:
    """把 fork Skill 转成 IndependentAgentSpec 并复用子 Agent 运行核心。

    Attributes:
        _runtime_builder: 每次 fork Skill 调用使用的独立运行装配器。
        _parent_permissions: 创建调用时权限模式快照的父控制器。
        _session_id_getter: 返回本次调用所属主会话 ID 的函数。
        _stable_prompt: 独立请求继承的应用固定系统提示。
        _options: fork Skill 使用的模型调用上限、并发和超时设置。
    """

    def __init__(
        self,
        runtime_builder: IndependentAgentRuntimeBuilder,
        parent_permissions: PermissionController,
        session_id_getter: Callable[[], str],
        stable_prompt: str,
        workspace_service: AgentWorkspaceService,
        *,
        options: AgentRunOptions | None = None,
    ) -> None:
        """连接 fork Skill 构造冻结运行输入所需的应用对象。

        Args:
            runtime_builder: 为每次调用创建独立对话、权限、缓存和 Hook 的
                子 Agent 运行装配器。
            parent_permissions: 创建 spec 时读取当前父权限模式的控制器。
            session_id_getter: 每次调用时返回当前主会话 ID 的函数。
            stable_prompt: 每轮 fork 模型请求都要发送的固定系统提示词。
            workspace_service: 为 fork Skill 强制准备独立 Worktree 的服务。
            options: fork 的模型调用上限、并发和超时配置；不传时使用 Agent 默认值。

        Returns:
            不返回数据；真正的独立运行对象会在每次 ``run`` 调用时创建。
        """

        self._runtime_builder = runtime_builder
        self._parent_permissions = parent_permissions
        self._session_id_getter = session_id_getter
        self._stable_prompt = stable_prompt
        if not isinstance(workspace_service, AgentWorkspaceService):
            raise ValueError("workspace_service 类型无效")
        self._workspace_service = workspace_service
        self._options = options or AgentRunOptions()

    async def run(
        self,
        skill: SkillDefinition,
        arguments: str,
        main_history: tuple[ChatMessage, ...],
        cancellation: CancellationToken,
    ) -> str:
        """按 Skill 的历史范围执行一次独立 Agent 对话。

        Args:
            skill: 本次要执行的 fork Skill 当前定义。
            arguments: 用户在斜杠命令或 LoadSkill 中传入的原始参数。
            main_history: 调用发生前主会话的只读消息快照。
            cancellation: 主界面用于停止本次独立运行的取消令牌。

        Returns:
            独立 Agent 最终生成的 assistant 文本。调用方决定如何写回主会话。

        Raises:
            SkillForkError: 执行器返回错误，或结束时没有产生最终回答。
            MyCodeError: 主历史的工具链损坏，无法安全复制到 fork。
        """

        task_text = f"执行 Skill {skill.name}。"
        if arguments.strip():
            task_text += f"\n用户参数：{arguments}"
        spec = IndependentAgentSpec(
            run_id=uuid4().hex,
            session_id=self._session_id_getter(),
            name=skill.name,
            description=f"执行 fork Skill {skill.name}",
            origin=IndependentAgentOrigin.SKILL_FORK,
            task_prompt=task_text,
            initial_messages=select_fork_history(main_history, skill.context),
            prompt=PromptContext(self._stable_prompt),
            inherited_runtime=(),
            initial_tool_names=None,
            role=None,
            model_override=skill.model,
            max_model_calls=self._options.max_model_calls,
            permission_mode=self._parent_permissions.mode,
            background=False,
            tool_view=build_child_tool_view(
                origin=IndependentAgentOrigin.SKILL_FORK,
                parent_visible_names=None,
                background=False,
                role=None,
                additional_allowlist=skill.allowed_tools,
            ),
            skill=skill,
            skill_arguments=arguments,
        )
        spec = await self._workspace_service.prepare(spec)
        try:
            handle = self._runtime_builder.build(spec).start()
        except Exception as exc:
            await self._workspace_service.abandon(
                spec,
                f"fork Skill 装配失败：{exc}",
            )
            raise
        cancellation_waiter = asyncio.create_task(cancellation.wait())
        try:
            done, _ = await asyncio.wait(
                {handle.task, cancellation_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancellation_waiter in done and handle.task not in done:
                handle.cancel()
            result = await handle.wait()
            if result.status in {
                BackgroundTaskStatus.COMPLETED,
                BackgroundTaskStatus.PARTIAL,
            }:
                assert result.final_text is not None
                return result.final_text
            raise SkillForkError(result.error or f"Skill {skill.name} 运行失败")
        except asyncio.CancelledError:
            handle.cancel()
            await asyncio.gather(handle.task, return_exceptions=True)
            raise
        finally:
            cancellation_waiter.cancel()
            await asyncio.gather(cancellation_waiter, return_exceptions=True)


def _complete_history(
    history: tuple[ChatMessage, ...],
) -> tuple[ChatMessage, ...]:
    """删除历史末尾未完成的工具调用链。

    Args:
        history: 主会话当前可见的消息。正常情况下工具调用和结果已配对。

    Returns:
        从开头到最后一个完整普通消息或完整工具结果组的独立元组。

    Raises:
        MyCodeError: 历史中间出现孤立或工具名不匹配的结果，说明会话已损坏。
    """

    complete: list[ChatMessage] = []
    index = 0
    while index < len(history):
        message = history[index]
        if isinstance(message, ToolResultMessage):
            raise MyCodeError("主会话包含没有对应调用的工具结果")
        if not isinstance(message, AssistantMessage) or not message.tool_calls:
            complete.append(message)
            index += 1
            continue
        expected = tuple(
            (call.id, call.name) for call in message.tool_calls
        )
        results: list[ToolResultMessage] = []
        cursor = index + 1
        while cursor < len(history) and isinstance(
            history[cursor],
            ToolResultMessage,
        ):
            results.append(history[cursor])
            cursor += 1
        actual = tuple(
            (result.tool_call_id, result.tool_name)
            for result in results
        )
        if actual != expected:
            # 只有尾部半轮可以安静丢弃；中间损坏不能猜测应保留哪些消息。
            if cursor == len(history):
                break
            raise MyCodeError("主会话中的工具调用与结果不完整匹配")
        complete.extend((message, *results))
        index = cursor
    return tuple(complete)


def select_fork_history(
    history: tuple[ChatMessage, ...],
    mode: SkillContextMode,
    *,
    recent_turns: int = 5,
) -> tuple[ChatMessage, ...]:
    """按 fork Skill 的 context 设置复制主会话历史。

    Args:
        history: 主会话当前可见消息，不会被本函数修改。
        mode: none、recent 或 full 历史范围。
        recent_turns: recent 模式最多保留的完整用户轮次数，默认 5。

    Returns:
        独立消息元组。none 为空；full 为全部完整历史；recent 从倒数第五
        个用户消息开始，并保留其后的完整工具调用链和助手回答。

    Raises:
        ValueError: recent_turns 不是正整数。
        MyCodeError: 历史中间存在损坏的工具调用链。
    """

    if recent_turns <= 0:
        raise ValueError("fork 近期轮数必须为正数")
    if mode is SkillContextMode.NONE:
        return ()
    complete = _complete_history(history)
    if mode is SkillContextMode.FULL:
        return tuple(complete)
    user_indexes = [
        index
        for index, message in enumerate(complete)
        if isinstance(message, UserMessage)
    ]
    if len(user_indexes) <= recent_turns:
        return tuple(complete)
    return tuple(complete[user_indexes[-recent_turns] :])
