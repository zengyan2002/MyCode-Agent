"""把对话历史划分成不可拆的消息组，并选出近期原文。"""

from __future__ import annotations

from dataclasses import dataclass

from mycode.constants import RECENT_MESSAGE_GROUPS, RECENT_MESSAGE_TOKENS
from mycode.context.estimator import estimate_message
from mycode.errors import MyCodeError
from mycode.models.messages import (
    AssistantMessage,
    ChatMessage,
    ToolResultMessage,
    UserMessage,
)


@dataclass(frozen=True)
class MessageGroup:
    """保存一组压缩时不能拆开的对话消息

    普通用户消息和不含工具调用的助手消息各自单独成组。
    如果助手调用了工具，该助手消息和随后返回的所有工具结果归为一组，
    避免压缩后只留下工具调用或只留下工具结果。
    """

    # 按原历史顺序保存且不能在压缩边界拆开的消息
    messages: tuple[ChatMessage, ...]
    # 标记该组是否包含用户原话；缩减摘要内容时不会删除这类消息
    contains_user_message: bool
    # 组内消息的 Token 估算总数，用来决定近期保留多少原文
    estimated_tokens: int


@dataclass(frozen=True)
class HistoryPartition:
    """录对话历史拆分后的两部分：较早消息用于生成摘要，近期消息保留原文"""

    # 较早的消息组，本次生成摘要时会发送给模型
    compactable_groups: tuple[MessageGroup, ...]
    # 最近的消息组，摘要成功后仍以原文留在对话记录中
    recent_groups: tuple[MessageGroup, ...]


class HistoryPartitioner:
    """检查工具调用和结果是否配对，再从对话末尾保留近期消息，并且不会拆开同一组消息"""

    def group(
        self,
        history: tuple[ChatMessage, ...],
    ) -> tuple[MessageGroup, ...]:
        """把对话历史整理成压缩时不能拆开的消息组。

        Args:
            history: 按时间顺序排列的全部对话消息。

        Returns:
            按原顺序排列的消息组。普通消息单独成组；助手的工具调用和对应的工具结果放在同一组。
        """
        # 用来保存已经整理好的消息组
        groups: list[MessageGroup] = []
        # 记录当前处理到 history 中的第几条消息
        index = 0
        while index < len(history):
            message = history[index]
            if isinstance(message, ToolResultMessage):
                raise MyCodeError("对话历史中存在没有对应助手调用的工具结果")
            if not isinstance(message, AssistantMessage) or not message.tool_calls:
                # 不涉及工具调用的普通消息，让它自己单独成为一组，，可以是用户消息也可以是不带工具调用的助手消息
                messages = (message,)
                groups.append(
                    MessageGroup(
                        messages,
                        isinstance(message, UserMessage),
                        sum(estimate_message(item) for item in messages),
                    )
                )
                index += 1
                continue
            # 走到这里说明是带有工具调用的助手消息
            # 取出本次消息调用工具的id
            expected = [call.id for call in message.tool_calls]
            # 保存当前这条助手消息调用工具后，紧接着返回的所有工具结果
            results: list[ToolResultMessage] = []
            # 找到这条助手消息调用的所有工具结果
            index += 1
            while index < len(history) and isinstance(
                history[index], ToolResultMessage
            ):
                results.append(history[index])
                index += 1
            # 取出实际工具结果对应的工具id
            actual = [result.tool_call_id for result in results]

            if actual != expected:
                raise MyCodeError("对话历史中的工具调用与结果不完整匹配")

            # 将调用工具请求的助手消息和工具调用结果放在一组
            messages = (message, *results)
            groups.append(
                MessageGroup(
                    messages,
                    False,
                    sum(estimate_message(item) for item in messages),
                )
            )
        return tuple(groups)

    def partition(
        self,
        history: tuple[ChatMessage, ...],
        *,
        minimum_recent_groups: int | None = None,
        recent_token_target: int | None = None,
    ) -> HistoryPartition:
        """按给定近期目标划分较早消息和要原样保留的尾部消息。"""

        if minimum_recent_groups is None:
            minimum_recent_groups = RECENT_MESSAGE_GROUPS
        if recent_token_target is None:
            recent_token_target = RECENT_MESSAGE_TOKENS
        if minimum_recent_groups < 1:
            raise ValueError("近期消息组下限必须至少为 1")
        if recent_token_target < 0:
            raise ValueError("近期消息 Token 目标不能为负数")

        groups = self.group(history)
        if not groups:
            return HistoryPartition((), ())
        start = max(0, len(groups) - minimum_recent_groups)
        # 后RECENT_MESSAGE_GROUPS组的token花费总数
        token_total = sum(group.estimated_tokens for group in groups[start:])
        # 组数满足了RECENT_MESSAGE_GROUPS，但是token花费总数不满足RECENT_MESSAGE_TOKENS，则向前继续找作为最近消息组来保存
        while start > 0 and token_total < recent_token_target:
            start -= 1
            token_total += groups[start].estimated_tokens
        return HistoryPartition(groups[:start], groups[start:])


def flatten_groups(groups: tuple[MessageGroup, ...]) -> tuple[ChatMessage, ...]:
    """把多个消息组重新合并成一条连续的对话记录

    Args:
        groups: 按对话顺序排列的消息组。

    Returns:
        从各组中依次取出的全部消息，顺序与原对话一致
    """

    return tuple(message for group in groups for message in group.messages)
