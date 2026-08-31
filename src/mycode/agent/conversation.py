"""保存当前进程正在使用的内存消息，不写文件"""

from __future__ import annotations

from collections.abc import Sequence

from mycode.models.messages import ChatMessage


class Conversation:
    """保存当前进程中已经确认并可发送给 Provider 的真实消息历史。"""

    def __init__(self) -> None:
        # 历史只存在当前进程内，不负责持久化、裁剪或 Provider 格式转换。
        # 何时提交由 AgentLoop 决定，这个容器只提供最小的追加/清空语义。
        self._history: list[ChatMessage] = []

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        # 返回不可变快照，防止 Provider 或调用方通过共享 list 修改已提交
        # 历史；内部仍使用 list，以便按完整工具轮高效追加多个消息。
        return tuple(self._history)

    def extend(self, messages: Sequence[ChatMessage]) -> None:
        """按顺序追加一批已经完成的用户、助手或工具消息。"""

        self._history.extend(messages)

    def replace(self, messages: Sequence[ChatMessage]) -> None:
        """把当前对话历史整体换成一份复制好的新历史，主要用于恢复旧会话，执行 /session resume <session_id>

        ContextManager 只在摘要验证和相关 artifact 提交完成后调用这里。
        先构造新列表再替换引用，调用方不会观察到半份新历史。
        """

        replacement = list(messages)
        self._history = replacement

    def clear(self) -> None:
        """删除当前进程内的全部对话消息。"""

        self._history.clear()
