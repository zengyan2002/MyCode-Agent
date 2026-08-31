"""把冻结命令注册表适配为 prompt-toolkit 的补全器。"""

from __future__ import annotations

from collections.abc import Iterable

from prompt_toolkit.completion import Completer, Completion, CompleteEvent
from prompt_toolkit.document import Document

from mycode.commands.registry import CommandRegistry


class CommandCompleter(Completer):
    """只补全输入框开头尚未带参数的斜杠命令。"""

    def __init__(self, registry: CommandRegistry) -> None:
        """保存补全时查询的冻结命令注册表。

        Args:
            registry: 已完成内置命令登记并冻结的注册表。

        Returns:
            None。
        """

        # 所有候选都从同一个注册表读取，避免帮助与补全清单不一致
        self._registry = registry

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> Iterable[Completion]:
        """返回光标前第一个斜杠命令词的候选。

        Args:
            document: 输入框当前文字、光标位置和选区信息。
            complete_event: 本次补全由按键触发还是后台自动触发的信息。

        Returns:
            可由 prompt-toolkit 替换到输入框中的命令候选迭代器。
        """

        del complete_event
        before_cursor = document.text_before_cursor
        if not before_cursor.startswith("/"):
            return
        if any(character.isspace() for character in before_cursor):
            return
        if document.cursor_position != len(document.text):
            return
        prefix = before_cursor[1:]
        for candidate in self._registry.complete(prefix):
            yield Completion(
                candidate,
                start_position=-len(before_cursor),
            )
