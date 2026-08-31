"""把用户输入拆成普通文本或一条未解释参数的斜杠命令。"""

from __future__ import annotations

from mycode.commands.models import ParsedCommand


def parse_command(text: str) -> ParsedCommand | None:
    """识别斜杠命令，并拆出命令名和剩余参数。

    Args:
        text: 用户按回车时提交的原始输入。

    Returns:
        斜杠输入对应的 ``ParsedCommand``；普通文本返回 ``None``。
    """

    cleaned = text.strip()
    if not cleaned.startswith("/"):
        return None
    body = cleaned[1:]
    if not body:
        return ParsedCommand(cleaned, "", "")
    parts = body.split(maxsplit=1)
    name = parts[0].casefold()
    args = parts[1].strip() if len(parts) == 2 else ""
    return ParsedCommand(cleaned, name, args)
