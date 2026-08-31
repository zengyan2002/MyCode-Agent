"""协议无关的服务器发送事件（SSE）解析器。"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class SSEEvent:
    """单个协议中立的服务器发送事件。"""

    event: str | None
    data: str


async def iter_sse(lines: AsyncIterable[str]) -> AsyncIterator[SSEEvent]:
    # lines 是可通过 async for 逐行消费的异步文本流。
    # data_lines 暂存当前 SSE 事件的所有 data 字段。
    data_lines: list[str] = []

    # event_name 保存可选事件名称；缺省时由上层按协议解释。
    event_name: str | None = None

    # 把当前缓存组装成 SSEEvent，并清空状态以解析下一条事件。
    def dispatch() -> SSEEvent | None:
        nonlocal data_lines, event_name
        if not data_lines:
            event_name = None
            return None
        result = SSEEvent(event=event_name, data="\n".join(data_lines))
        data_lines = []
        event_name = None
        return result

    async for raw_line in lines:
        # 上游已按 \n 分行，这里只去掉可能残留的 \r。
        line = raw_line[:-1] if raw_line.endswith("\r") else raw_line
        if line == "":
            # 空行表示当前事件结束。
            event = dispatch()
            if event:
                yield event
            continue

        # 遇到 SSE 注释或心跳行时跳过，不把它当成事件数据处理。
        if line.startswith(":"):
            continue

        # 将一行 SSE 文本拆成“字段名”和“字段值”。
        if ":" in line:
            # 有冒号时，冒号前是字段名，后面是字段值。
            field, value = line.split(":", 1)
            if value.startswith(" "):
                value = value[1:]
        else:
            # 没有冒号时，整行是字段名，字段值为空。
            field, value = line, ""

        if field == "data":
            # data 字段可以重复出现，最终按换行拼接。
            data_lines.append(value)
        elif field == "event":
            # event 字段用于声明事件名称。
            event_name = value

    # 流结束前可能没有尾随空行，仍需交付最后一条已缓存事件。
    event = dispatch()
    if event:
        yield event
