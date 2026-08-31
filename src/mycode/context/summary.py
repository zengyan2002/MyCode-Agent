"""构造无工具摘要请求，并验证模型返回的结构化摘要。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from mycode.context.history import MessageGroup
from mycode.errors import MyCodeError
from mycode.models.messages import (
    ChatMessage,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from mycode.models.prompts import PromptContext
from mycode.models.provider import (
    ModelStopReason,
    ProviderCompleted,
    ProviderRequest,
    ToolChoice,
)

SUMMARY_HEADINGS = (
    "## 1. 主要请求和意图",
    "## 2. 关键技术概念",
    "## 3. 文件和代码段",
    "## 4. 错误和修复",
    "## 5. 问题解决过程",
    "## 6. 所有用户消息",
    "## 7. 待办任务",
    "## 8. 当前工作",
    "## 9. 可能的下一步",
)

SUMMARY_SYSTEM_PROMPT = """你负责压缩给定的对话材料，只能处理输入文本。
禁止调用、建议调用或模拟任何工具；本请求也不会提供工具定义。

输出必须严格采用以下结构，标签外不能出现任何内容：
<analysis>
先梳理事实、决定、文件、错误、当前进度和遗漏风险。这里只写草稿。
</analysis>
<summary>
按给定的九个标题和固定顺序写正式摘要。
</summary>

正式摘要必须遵守：
- 九个标题各出现一次，不得改名、合并或调换顺序。
- 第 8 部分“当前工作”最详细，要能让后续模型从中继续执行。
- 第 6 部分只复制材料提供的用户原话，不得摘要、润色或改写。
- 不把建议、猜测或未批准事项写成用户决定。
- 不补写材料中不存在的代码、文件内容或执行结果。

固定标题：
""" + "\n".join(SUMMARY_HEADINGS)

# 检查模型响应是否只有 analysis 和 summary 两部分；分别取出分析草稿和正式摘要
_RESPONSE = re.compile(
    r"^\s*<analysis>(?P<analysis>.*?)</analysis>\s*"
    r"<summary>(?P<summary>.*?)</summary>\s*$",
    re.DOTALL,
)


@dataclass(frozen=True)
class CompactionMaterial:
    """保存本次生成对话摘要时要发给模型的材料

    这里包含上一次摘要、本次要摘要的较早消息，以及要求模型原样复制的
    用户消息。近期继续保留原文的消息不放在这里。用户消息过多时，还会
    记录省略数量和完整原话文件的路径
    """

    # 上一次生成的摘要；第一次生成摘要时没有这个值
    previous_summary: str | None
    # 本次需要交给模型整理的较早对话
    groups: tuple[MessageGroup, ...]
    # 要求模型在摘要第 6 部分原样复制的用户消息
    user_messages: tuple[UserMessage, ...]
    # 因为长度限制，没有直接放进摘要请求的用户消息条数
    omitted_user_messages: int = 0
    # 有用户消息被省略时，保存全部用户原话的文件路径
    user_transcript_path: str | None = None
    # 用户手动压缩时要求摘要额外保留的内容；普通压缩没有这个值
    retention_focus: str | None = None


def _message_payload(message: ChatMessage) -> dict[str, object]:
    """把一条对话消息整理成摘要请求中使用的字典

    Args:
        message: 一条用户消息、助手消息或工具执行结果。

    Returns:
        包含消息角色和正文的字典，可以继续转换成 JSON 发给摘要模型。
    """

    # 用户消息
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}
    # 工具调用结果
    if isinstance(message, ToolResultMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "is_error": message.is_error,
            "content": message.content,
        }
    # 走到这里，就说明是助手消息了
    blocks: list[dict[str, object]] = []
    # 遍历助手消息里的内容块，把它们逐个转换成普通字典，方便后面转成 JSON 发给摘要模型。
    for block in message.content:
        if isinstance(block, TextBlock):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolCall):
            blocks.append(
                {
                    "type": "tool_call",
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.arguments,
                }
            )
        elif isinstance(block, ThinkingBlock):
            blocks.append({"type": "thinking", "content": block.thinking})
        elif isinstance(block, RedactedThinkingBlock):
            blocks.append({"type": "redacted_thinking"})
    return {"role": "assistant", "content": blocks}


class SummaryCodec:
    """准备发给模型的摘要请求，并检查模型返回的摘要格式

    它会把较早对话和用户原话整理成请求内容。模型返回后，检查九个标题是否完整且顺序正确，最后只取出正式摘要
    """

    def build_request(
            self,
            material: CompactionMaterial,
            *,
            max_output_tokens: int,
    ) -> ProviderRequest:
        """把摘要材料整理成文字，并生成发给模型的摘要请求。

        Args:
            material: 本次摘要使用的上一次摘要、较早对话和用户原话。
            max_output_tokens: 本次摘要最多允许模型输出的 Token 数量。

        Returns:
            不提供任何工具、可以直接发送给模型的摘要请求。
        """

        history = [
            _message_payload(message)
            for group in material.groups
            for message in group.messages
        ]
        user_lines = [
            f"{index}. {message.content}"
            for index, message in enumerate(material.user_messages, start=1)
        ]
        if material.omitted_user_messages:
            user_lines.append(
                f"[另有 {material.omitted_user_messages} 条较早用户原话未放入"
                f"本次请求；完整记录：{material.user_transcript_path}]"
            )

        sections = [
            "【已有摘要】\n" + (material.previous_summary or "无"),
            "【本次需要摘要的较早对话】\n"
            + json.dumps(
                history,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "【第 6 部分必须原样复制的用户消息】\n"
            + ("\n".join(user_lines) or "无"),
        ]
        if material.retention_focus is not None:
            sections.append(
                "【用户要求额外保留的重点】\n" + material.retention_focus
            )
        content = "\n\n".join(sections)

        return ProviderRequest(
            messages=(UserMessage(content),),
            tools=(),
            tool_choice=ToolChoice.NONE,
            prompt=PromptContext(stable=SUMMARY_SYSTEM_PROMPT),
            max_output_tokens=max_output_tokens,
        )

    def parse(self, completed: ProviderCompleted) -> str:
        """检查模型返回的摘要格式，并取出正式摘要正文。

        只有响应正常结束、analysis 和 summary 标签完整，并且九个标题各出现一次且
        顺序正确时，才会返回 summary 标签里的内容。

        Args:
            completed: 模型完成摘要请求后返回的完整响应。

        Returns:
            去掉前后空白的正式摘要正文，不包含 analysis 草稿和标签。
        """

        if completed.stop_reason is not ModelStopReason.END_TURN:
            raise MyCodeError("摘要响应未正常结束")

        """
        用正则检查模型返回的整段文字格式，并为后面提取正式摘要做好准备
        <analysis>草稿</analysis>
        <summary>正式摘要</summary>
        """
        match = _RESPONSE.fullmatch(completed.assistant_message.text)
        if match is None:
            raise MyCodeError("摘要响应缺少完整的 analysis/summary 标签")
        summary = match.group("summary").strip()
        if not summary:
            raise MyCodeError("摘要正文为空")
        # 用来保存九个固定标题在摘要正文中的位置，每个标题从第几个字符开始出现
        positions: list[int] = []
        for heading in SUMMARY_HEADINGS:
            if summary.count(heading) != 1:
                raise MyCodeError(f"摘要标题缺失或重复：{heading}")
            positions.append(summary.index(heading))
        if positions != sorted(positions):
            raise MyCodeError("摘要标题顺序不正确")
        return summary
