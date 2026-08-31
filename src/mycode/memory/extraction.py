"""构造自动笔记提取请求，并解析模型返回的操作。"""

from __future__ import annotations

import json

from jsonschema import Draft202012Validator

from mycode.errors import MyCodeError
from mycode.models.memory import (
    CompletedTurn,
    MemoryAction,
    MemoryNote,
    MemoryOperation,
    MemorySnapshot,
    MemoryType,
    MemoryUpdate,
)
from mycode.models.messages import (
    AssistantMessage,
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

MEMORY_EXTRACTION_PROMPT = """你负责从一个刚结束的编程对话回合中维护长期笔记。
本请求不提供工具。只依据给出的本轮消息和现有笔记返回 JSON，不要输出代码围栏或说明文字。

只记录以后仍有用的信息：
- user：跨项目适用的用户编码偏好和表达偏好。
- feedback：用户明确指出助手哪里做错、以后应该怎样做。
- project：只适用于当前项目的技术栈、结构、约束和重要决定。
- reference：当前项目需要反复查阅的外部资料或链接。

已有同义笔记时更新原文件，不要创建重复文件。短期请求、寒暄、一次性进度和“做完了”不值得记录。
没有值得记录的内容时返回 {"operations":[]}。

输出对象只有 operations 字段。每项操作格式如下：
- 创建或更新：{"action":"create|update","type":"user|feedback|project|reference","target":"note.md","note":{"filename":"note.md","name":"名称","description":"一句说明","type":"同上","body":"实际要记住的内容及适用方式"}}
- 删除：{"action":"delete","type":"user|feedback|project|reference","target":"note.md"}
"""

# 用来规定后台模型输出的“一份记忆笔记”必须长什么样，只负责检查数据格式，不负责创建或写入文件
_NOTE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["filename", "name", "description", "type", "body"],
    "properties": {
        "filename": {"type": "string"},
        "name": {"type": "string"},
        "description": {"type": "string"},
        "type": {"enum": [item.value for item in MemoryType]},
        "body": {"type": "string"},
    },
}


# 规定模型返回的整批记忆修改JSON
MEMORY_UPDATE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["operations"],
    "properties": {
        "operations": {
            "type": "array",
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["action", "type", "target", "note"],
                        "properties": {
                            "action": {"enum": ["create", "update"]},
                            "type": {"enum": [item.value for item in MemoryType]},
                            "target": {"type": "string"},
                            "note": _NOTE_SCHEMA,
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["action", "type", "target"],
                        "properties": {
                            "action": {"const": "delete"},
                            "type": {"enum": [item.value for item in MemoryType]},
                            "target": {"type": "string"},
                        },
                    },
                ]
            },
        }
    },
}


class MemoryResponseFormatError(MyCodeError):
    """模型正常结束，但返回内容不能转换成一整批记忆操作。"""


def _message_payload(message: ChatMessage) -> dict[str, object]:
    """把一条会话消息转换成可写入记忆提取请求的字典

    用户消息保留正文；工具结果保留调用 ID、工具名、结果正文和错误标记；
    助手消息按原顺序转换文本、工具调用和思考内容。普通思考不携带签名,
    脱敏思考只记录类型，不传递其不透明数据。

    Args:
        message: 本轮对话中的用户消息、助手消息或工具结果

    Returns:
        包含消息角色和实际内容的字典，可继续使用 json.dumps() 编码
    """
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, ToolResultMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "content": message.content,
            "is_error": message.is_error,
        }
    # 走到这里说明是助手消息
    blocks: list[dict[str, object]] = []
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


class MemoryExtractionCodec:
    """把本轮对话和已有笔记整理成模型请求

    模型返回后，本类检查 JSON 格式，并转换成创建、更新或删除笔记的操作。
    本类不负责发送请求，也不负责写入笔记文件
    """

    def __init__(self) -> None:
        # 模型 JSON 格式检查器
        self._validator = Draft202012Validator(MEMORY_UPDATE_SCHEMA)

    def build_request(
        self,
        turn: CompletedTurn,
        snapshot: MemorySnapshot,
    ) -> ProviderRequest:
        """把刚结束的一轮对话和现有笔记整理成记忆提取请求

        函数将会话 ID、本轮新增消息、用户级和项目级索引，以及已有笔记正文
        编码成 JSON，作为一条用户消息交给记忆提取模型。该请求使用固定提示词，
        并关闭工具调用。函数只构造请求，不负责发送请求或修改笔记文件

        Args:
            turn: 刚刚正常结束的对话回合，包含会话 ID 和本轮新增消息。
            snapshot: 请求开始前读取的记忆内容，包含两级索引和已有笔记。

        Returns:
            可以交给 ProviderRequestRunner 发送的无工具模型请求
        """

        # 组装要发给模型的全部材料
        material = {
            "session_id": turn.session_id,
            "turn_messages": [_message_payload(item) for item in turn.messages],
            "indexes": {
                "user": snapshot.user_index,
                "project": snapshot.project_index,
            },
            "existing_notes": [
                {
                    "filename": note.filename,
                    "name": note.name,
                    "description": note.description,
                    "type": note.type.value,
                    "body": note.body,
                }
                for note in snapshot.notes
            ],
        }
        return ProviderRequest(
            messages=(
                UserMessage(
                    json.dumps(material, ensure_ascii=False, separators=(",", ":"))
                ),
            ),
            tools=(),
            tool_choice=ToolChoice.NONE,
            prompt=PromptContext(MEMORY_EXTRACTION_PROMPT),
        )

    def build_correction_request(
        self,
        original: ProviderRequest,
        completed: ProviderCompleted,
        error: MemoryResponseFormatError,
    ) -> ProviderRequest:
        """根据首次响应的格式错误构造一次纠正请求

         函数保留首次请求中的对话材料和请求设置，并在消息末尾加入模型首次返回的
        无效内容以及具体校验错误，让模型重新输出一份完整 JSON。函数只构造第二次
        请求，不负责发送，也不会尝试在本地修补首次响应。

        Args:
            original: 首次记忆提取请求，包含本轮对话、已有笔记和请求设置。
            completed: 模型首次正常结束但未通过格式检查的完整响应。
            error: 解析首次响应时得到的具体 JSON、Schema 或字段错误。

        Returns:
            保留原请求设置并附带纠正说明的第二次模型请求
        """

        correction = UserMessage(
            "上一次响应未通过校验："
            f"{error}\n"
            "请根据最初提供的材料重新输出一份完整 JSON。"
            "只输出符合规定结构的 JSON，不要解释，也不要使用代码围栏。"
        )
        return ProviderRequest(
            messages=(
                *original.messages,
                completed.assistant_message,
                correction,
            ),
            tools=original.tools,
            tool_choice=original.tool_choice,
            prompt=original.prompt,
            max_output_tokens=original.max_output_tokens,
        )

    def parse(self, completed: ProviderCompleted) -> MemoryUpdate:
        """检查模型返回的记忆 JSON，并转换成笔记修改操作

        函数先确认模型正常结束回复，再解析 JSON 并检查整体结构。格式通过后，将其中的创建、更新和删除数据转换成 MemoryOperation，最后组成一批
        MemoryUpdate。本函数只解析响应，不会修改笔记文件

        Args:
            completed: Provider 返回的完整响应，正文应为记忆修改 JSON

        Returns:
            模型提出的整批笔记修改；没有需要记录的内容时，operations 为空
        """

        if completed.stop_reason is not ModelStopReason.END_TURN:
            raise MyCodeError("记忆提取响应未正常结束")
        text = completed.assistant_message.text.strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MemoryResponseFormatError(
                "记忆提取响应不是合法 JSON"
            ) from exc
        errors = sorted(
            self._validator.iter_errors(payload),
            key=lambda error: tuple(str(item) for item in error.absolute_path),
        )
        if errors:
            raise MemoryResponseFormatError(
                f"记忆提取响应格式无效：{errors[0].message}"
            )
        try:
            operations: list[MemoryOperation] = []
            for item in payload["operations"]:
                action = MemoryAction(item["action"])
                memory_type = MemoryType(item["type"])
                note_payload = item.get("note")
                note = (
                    MemoryNote(
                        filename=note_payload["filename"],
                        name=note_payload["name"],
                        description=note_payload["description"],
                        type=MemoryType(note_payload["type"]),
                        body=note_payload["body"],
                    )
                    if note_payload is not None
                    else None
                )
                operations.append(
                    MemoryOperation(
                        action=action,
                        type=memory_type,
                        target=item["target"],
                        note=note,
                    )
                )
            return MemoryUpdate(tuple(operations))
        except (KeyError, TypeError, ValueError) as exc:
            raise MemoryResponseFormatError(
                f"记忆提取响应字段无效：{exc}"
            ) from exc
