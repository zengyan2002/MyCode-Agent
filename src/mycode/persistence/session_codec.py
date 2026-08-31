"""在统一消息模型与一行 JSONL 记录之间转换。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

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
from mycode.models.sessions import SessionRuntimeMetadata
from mycode.models.skills import SkillSessionState
from mycode.models.teams import TeamBinding


class SessionDecodeError(ValueError):
    """表示某一行 JSONL 无法转换成 MyCode 对话消息。"""


class SkillSessionMetadataDecodeError(ValueError):
    """表示会话旁路文件无法还原成活动 Skill 名单。"""


class SessionRuntimeMetadataDecodeError(ValueError):
    """表示会话旁路文件无法还原 Skill 状态和团队绑定。"""


@dataclass(frozen=True)
class SessionRecord:
    """保存一条已经发生的对话消息及其写入时间。"""

    timestamp: datetime
    message: ChatMessage


def _string(value: Any, field: str) -> str:
    """
    JSONL 解码阶段的字符串字段检查器，防止类型错误的数据混进恢复后的会话。
    """
    if not isinstance(value, str):
        raise SessionDecodeError(f"字段 {field} 必须是字符串")
    return value


def _assistant_blocks(raw: Any) -> tuple[Any, ...]:
    """把会话文件中的助手 content 数组还原成消息内容块 Python 消息对象 → 可以写入 JSONL 的字典列表


    函数按照数组中的原始顺序检查每一项，并根据 type 字段创建TextBlock、ThinkingBlock、RedactedThinkingBlock 或 ToolCall。
    读取到未知类型或缺少必要字段时，拒绝恢复这条助手消息

    Args:
        raw: 从 JSONL 记录中读取的助手 content 字段，应当是由多个JSON 对象组成的数组。

    Returns:
        按原始顺序排列的助手消息内容块元组
    """
    if not isinstance(raw, list):
        raise SessionDecodeError("助手消息 content 必须是数组")
    blocks: list[Any] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SessionDecodeError(f"助手内容块 {index} 必须是对象")
        # 校验content中的type字段
        kind = _string(item.get("type"), f"content[{index}].type")
        if kind == "text":
            blocks.append(TextBlock(_string(item.get("text"), "text")))
        elif kind == "thinking":
            blocks.append(
                ThinkingBlock(
                    _string(item.get("thinking"), "thinking"),
                    _string(item.get("signature"), "signature"),
                )
            )
        elif kind == "redacted_thinking":
            blocks.append(
                RedactedThinkingBlock(_string(item.get("data"), "data"))
            )
        elif kind == "tool_call":
            arguments = item.get("arguments")
            if not isinstance(arguments, dict):
                raise SessionDecodeError("工具调用 arguments 必须是 JSON 对象")
            blocks.append(
                ToolCall(
                    _string(item.get("id"), "id"),
                    _string(item.get("name"), "name"),
                    arguments,
                )
            )
        else:
            raise SessionDecodeError(f"未知助手内容块类型：{kind}")
    return tuple(blocks)


def _encode_assistant(message: AssistantMessage) -> list[dict[str, Any]]:
    """把助手消息中的内容块转换成可写入 JSONL 的字典列表。  从 JSONL 读取的字典列表 → Python 消息对象

    函数按照原始顺序转换文本、思考、隐藏思考和工具调用内容块。
    返回的字典列表会由 SessionCodec.encode() 继续编码成 JSON 字符串。

    Args:
        message: 需要写入会话文件的助手消息。

    Returns:
        与助手内容块顺序一致、可以交给 json.dumps() 编码的字典列表。
    """
    blocks: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingBlock):
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.signature,
                }
            )
        elif isinstance(block, RedactedThinkingBlock):
            blocks.append({"type": "redacted_thinking", "data": block.data})
        else:
            blocks.append(
                {
                    "type": "tool_call",
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.arguments,
                }
            )
    return blocks


class SessionCodec:
    """把每条消息编码为独立 JSON 行，并严格解析恢复输入。"""

    def encode(self, record: SessionRecord) -> str:
        """把一条会话消息转换成可写入 JSONL 文件的 JSON 字符串。

        函数先写入消息发生时间，再根据消息类型保存用户正文、助手内容块
        或工具执行结果。返回值只包含一条 JSON 记录，不包含行末换行符；
        SessionManager 写入会话文件时会补上换行符

        Args:
            record: 需要保存的会话记录，包含消息发生时间和一条用户、助手或工具结果消息。

        Returns:
            表示这条会话记录的单行 JSON 字符串。
        """
        message = record.message
        payload: dict[str, Any] = {"ts": record.timestamp.isoformat()}
        if isinstance(message, UserMessage):
            payload.update({"type": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            payload.update(
                {"type": "assistant", "content": _encode_assistant(message)}
            )
        else:
            payload.update(
                {
                    "type": "tool_result",
                    "tool_call_id": message.tool_call_id,
                    "tool_name": message.tool_name,
                    "content": message.content,
                    "is_error": message.is_error,
                }
            )
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def decode(self, line: str) -> SessionRecord:
        """把会话文件中的一行 JSON 还原成会话记录

        函数先解析 JSON 和消息时间，再根据最外层 type 字段创建
        UserMessage、AssistantMessage 或 ToolResultMessage。读取助手消息时，
        还会继续解析 content 数组中的文本、思考和工具调用内容块。

        Args:
            line: 从 JSONL 会话文件中读取的一行 JSON 字符串。

        Returns:
            包含消息发生时间和对应消息对象的 SessionRecord。
        """
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SessionDecodeError("记录不是合法 JSON") from exc
        if not isinstance(payload, dict):
            raise SessionDecodeError("记录顶层必须是 JSON 对象")
        raw_timestamp = _string(payload.get("ts"), "ts")
        try:
            timestamp = datetime.fromisoformat(raw_timestamp)
        except ValueError as exc:
            raise SessionDecodeError("字段 ts 不是合法 ISO 8601 时间") from exc
        kind = _string(payload.get("type"), "type")
        if kind == "user":
            message: ChatMessage = UserMessage(
                _string(payload.get("content"), "content")
            )
        elif kind == "assistant":
            message = AssistantMessage(_assistant_blocks(payload.get("content")))
        elif kind == "tool_result":
            is_error = payload.get("is_error")
            if not isinstance(is_error, bool):
                raise SessionDecodeError("字段 is_error 必须是布尔值")
            message = ToolResultMessage(
                _string(payload.get("tool_call_id"), "tool_call_id"),
                _string(payload.get("tool_name"), "tool_name"),
                _string(payload.get("content"), "content"),
                is_error,
            )
        else:
            raise SessionDecodeError(f"未知消息类型：{kind}")
        return SessionRecord(timestamp, message)


class SessionRuntimeMetadataCodec:
    """编码和解析会话旁路文件中的 Skill 状态与可选团队绑定。"""

    def encode(self, state: SessionRuntimeMetadata) -> str:
        """把组合运行状态编码成一个 JSON 对象。

        Args:
            state: 当前 inline Skill 顺序和可选 TeamBinding。

        Returns:
            不带行末换行符的紧凑 JSON 文本。
        """

        payload: dict[str, Any] = {
            "active_skills": list(state.skills.active_skills),
        }
        if state.team is not None:
            payload["team"] = {
                "team_id": state.team.team_id,
                "lead_generation": state.team.lead_generation,
            }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def decode(self, text: str) -> SessionRuntimeMetadata:
        """把新旧两种旁路 JSON 还原成组合运行状态。

        Args:
            text: 从 ``.meta.json`` 读取的完整 UTF-8 文本。

        Returns:
            Skill 顺序和可选 TeamBinding。旧文件没有 team 字段时绑定为空。

        Raises:
            SessionRuntimeMetadataDecodeError: JSON 或字段类型无效。
        """

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SessionRuntimeMetadataDecodeError("会话元数据不是合法 JSON") from exc
        if not isinstance(payload, dict) or not set(payload).issubset({"active_skills", "team"}):
            raise SessionRuntimeMetadataDecodeError("会话元数据包含未知字段")
        raw_names = payload.get("active_skills", [])
        if not isinstance(raw_names, list):
            raise SessionRuntimeMetadataDecodeError("active_skills 必须是字符串数组")
        names: list[str] = []
        seen: set[str] = set()
        for index, raw_name in enumerate(raw_names):
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise SessionRuntimeMetadataDecodeError(
                    f"active_skills[{index}] 必须是非空字符串"
                )
            normalized = raw_name.strip().casefold()
            if normalized in seen:
                raise SessionRuntimeMetadataDecodeError(
                    f"active_skills 包含重复名字：{raw_name}"
                )
            seen.add(normalized)
            names.append(raw_name.strip())
        team_raw = payload.get("team")
        team: TeamBinding | None = None
        if team_raw is not None:
            if not isinstance(team_raw, dict) or set(team_raw) != {"team_id", "lead_generation"}:
                raise SessionRuntimeMetadataDecodeError("team 字段格式无效")
            team_id = team_raw.get("team_id")
            generation = team_raw.get("lead_generation")
            if not isinstance(team_id, str) or not team_id.strip():
                raise SessionRuntimeMetadataDecodeError("team_id 必须是非空字符串")
            if isinstance(generation, bool) or not isinstance(generation, int) or generation <= 0:
                raise SessionRuntimeMetadataDecodeError("lead_generation 必须是正整数")
            team = TeamBinding(team_id.strip(), generation)
        return SessionRuntimeMetadata(SkillSessionState(tuple(names)), team)


class SkillSessionMetadataCodec:
    """兼容旧调用方的 Skill-only 旁路编解码器。"""

    def encode(self, state: SkillSessionState) -> str:
        """把活动 Skill 顺序编码成旧格式兼容的 JSON。"""

        return SessionRuntimeMetadataCodec().encode(SessionRuntimeMetadata(state))

    def decode(self, text: str) -> SkillSessionState:
        """读取新旧旁路 JSON，并只返回其中的 Skill 状态。"""

        try:
            raw = json.loads(text)
            if not isinstance(raw, dict) or "active_skills" not in raw:
                raise SkillSessionMetadataDecodeError(
                    "Skill 会话元数据必须包含 active_skills 字段"
                )
            return SessionRuntimeMetadataCodec().decode(text).skills
        except json.JSONDecodeError as exc:
            raise SkillSessionMetadataDecodeError("Skill 会话元数据不是合法 JSON") from exc
        except SessionRuntimeMetadataDecodeError as exc:
            raise SkillSessionMetadataDecodeError(str(exc)) from exc
