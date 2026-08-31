"""构造定义式系统指令、Fork 消息补齐和 Fork 行为约束。"""

from __future__ import annotations

import json

from mycode.agents.snapshots import ParentRunSnapshot
from mycode.models.agents import AgentDefinition
from mycode.models.messages import ChatMessage, ToolResultMessage, UserMessage
from mycode.models.prompts import PromptSection


def definition_role_section(role: AgentDefinition) -> PromptSection:
    """把角色正文转换成稳定系统提示 section。

    Args:
        role: Catalog 中已经生效的定义式角色。

    Returns:
        名字和优先级固定、正文来自角色 Markdown 的 PromptSection。
    """

    return PromptSection(
        name=f"subagent-role-{role.key}",
        priority=910,
        content=role.prompt_body,
    )


def subagent_constraints_section() -> PromptSection:
    """生成所有独立子 Agent 都必须遵守的固定执行约束。

    Returns:
        要求非交互、不能再次委派、只做给定任务并最终汇报的稳定 section。
    """

    return PromptSection(
        name="independent-subagent-constraints",
        priority=900,
        content=(
            "你正在独立完成一项已经分配好的任务。不要与用户对话，不要提问，"
            "不要创建其他 Agent 或管理后台任务。直接使用当前可见工具，在任务"
            "边界内完成工作，并在结束时返回可供主 Agent 使用的结果。"
        ),
    )


def fork_boilerplate(task_prompt: str) -> str:
    """把 Fork 任务和不可协商的输出规则包装成一条用户消息。

    Args:
        task_prompt: 主 Agent 分配给 Fork 子 Agent 的完整任务正文。

    Returns:
        含 ``<fork_boilerplate>`` 标签和结构化汇报字段的文本。

    Raises:
        ValueError: 任务正文为空。
    """

    if not task_prompt.strip():
        raise ValueError("Fork 任务正文不能为空")
    return (
        f"{task_prompt.strip()}\n\n"
        "<fork_boilerplate>\n"
        "你是从主 Agent Fork 出来的独立工作进程，不是主 Agent。\n"
        "1. 不得再次创建 Agent 或管理后台任务。\n"
        "2. 不得与用户对话、提问或等待确认。\n"
        "3. 直接使用当前可见工具，只完成上面的任务范围。\n"
        "4. 最终按 Scope、Result、Key files、Files changed、Issues 汇报。\n"
        "</fork_boilerplate>"
    )


def build_fork_messages(
    snapshot: ParentRunSnapshot,
    task_prompt: str,
) -> tuple[ChatMessage, ...]:
    """复制父请求前缀，并补齐当前助手消息中的待处理工具调用。

    Args:
        snapshot: Recorder 保存的实际父 Provider 请求、工具视图和响应。
        task_prompt: Fork 子 Agent 要追加执行的新任务。

    Returns:
        父请求消息、完整助手响应、每个工具调用的 placeholder 结果，以及
        最后一条 Fork 任务用户消息组成的合法历史。
    """

    messages: list[ChatMessage] = [*snapshot.request.messages, snapshot.response]
    for call in snapshot.response.tool_calls:
        placeholder = json.dumps(
            {
                "success": False,
                "content": "该工具调用由父 Agent 继续处理，Fork 历史只保留协议占位。",
                "error_code": "fork_placeholder",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        messages.append(
            ToolResultMessage(
                tool_call_id=call.id,
                tool_name=call.name,
                content=placeholder,
                is_error=True,
            )
        )
    messages.append(UserMessage(fork_boilerplate(task_prompt)))
    return tuple(messages)
