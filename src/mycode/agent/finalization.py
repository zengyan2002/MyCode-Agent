"""生成模型调用预算提醒，并解析最后一次请求返回的正式报告。"""

from __future__ import annotations

from mycode.models.events import AgentFinalizationProfile
from mycode.models.prompts import RuntimeInstruction, RuntimeInstructionKind

_OPEN_TAG = "<final-report>"
_CLOSE_TAG = "</final-report>"

_PROFILE_REQUIREMENTS = {
    AgentFinalizationProfile.MAIN: (
        "直接回答用户，并明确列出尚未核实的内容；不要输出内部状态码。"
    ),
    AgentFinalizationProfile.EXPLORE: (
        "报告关键文件、类、函数、调用顺序，以及仍未确认的分支。"
    ),
    AgentFinalizationProfile.PLAN: (
        "报告涉及模块、数据结构、状态流、风险和测试方案。"
    ),
    AgentFinalizationProfile.VERIFICATION: (
        "给出 PASS、FAIL 或 PARTIAL 判断、实际证据和未执行检查。"
    ),
    AgentFinalizationProfile.GENERIC: (
        "按照当前角色的系统提示完成任务，并报告仍然存在的缺口。"
    ),
}


def budget_instruction(
    remaining_model_calls: int,
) -> RuntimeInstruction | None:
    """在只剩 2～5 次模型调用时生成本次请求专用的收敛提醒。

    Args:
        remaining_model_calls: 当前请求发出前还可调用模型的次数，包含本次。

    Returns:
        剩余 2～5 次时返回提醒；其余阶段返回 ``None``。

    Raises:
        ValueError: 剩余次数不是正整数。
    """

    if (
        isinstance(remaining_model_calls, bool)
        or not isinstance(remaining_model_calls, int)
        or remaining_model_calls <= 0
    ):
        raise ValueError("剩余模型调用次数必须是正整数")
    if remaining_model_calls > 5 or remaining_model_calls == 1:
        return None
    return RuntimeInstruction(
        RuntimeInstructionKind.MODEL_CALL_BUDGET,
        (
            f"当前任务还剩 {remaining_model_calls} 次模型调用（包含本次）。"
            "停止扩大调查范围，只补齐回答所必需的证据，并开始组织最终回答。"
            "最后一次调用不会提供工具。"
        ),
    )


def finalization_instruction(
    profile: AgentFinalizationProfile,
) -> RuntimeInstruction:
    """生成最后一次无工具请求使用的正式报告要求。

    Args:
        profile: 当前运行的角色类型，用来选择报告内容侧重点。

    Returns:
        一条要求模型停止调查并输出完整 ``final-report`` 的运行时指令。

    Raises:
        ValueError: ``profile`` 不是支持的角色枚举。
    """

    if not isinstance(profile, AgentFinalizationProfile):
        raise ValueError("强制收尾 profile 类型无效")
    return RuntimeInstruction(
        RuntimeInstructionKind.FINALIZATION,
        (
            "这是当前任务最后一次模型调用，工具已经不可用。立即停止调查，"
            "只能依据已经取得的证据完成正式报告。报告要区分已确认内容、合理推断"
            "和尚未核实内容。把全部正式正文放入且只放入一组完整的 "
            f"{_OPEN_TAG}...{_CLOSE_TAG} 标记中。"
            f"{_PROFILE_REQUIREMENTS[profile]}"
        ),
    )


def strip_model_budget_instructions(
    runtime: tuple[RuntimeInstruction, ...],
) -> tuple[RuntimeInstruction, ...]:
    """移除父 Agent 已经过期的预算提醒和强制收尾指令。

    Args:
        runtime: Fork 准备继承的父请求运行时指令。

    Returns:
        保持原顺序、但不含预算和收尾种类的新元组。
    """

    removed = {
        RuntimeInstructionKind.MODEL_CALL_BUDGET,
        RuntimeInstructionKind.FINALIZATION,
    }
    return tuple(item for item in runtime if item.kind not in removed)


def parse_final_report(text: str) -> str | None:
    """从最后一次响应中提取一份完整且非空的正式报告。

    Args:
        text: Provider 返回的完整可见文本。

    Returns:
        文本首尾恰好由一组 ``final-report`` 包裹时返回去掉标记的正文；
        标记缺失、重复、未闭合或正文为空时返回 ``None``。
    """

    candidate = text.strip()
    if not candidate.startswith(_OPEN_TAG) or not candidate.endswith(_CLOSE_TAG):
        return None
    if candidate.count(_OPEN_TAG) != 1 or candidate.count(_CLOSE_TAG) != 1:
        return None
    body = candidate[len(_OPEN_TAG) : -len(_CLOSE_TAG)].strip()
    return body or None
