"""生成 Team Lead 和成员每轮都需要遵守的简短运行说明。"""

from __future__ import annotations

from mycode.models.teams import TeamActorContext


def build_team_instruction(actor: TeamActorContext) -> str:
    """根据可信团队身份生成不会泄漏其他会话内容的运行说明。

    Args:
        actor: 本地运行时创建的 Lead 或成员身份。

    Returns:
        Lead 收到协调、合并和验证规则；成员收到认领、沟通和提交规则。
    """

    if actor.actor_kind == "lead":
        return (
            "你正在管理一个 Agent Team。你负责拆分任务、理解成员结果、"
            "协调依赖、合并已验收提交并运行验证；团队存续期间不要直接实现功能代码。"
            "创建成员必须调用 Agent，并同时填写 team_name、name、subagent_type 和 prompt；"
            "团队成员由已选后端长期运行，不要填写 run_in_background，也不要用 Shell 查找创建方式。"
        )
    return (
        "你是 Agent Team 的长期成员。先用 TeamTaskList 查看任务并自主认领；"
        "需要队友知道的信息必须用 SendMessage，普通文本回复不会被其他成员看到；"
        "代码任务完成前提交改动并保持 Worktree 干净。"
    )
