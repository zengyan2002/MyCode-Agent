"""把 Skill Catalog 映射成动态斜杠命令，并实现 /skill 管理入口。"""

from __future__ import annotations

from collections.abc import Sequence

from mycode.commands.models import (
    Command,
    CommandContext,
    CommandResult,
    CommandType,
)
from mycode.errors import MyCodeError
from mycode.models.skills import SkillDefinition


async def handle_skill_command(context: CommandContext) -> CommandResult:
    """把当前动态命令转换成 inline 或 fork Skill 提交。

    Args:
        context: 包含规范命令名、原始参数和当前 SkillService 的命令上下文。

    Returns:
        Application 可以继续执行的 Skill 提交；目标已经删除时显示错误并
        返回空结果。
    """

    service = context.skill_service
    if service is None:
        context.ui.show_error("Skill 系统尚未初始化")
        return CommandResult()
    submission = service.submission_for(
        context.invocation.name,
        context.invocation.args,
        context.invocation.raw_input,
    )
    if submission is None:
        context.ui.show_error(
            f"Skill /{context.invocation.name} 当前不可用，请执行 /skill reload"
        )
        return CommandResult()
    return CommandResult(skill_submission=submission)


def build_skill_commands(
    skills: Sequence[SkillDefinition],
) -> tuple[Command, ...]:
    """为当前有效 Skill 创建共用同一个 handler 的动态命令。

    Args:
        skills: Catalog 中已经完成覆盖选择和解析校验的 Skill 定义。

    Returns:
        按传入顺序排列、可交给 CommandRegistry 原子替换的 Command 元组。
    """

    return tuple(
        Command(
            name=skill.name,
            aliases=(),
            description=skill.description,
            usage=f"/{skill.name} [参数]",
            type=CommandType.PROMPT,
            handler=handle_skill_command,
            skill=True,
        )
        for skill in skills
    )


async def handle_skill_management(context: CommandContext) -> CommandResult:
    """执行 /skill list、info、reload 或 deactivate 子命令。

    Args:
        context: 包含子命令文本、UI 和当前 SkillService 的命令上下文。

    Returns:
        管理操作完成后的空命令结果，不会启动普通 Agent Loop。
    """

    service = context.skill_service
    if service is None:
        context.ui.show_error("Skill 系统尚未初始化")
        return CommandResult()
    parts = context.invocation.args.split()
    if not parts or parts == ["list"]:
        context.ui.show_status(service.format_list())
        return CommandResult()
    if len(parts) == 2 and parts[0] == "info":
        try:
            context.ui.show_status(service.format_info(parts[1]))
        except MyCodeError as exc:
            context.ui.show_error(str(exc))
        return CommandResult()
    if parts == ["reload"]:
        context.ui.show_status(service.format_reload(service.reload()))
        return CommandResult()
    if len(parts) == 2 and parts[0] == "deactivate":
        if service.deactivate(parts[1]):
            context.ui.show_status(f"已停用 Skill：{parts[1]}")
        else:
            context.ui.show_error(
                f"Skill {parts[1]} 没有在当前主会话中激活"
            )
        return CommandResult()
    context.ui.show_error(
        "用法：/skill [list|info <名称>|reload|deactivate <名称>]"
    )
    return CommandResult()
