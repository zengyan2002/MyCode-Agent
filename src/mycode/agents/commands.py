"""实现 `/agent` 的角色查询和热重载命令。"""

from __future__ import annotations

from mycode.commands.models import CommandContext, CommandResult


async def handle_agent_management(context: CommandContext) -> CommandResult:
    """执行 `/agent list|info|reload`，结果只显示在当前终端。

    Args:
        context: 命令分发器创建的上下文，其中 ``agent_service`` 提供当前
            角色目录、详细信息和逐角色热重载能力。

    Returns:
        不退出应用、不启动主 Agent 的空 CommandResult。参数或角色名错误
        会通过 UI 显示，不向命令循环抛出。
    """

    service = context.agent_service
    if service is None:
        context.ui.show_error("Agent 角色服务尚未启用")
        return CommandResult()
    args = context.invocation.args.split()
    if not args or args == ["list"]:
        context.ui.show_status(service.format_list())
        return CommandResult()
    if len(args) == 2 and args[0] == "info":
        try:
            text = service.format_info(args[1])
        except KeyError as exc:
            context.ui.show_error(str(exc))
        else:
            context.ui.show_status(text)
        return CommandResult()
    if args == ["reload"]:
        report = service.reload()
        lines = [
            "Agent 角色已重新扫描：",
            f"新增：{', '.join(report.added) or '无'}",
            f"更新：{', '.join(report.updated) or '无'}",
            f"删除：{', '.join(report.removed) or '无'}",
            f"保留旧版：{', '.join(report.retained) or '无'}",
        ]
        lines.extend(
            f"[{item.level.value}] {item.path}: {item.message}"
            for item in report.diagnostics
        )
        context.ui.show_status("\n".join(lines))
        return CommandResult()
    context.ui.show_error("用法：/agent [list|info <名称>|reload]")
    return CommandResult()
