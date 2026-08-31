"""登记并执行 MyCode 的 11 个内置斜杠命令。"""

from __future__ import annotations

import asyncio
import re
from collections import Counter

from mycode.commands.models import (
    AgentSubmission,
    Command,
    CommandContext,
    CommandResult,
    CommandType,
)
from mycode.commands.registry import CommandRegistry
from mycode.models.memory import MemoryType
from mycode.models.events import CompactionStatusKind
from mycode.models.permissions import (
    PermissionLayer,
    PermissionMode,
    PermissionRule,
    PermissionScope,
)
from mycode.skills.commands import handle_skill_management
from mycode.agents.commands import handle_agent_management
from mycode.worktrees.commands import handle_worktree

_SESSION_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")


def _show_usage(context: CommandContext, usage: str) -> CommandResult:
    """向当前 UI 显示参数用法错误。

    Args:
        context: 包含本次界面和命令输入的执行上下文。
        usage: 用户可以直接照着输入的完整命令用法。

    Returns:
        不退出应用、也不启动 Agent 的空命令结果。
    """

    context.ui.show_error(f"用法：{usage}")
    return CommandResult()


def _format_aliases(command: Command) -> str:
    """把一条命令的别名整理成帮助列表文字。

    Args:
        command: 需要展示正式名称和别名的命令定义。

    Returns:
        以斜杠开头、逗号分隔的正式名称和别名。
    """

    return ", ".join(f"/{value}" for value in (command.name, *command.aliases))


async def handle_help(context: CommandContext) -> CommandResult:
    """显示全部可见命令，或显示指定命令的详细用法。

    Args:
        context: 包含注册表、输入参数和当前 UI 的执行上下文。

    Returns:
        已在本地完成显示的空命令结果。
    """

    target = context.invocation.args.strip().removeprefix("/")
    if target:
        command = context.registry.find(target)
        if command is None or command.hidden:
            context.ui.show_error(
                f"未知命令：/{target}。输入 /help 查看可用命令"
            )
            return CommandResult()
        aliases = _format_aliases(command)
        context.ui.show_status(
            f"{aliases}\n{command.description}\n用法：{command.usage}"
        )
        return CommandResult()

    lines = ["可用命令："]
    lines.extend(
        f"{_format_aliases(command)}"
        f"{' [skill]' if command.skill else ''} — {command.description}"
        for command in context.registry.visible_commands
    )
    lines.append("输入 /help <命令名> 查看详细用法。")
    context.ui.show_status("\n".join(lines))
    return CommandResult()


async def handle_exit(context: CommandContext) -> CommandResult:
    """请求应用在当前命令结束后正常退出。

    Args:
        context: 包含本次命令参数的执行上下文。

    Returns:
        无参数时返回退出请求；参数错误时返回空结果。
    """

    if context.invocation.args:
        return _show_usage(context, "/exit")
    return CommandResult(exit_requested=True)


async def handle_plan(context: CommandContext) -> CommandResult:
    """进入计划模式，并可把参数作为规划任务交给 Agent。

    Args:
        context: 包含共享模式状态、UI 和原始参数的执行上下文。

    Returns:
        无参数时返回空结果；有任务时返回计划模式 Agent 提交内容。
    """

    task = context.invocation.args
    if task.casefold() in {"on", "off"}:
        context.ui.show_error(
            "不支持 /plan on|off；请用 /plan 进入计划模式，用 /do 返回执行模式"
        )
        return CommandResult()
    context.runtime_state.plan_only = True
    context.ui.set_plan_mode(True)
    if not task:
        context.ui.show_status("已切换到 Plan 模式")
        return CommandResult()
    return CommandResult(
        agent_submission=AgentSubmission(
            display_text=context.invocation.raw_input,
            prompt=task,
            plan_only=True,
        )
    )


async def handle_do(context: CommandContext) -> CommandResult:
    """退出计划模式并恢复可以执行操作的 Agent 模式。

    Args:
        context: 包含共享模式状态、UI 和原始参数的执行上下文。

    Returns:
        完成本地模式切换后的空命令结果。
    """

    if context.invocation.args:
        return _show_usage(context, "/do")
    context.runtime_state.plan_only = False
    context.ui.set_plan_mode(False)
    context.ui.show_status("已切换到执行模式")
    return CommandResult()


async def handle_clear(context: CommandContext) -> CommandResult:
    """创建空会话并清空界面，保留旧会话文件。

    Args:
        context: 包含 Agent、UI 和本次命令参数的执行上下文。

    Returns:
        新会话创建完成后的空命令结果。
    """

    if context.invocation.args:
        return _show_usage(context, "/clear")
    session_id = await context.agent.new_session()
    context.ui.clear_transcript()
    context.ui.show_status(f"已创建新的空会话：{session_id}")
    return CommandResult()


def _format_session_summary(context: CommandContext) -> str:
    """读取当前会话并整理四项概要字段。

    Args:
        context: 包含当前会话管理器的命令上下文。

    Returns:
        包含 ID、标题、最后活动时间和消息数的多行文字。
    """

    summary = context.session_manager.current_summary()
    return "\n".join(
        (
            f"会话 ID：{summary.session_id}",
            f"标题：{summary.title}",
            f"最后活动：{summary.last_active:%Y-%m-%d %H:%M:%S}",
            f"消息数：{summary.message_count}",
        )
    )


async def handle_session(context: CommandContext) -> CommandResult:
    """查询、新建、恢复或删除当前项目的会话。

    Args:
        context: 包含 SessionManager、Agent、UI 和取消令牌的执行上下文。

    Returns:
        完成所选会话操作后的空命令结果。
    """

    args = context.invocation.args.split()
    if not args:
        context.ui.show_status(_format_session_summary(context))
        return CommandResult()
    if args == ["list"]:
        sessions = await asyncio.to_thread(context.session_manager.list_sessions)
        if not sessions:
            context.ui.show_status("当前项目还没有会话")
        else:
            context.ui.show_status(
                "会话列表：\n"
                + "\n".join(
                    f"{item.last_active:%Y-%m-%d %H:%M} · "
                    f"{item.title} · {item.session_id}"
                    for item in sessions
                )
            )
        return CommandResult()
    if args == ["new"]:
        session_id = await context.agent.new_session()
        context.ui.clear_transcript()
        context.ui.show_status(f"已创建新会话：{session_id}")
        return CommandResult()
    if len(args) == 2 and args[0] == "resume":
        if _SESSION_ID.fullmatch(args[1]) is None:
            return _show_usage(
                context,
                "/session [list|new|resume <ID>|delete <ID>]",
            )
        result = await context.agent.restore_session(args[1], context.cancellation)
        details = [f"已恢复会话：{result.session_id}"]
        if result.skipped_lines:
            details.append(f"跳过 {result.skipped_lines} 行损坏记录")
        if result.chain_truncated:
            details.append("已截去缺少工具结果的末尾消息")
        if result.compactions:
            details.append(f"为适应上下文窗口压缩了 {result.compactions} 次")
        if result.time_gap_notice_added:
            details.append("已提示 Agent 留意会话间隔期间的文件变化")
        if result.worktree_warnings:
            details.extend(
                f"Worktree 警告：{warning}"
                for warning in result.worktree_warnings
            )
        context.ui.clear_transcript()
        context.ui.show_status("；".join(details))
        return CommandResult()
    if len(args) == 2 and args[0] == "delete":
        if _SESSION_ID.fullmatch(args[1]) is None:
            return _show_usage(
                context,
                "/session [list|new|resume <ID>|delete <ID>]",
            )
        confirmed = await context.ui.confirm(f"永久删除会话 {args[1]}？")
        if not confirmed:
            context.ui.show_status("已取消删除会话")
            return CommandResult()
        await asyncio.to_thread(context.session_manager.delete, args[1])
        context.ui.show_status(f"已删除会话：{args[1]}")
        return CommandResult()
    return _show_usage(
        context,
        "/session [list|new|resume <ID>|delete <ID>]",
    )


def _memory_scope(memory_type: MemoryType) -> str:
    """返回一种记忆类别所属的用户级或项目级范围。

    Args:
        memory_type: 笔记登记的四类记忆之一。

    Returns:
        用户级类别返回“用户级”，其余类别返回“项目级”。
    """

    return (
        "用户级"
        if memory_type in (MemoryType.USER, MemoryType.FEEDBACK)
        else "项目级"
    )


async def handle_memory(context: CommandContext) -> CommandResult:
    """只读展示长期记忆的数量、类别和笔记元数据。

    Args:
        context: 包含 MemoryStore、UI 和子命令参数的执行上下文。

    Returns:
        完成只读展示后的空命令结果。
    """

    args = context.invocation.args.split()
    if args not in ([], ["list"]):
        return _show_usage(context, "/memory [list]")
    snapshot = await asyncio.to_thread(context.memory_store.load_snapshot)
    counts = Counter(note.type for note in snapshot.notes)
    if not args:
        user_count = counts[MemoryType.USER] + counts[MemoryType.FEEDBACK]
        project_count = counts[MemoryType.PROJECT] + counts[MemoryType.REFERENCE]
        context.ui.show_status(
            "记忆概要：\n"
            f"用户级：{user_count}\n"
            f"项目级：{project_count}\n"
            + "\n".join(
                f"{memory_type.value}：{counts[memory_type]}"
                for memory_type in MemoryType
            )
        )
        return CommandResult()
    if not snapshot.notes:
        context.ui.show_status("当前没有长期记忆笔记")
        return CommandResult()
    notes = sorted(
        snapshot.notes,
        key=lambda note: (
            _memory_scope(note.type),
            note.type.value,
            note.filename.casefold(),
        ),
    )
    context.ui.show_status(
        "记忆列表：\n"
        + "\n".join(
            f"[{_memory_scope(note.type)}/{note.type.value}] "
            f"{note.filename} · {note.name} — {note.description}"
            for note in notes
        )
    )
    return CommandResult()


def _format_permission_rule(rule: PermissionRule) -> str:
    """把一条已编译权限规则整理成只读列表文字。

    Args:
        rule: 包含工具、匹配模式、效果和来源的权限规则。

    Returns:
        不包含内部正则对象的单行规则说明。
    """

    return (
        f"{rule.tool.value}({rule.pattern}) → {rule.effect.value} "
        f"· {rule.source}"
    )


def _permission_layers(context: CommandContext) -> tuple[PermissionLayer, ...]:
    """取得当前生效顺序中的四层权限规则。

    Args:
        context: 包含 Controller 最新状态和启动配置快照的命令上下文。

    Returns:
        按 SESSION、LOCAL、PROJECT、USER 排列的权限层。
    """

    session = PermissionLayer(
        PermissionScope.SESSION,
        None,
        context.permission_controller.session_rules(),
        None,
    )
    return (
        session,
        context.permission_controller.local_layer(),
        context.permission_settings.project,
        context.permission_settings.user,
    )


async def handle_permission(context: CommandContext) -> CommandResult:
    """查询权限模式、切换模式或只读列出分层规则。

    Args:
        context: 包含权限 Controller、启动配置和 UI 的执行上下文。

    Returns:
        完成本地权限操作后的空命令结果。
    """

    args = context.invocation.args.split()
    if not args:
        context.ui.show_status(
            f"当前权限模式：{context.permission_controller.mode.value}"
        )
        return CommandResult()
    if len(args) == 2 and args[0] == "mode":
        try:
            mode = PermissionMode(args[1])
        except ValueError:
            return _show_usage(
                context,
                "/permission [rules|mode strict|default|allow]",
            )
        context.permission_controller.set_mode(mode)
        context.ui.set_permission_mode(mode)
        context.ui.show_status(f"权限模式已切换为：{mode.value}")
        return CommandResult()
    if args == ["rules"]:
        lines = ["权限规则："]
        for layer in _permission_layers(context):
            mode = layer.mode.value if layer.mode is not None else "未设置"
            lines.append(f"[{layer.scope.value.upper()}] mode={mode}")
            lines.extend(
                f"  {_format_permission_rule(rule)}" for rule in layer.rules
            )
            if not layer.rules:
                lines.append("  无规则")
        context.ui.show_status("\n".join(lines))
        return CommandResult()
    return _show_usage(
        context,
        "/permission [rules|mode strict|default|allow]",
    )


async def handle_status(context: CommandContext) -> CommandResult:
    """汇总模式、模型、上下文、会话和权限五类状态。

    Args:
        context: 包含所有状态查询依赖的执行上下文。

    Returns:
        状态已显示后的空命令结果。
    """

    if context.invocation.args:
        return _show_usage(context, "/status")
    plan_only = context.runtime_state.plan_only
    tokens = context.agent.estimate_input_tokens(plan_only=plan_only)
    session = context.session_manager.current_summary()
    lines = [
                f"模式：{'PLAN' if plan_only else 'DEFAULT'}",
                f"Provider：{context.provider_config.name}/{context.provider_config.model}",
                f"当前输入 Token：约 {tokens:,}/{context.provider_config.context_window_tokens:,}",
                f"会话：{session.session_id}（{session.message_count} 条消息）",
                f"权限：{context.permission_controller.mode.value}",
    ]
    if context.worktree_manager is not None:
        assignment = context.worktree_manager.binding.snapshot()
        lines.extend(
            (
                f"工作目录：{assignment.root}",
                f"Worktree：{assignment.worktree_name or '主仓库'}",
                f"分支：{assignment.branch or 'detached'}",
            )
        )
    context.ui.show_status("\n".join(lines))
    return CommandResult()


async def handle_compact(context: CommandContext) -> CommandResult:
    """手动压缩较早对话，并在成功时显示估算变化。

    Args:
        context: 包含 Agent、UI、共享模式和取消令牌的执行上下文。

    Returns:
        摘要流结束后的空命令结果。
    """

    before = context.agent.estimate_input_tokens(
        plan_only=context.runtime_state.plan_only
    )
    succeeded = False
    async for event in context.agent.stream_compact(
        retention_focus=context.invocation.args or None,
        cancellation=context.cancellation,
    ):
        context.ui.render_event(event)
        succeeded = event.kind is CompactionStatusKind.SUCCEEDED
    if succeeded:
        after = context.agent.estimate_input_tokens(
            plan_only=context.runtime_state.plan_only
        )
        context.ui.show_status(
            f"当前输入 Token 估算：约 {before:,} → 约 {after:,}"
        )
    return CommandResult()


def create_builtin_registry() -> CommandRegistry:
    """登记并冻结本次进程使用的内置命令。

    Returns:
        已完成名称和别名冲突检查、后续只读的命令注册表。
    """

    registry = CommandRegistry()
    commands = (
        Command("help", ("h", "?"), "显示命令帮助", "/help [命令名]", CommandType.LOCAL, handle_help),
        Command("compact", ("c",), "压缩较早对话", "/compact [额外保留重点]", CommandType.LOCAL, handle_compact),
        Command("clear", (), "创建空会话并清屏", "/clear", CommandType.LOCAL_UI, handle_clear),
        Command("plan", ("p",), "进入计划模式", "/plan [任务]", CommandType.LOCAL_UI, handle_plan),
        Command("do", (), "切换到执行模式", "/do", CommandType.LOCAL_UI, handle_do),
        Command("session", (), "查询和管理会话", "/session [list|new|resume <ID>|delete <ID>]", CommandType.LOCAL, handle_session),
        Command("memory", (), "只读查看长期记忆", "/memory [list]", CommandType.LOCAL, handle_memory),
        Command("permission", (), "查询权限或切换模式", "/permission [rules|mode strict|default|allow]", CommandType.LOCAL, handle_permission),
        Command("status", ("s",), "显示当前综合状态", "/status", CommandType.LOCAL, handle_status),
        Command("skill", (), "查看和维护 Skill", "/skill [list|info <名称>|reload|deactivate <名称>]", CommandType.LOCAL, handle_skill_management),
        Command("agent", (), "查看和维护子 Agent 角色", "/agent [list|info <名称>|reload]", CommandType.LOCAL, handle_agent_management),
        Command("worktree", ("wt",), "创建和管理隔离工作目录", "/worktree [list|create <名称>|status <名称>|enter <名称>|exit|remove <名称>|delete-branch <名称>]", CommandType.LOCAL, handle_worktree),
        Command("exit", (), "退出 MyCode", "/exit", CommandType.LOCAL, handle_exit),
    )
    for command in commands:
        registry.register(command)
    registry.freeze()
    return registry
