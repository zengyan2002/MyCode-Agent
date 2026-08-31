"""解析并执行用户手动管理 Worktree 的 ``/worktree`` 命令。"""

from __future__ import annotations

from mycode.commands.models import CommandContext, CommandResult
from mycode.models.worktrees import (
    CommitRelation,
    WorktreeChangeSummary,
    WorktreeSnapshot,
)
from mycode.worktrees.manager import WorktreeManagerError


_USAGE = (
    "/worktree [list|create <名称>|status <名称>|enter <名称>|exit|"
    "remove <名称>|delete-branch <名称>]"
)


def _show_usage(context: CommandContext) -> CommandResult:
    """显示 Worktree 命令的完整参数格式。

    Args:
        context: 包含当前终端界面的命令上下文。

    Returns:
        不退出应用、也不启动 Agent 的空命令结果。
    """

    context.ui.show_error(f"用法：{_USAGE}")
    return CommandResult()


def _format_snapshot(snapshot: WorktreeSnapshot) -> str:
    """把一条 Manager 快照整理成用户可读的多行状态。

    Args:
        snapshot: ``WorktreeManager.list/status`` 返回的只读快照。

    Returns:
        包含名称、路径、分支、类型、生命周期、租约和会话绑定的文字。
    """

    record = snapshot.record
    sessions = snapshot.session_ids
    return "\n".join(
        (
            f"名称：{record.name}",
            f"路径：{record.path}",
            f"分支：{record.branch}",
            f"类型：{record.kind.value}",
            f"状态：{record.lifecycle.value}",
            f"占用：{'是' if snapshot.leased else '否'}",
            f"绑定会话：{', '.join(sessions) if sessions else '无'}",
            f"最后使用：{record.last_used_at:%Y-%m-%d %H:%M:%S}",
        )
    )


def _format_changes(changes: WorktreeChangeSummary) -> str:
    """把 Git 变更摘要整理成 ``/worktree status`` 的附加信息。

    Args:
        changes: Manager 返回的文件、提交关系和未推送状态摘要。

    Returns:
        暂存、未暂存、未追踪数量，以及提交关系和未推送数量。
    """

    return "\n".join(
        (
            f"暂存修改：{len(changes.staged)}",
            f"未暂存修改：{len(changes.unstaged)}",
            f"未追踪文件：{len(changes.untracked)}",
            f"相对基线：{changes.relation_to_base.value}",
            f"新增提交：{changes.new_commit_count}",
            "未推送提交："
            + (
                "未知"
                if changes.unpushed_commit_count is None
                else str(changes.unpushed_commit_count)
            ),
        )
    )


async def handle_worktree(context: CommandContext) -> CommandResult:
    """创建、查询、进入、退出或分两步删除受管 Worktree。

    Args:
        context: 包含命令参数、当前会话、AgentLoop、UI 和 WorktreeManager 的
            一次命令执行上下文。

    Returns:
        所有子命令都在本地执行，返回不退出应用、不启动 Agent 的空结果。
        参数错误、用户取消或 Manager 拒绝操作也返回空结果，并由 UI 展示原因。
    """

    manager = context.worktree_manager
    if manager is None:
        context.ui.show_error("Worktree 功能尚未初始化")
        return CommandResult()
    args = context.invocation.args.split()
    try:
        if not args or args == ["list"]:
            snapshots = await manager.list()
            if not snapshots:
                context.ui.show_status("当前没有受 MyCode 管理的 Worktree")
            else:
                context.ui.show_status(
                    "Worktree 列表：\n\n"
                    + "\n\n".join(_format_snapshot(item) for item in snapshots)
                )
            return CommandResult()

        if len(args) == 2 and args[0] == "create":
            record = await manager.create_manual(
                args[1],
                context.session_manager.current_id,
            )
            context.ui.show_status(
                f"已创建 Worktree：{record.name}\n"
                f"路径：{record.path}\n分支：{record.branch}"
            )
            return CommandResult()

        if len(args) == 2 and args[0] == "status":
            snapshot = await manager.status(args[1])
            details = [_format_snapshot(snapshot)]
            try:
                changes = await manager.inspect_changes(args[1])
            except WorktreeManagerError as exc:
                details.append(f"变更状态：无法检查（{exc}）")
            else:
                details.append(_format_changes(changes))
            context.ui.show_status("\n".join(details))
            return CommandResult()

        if len(args) == 2 and args[0] == "enter":
            assignment = await manager.bind_session(
                context.session_manager.current_id,
                args[1],
            )
            warnings = await context.agent.activate_workspace(assignment)
            lines = [
                f"当前会话已进入 Worktree：{args[1]}",
                f"路径：{assignment.root}",
                f"分支：{assignment.branch}",
            ]
            lines.extend(f"项目指令警告：{item}" for item in warnings)
            context.ui.show_status("\n".join(lines))
            return CommandResult()

        if args == ["exit"]:
            assignment, warnings = await context.agent.exit_workspace()
            lines = [f"当前会话已回到主仓库：{assignment.root}"]
            lines.extend(f"项目指令警告：{item}" for item in warnings)
            context.ui.show_status("\n".join(lines))
            return CommandResult()

        if len(args) == 2 and args[0] == "remove":
            snapshot = await manager.status(args[1])
            try:
                changes = await manager.inspect_changes(args[1])
            except WorktreeManagerError as exc:
                discard_changes = True
                risk = f"无法可靠确认变更：{exc}"
            else:
                discard_changes = not (
                    not changes.has_file_changes
                    and changes.relation_to_base is CommitRelation.SAME
                    and changes.new_commit_count == 0
                    and changes.unpushed_commit_count == 0
                )
                risk = (
                    "目录包含文件修改、新提交或未推送提交"
                    if discard_changes
                    else "目录可以确认没有文件修改和新提交"
                )
            confirmed = await context.ui.confirm(
                f"移除 Worktree 目录？\n名称：{snapshot.record.name}\n"
                f"路径：{snapshot.record.path}\n分支：{snapshot.record.branch}\n"
                f"受影响的非活动会话：{len(snapshot.session_ids)}\n{risk}\n"
                + (
                    "确认后会丢弃目录中的修改和提交；分支仍会保留。"
                    if discard_changes
                    else "分支仍会保留，之后需单独执行 delete-branch。"
                )
            )
            if not confirmed:
                context.ui.show_status("已取消移除 Worktree")
                return CommandResult()
            report = await manager.remove(
                args[1],
                discard_changes=discard_changes,
            )
            context.ui.show_status(report.message)
            return CommandResult()

        if len(args) == 2 and args[0] == "delete-branch":
            snapshot = await manager.status(args[1])
            merged = await manager.branch_merged_status(args[1])
            discard_commits = merged is not True
            risk = (
                "分支已确认合入创建基准"
                if merged is True
                else (
                    "分支尚未合入创建基准"
                    if merged is False
                    else "无法确认分支是否已合入创建基准"
                )
            )
            confirmed = await context.ui.confirm(
                f"永久删除 Worktree 分支？\n名称：{snapshot.record.name}\n"
                f"分支：{snapshot.record.branch}\n{risk}\n"
                "这个确认只授权删除分支，不授权其他目录操作。"
            )
            if not confirmed:
                context.ui.show_status("已取消删除 Worktree 分支")
                return CommandResult()
            report = await manager.delete_branch(
                args[1],
                discard_commits=discard_commits,
            )
            context.ui.show_status(report.message)
            return CommandResult()
    except (ValueError, RuntimeError, WorktreeManagerError) as exc:
        context.ui.show_error(str(exc))
        return CommandResult()

    return _show_usage(context)
