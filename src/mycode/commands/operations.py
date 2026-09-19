"""提供本地工具操作查询、原步骤重试和人工核查命令。"""

from __future__ import annotations

import json
from dataclasses import asdict

from mycode.commands.models import CommandContext, CommandResult
from mycode.errors import MyCodeError, redact_secrets
from mycode.models.operations import ResolutionVerdict


OPERATIONS_USAGE = (
    "/operations list [--all] | show <ID> | retry <ID> | "
    "resolve <ID> completed|not-applied <核查说明>"
)


async def handle_operations(context: CommandContext) -> CommandResult:
    """只接受本地用户的核查判断，模型没有调用这个命令的工具入口。"""
    args = context.invocation.args.split(maxsplit=3)
    try:
        if args in ([], ["list"], ["list", "--all"]):
            records = await context.agent.list_operations(all_sessions=args == ["list", "--all"])
            text = "\n".join(f"{r.operation_id}  {r.state.value}  {r.call.name}  {r.scope.runtime_id}"
                             for r in records) or "没有工具执行记录"
        elif len(args) == 2 and args[0] == "show":
            record, notes = await context.agent.inspect_operation(args[1])
            text = json.dumps({"operation": asdict(record), "resolutions": notes},
                              ensure_ascii=False, indent=2, default=str)
        elif len(args) == 2 and args[0] == "retry":
            result = await context.agent.retry_operation(args[1], context.cancellation,
                                                        plan_only=context.runtime_state.plan_only)
            text = result.to_model_json()
        elif len(args) == 4 and args[0] == "resolve":
            # split(maxsplit=3) 将判断放在 args[2]，最后一项完整保留用户说明。
            verdict = ResolutionVerdict(args[2])
            record = await context.agent.resolve_operation(args[1], verdict, args[3])
            text = f"已保存用户核查：{record.operation_id}，状态：{record.state.value}。本次未执行工具。"
        else:
            raise ValueError(OPERATIONS_USAGE)
        context.ui.show_status(redact_secrets(text, context.secrets))
    except (MyCodeError, ValueError) as exc:
        context.ui.show_error(redact_secrets(f"{exc}\n用法：{OPERATIONS_USAGE}", context.secrets))
    return CommandResult()
