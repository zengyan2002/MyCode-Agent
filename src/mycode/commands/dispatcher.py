"""统一查找并执行斜杠命令，错误不会回退到普通 Agent 输入。"""

from __future__ import annotations

from mycode.commands.models import CommandContext, CommandResult, ParsedCommand
from mycode.commands.registry import CommandRegistry
from mycode.errors import MyCodeError, redact_secrets


class CommandDispatcher:
    """使用一个冻结注册表分发所有本地和提示词命令。"""

    def __init__(self, registry: CommandRegistry) -> None:
        """保存应用启动时已经冻结的命令注册表。

        Args:
            registry: 后续只用于查找和帮助展示的命令注册表。

        Returns:
            None。
        """

        # 所有正式名称和别名都由这一份注册表解析
        self._registry = registry

    async def dispatch(
        self,
        parsed: ParsedCommand,
        context: CommandContext,
    ) -> CommandResult:
        """查找并执行一条命令，统一展示查找和执行错误。

        Args:
            parsed: 解析器产生的原始短命令、规范名称和参数。
            context: Handler 本次可以使用的真实应用对象。

        Returns:
            Handler 请求退出、启动 Agent 或留在本地的执行结果。
        """

        name = parsed.name or "help"
        command = self._registry.find(name)
        if command is None:
            context.ui.show_error(
                f"未知命令：/{parsed.name}。输入 /help 查看可用命令"
            )
            return CommandResult()
        if command.arg_prompt is not None and not parsed.args:
            context.ui.show_error(command.arg_prompt)
            return CommandResult()
        try:
            return await command.handler(context)
        except MyCodeError as exc:
            context.ui.show_error(redact_secrets(str(exc), context.secrets))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            context.ui.show_error("命令执行失败，当前会话保持不变")
        return CommandResult()
