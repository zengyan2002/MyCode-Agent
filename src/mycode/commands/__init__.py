"""斜杠命令的登记、解析、补全和执行入口。"""

from mycode.commands.models import (
    AgentSubmission,
    Command,
    CommandContext,
    CommandHandler,
    CommandResult,
    CommandRuntimeState,
    CommandType,
    ParsedCommand,
)
from mycode.commands.completion import CommandCompleter
from mycode.commands.builtins import create_builtin_registry
from mycode.commands.dispatcher import CommandDispatcher
from mycode.commands.parser import parse_command
from mycode.commands.registry import CommandRegistry

__all__ = [
    "CommandCompleter",
    "create_builtin_registry",
    "CommandDispatcher",
    "AgentSubmission",
    "Command",
    "CommandContext",
    "CommandHandler",
    "CommandRegistry",
    "CommandResult",
    "CommandRuntimeState",
    "CommandType",
    "ParsedCommand",
    "parse_command",
]
