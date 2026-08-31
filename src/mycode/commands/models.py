"""命令定义、单次调用参数和执行结果。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from mycode.agents.service import AgentService
    from mycode.agent.cancellation import CancellationToken
    from mycode.agent.loop import AgentLoop
    from mycode.app.terminal_ui import TerminalUI
    from mycode.commands.registry import CommandRegistry
    from mycode.memory.store import MemoryStore
    from mycode.models.config import ProviderConfig, SecretValue
    from mycode.models.permissions import (
        LoadedPermissionSettings,
    )
    from mycode.permissions.policy import PermissionController
    from mycode.persistence.sessions import SessionManager
    from mycode.skills.service import SkillService
    from mycode.worktrees.manager import WorktreeManager


class CommandType(str, Enum):
    """说明一条命令由本地、界面状态还是 Agent 通道执行。"""

    LOCAL = "local"
    LOCAL_UI = "local_ui"
    PROMPT = "prompt"


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    """保存用户本次输入中解析出的命令名和未解释参数。"""

    # 用户实际提交的完整命令，用于在对话界面显示短命令。
    raw_input: str
    # 已执行 casefold 的命令名，不包含开头的斜杠。
    name: str
    # 命令名之后的原始参数，保留大小写和内部空白。
    args: str


@dataclass(slots=True)
class CommandRuntimeState:
    """保存应用循环和模式命令共同使用的当前 Plan 开关。"""

    # True 表示后续 Agent 请求只能规划；False 表示可以执行写操作。
    plan_only: bool = False


@dataclass(frozen=True, slots=True)
class AgentSubmission:
    """描述命令要显示给用户并实际发送给 Agent 的两段文本。"""

    # 对话界面显示的用户原始短命令。
    display_text: str
    # AgentLoop 接收并写入当前会话的完整用户消息。
    prompt: str
    # 本次 Agent 请求是否以只规划模式运行。
    plan_only: bool


@dataclass(frozen=True, slots=True)
class InlineSkillSubmission:
    """告诉应用在主 Agent Loop 中执行一个 inline Skill。"""

    # Catalog 中的目标 Skill 名。
    name: str
    # Skill 名之后的原始参数，用来替换 SOP 中的 $ARGUMENTS。
    arguments: str
    # UI 和主会话历史中保留的简短斜杠命令。
    display_text: str


@dataclass(frozen=True, slots=True)
class ForkSkillSubmission:
    """告诉应用在临时 Agent 对话中执行一个 fork Skill。"""

    # Catalog 中的目标 Skill 名。
    name: str
    # Skill 名之后的原始参数，用来替换 SOP 中的 $ARGUMENTS。
    arguments: str
    # fork 完成后写入主会话的简短用户命令。
    display_text: str


SkillSubmission: TypeAlias = InlineSkillSubmission | ForkSkillSubmission


@dataclass(frozen=True, slots=True)
class CommandResult:
    """告诉应用命令结束后要退出、启动 Agent，还是执行 Skill。"""

    # True 时，应用在当前命令结束后正常关闭。
    exit_requested: bool = False
    # 有值时，分发器继续把完整提示词交给 Agent；本地命令为 None。
    agent_submission: AgentSubmission | None = None
    # 有值时，应用把简短调用交给 SkillService；普通命令为 None。
    skill_submission: SkillSubmission | None = None

    def __post_init__(self) -> None:
        """拒绝同时请求退出和启动 Agent 的矛盾结果。

        Args:
            self: 刚创建的命令结果。

        Returns:
            无返回值；结果合法时保持原值，矛盾时抛出 ``ValueError``。
        """

        action_count = sum(
            (
                self.exit_requested,
                self.agent_submission is not None,
                self.skill_submission is not None,
            )
        )
        if action_count > 1:
            raise ValueError("命令只能请求退出、启动 Agent 或执行 Skill 之一")


@dataclass(frozen=True, slots=True)
class CommandContext:
    """保存一次命令执行会实际读取或修改的现有应用对象。

    Attributes:
        invocation: 已解析的命令名、参数和原始输入。
        registry: 帮助、查询和补全共用的命令注册表。
        agent: 处理主对话和压缩请求的 AgentLoop。
        ui: 显示状态、错误和确认请求的终端界面。
        session_manager: 保存并切换主会话的管理器。
        memory_store: `/memory` 读取的长期记忆存储。
        permission_controller: 当前会话的权限模式和临时规则。
        permission_settings: 启动时加载的用户级与项目级权限规则。
        provider_config: `/status` 展示的 Provider 和模型配置。
        runtime_state: `/plan` 与 `/do` 共用的计划模式开关。
        cancellation: 当前命令可用于停止耗时工作的取消令牌。
        secrets: 错误展示前需要替换的密钥原文。
        skill_service: 可选的 Skill 查询和热重载服务。
        agent_service: 可选的 Agent 角色查询和热重载服务。
        worktree_manager: 可选的受管 Worktree 生命周期和状态查询服务。
    """

    # 本次命令的原始文本、规范化名称和未解释参数。
    invocation: ParsedCommand
    # 当前进程内已经冻结的命令注册表，供帮助命令读取。
    registry: CommandRegistry
    # 当前 Agent 实例，供压缩、会话恢复和提示词命令使用。
    agent: AgentLoop
    # 当前实际终端界面，用于展示消息、状态和确认请求。
    ui: TerminalUI
    # 当前项目的会话管理器，供会话查询和删除使用。
    session_manager: SessionManager
    # 用户级和项目级记忆存储，供只读记忆命令使用。
    memory_store: MemoryStore
    # 当前会话权限状态，供模式查询和切换使用。
    permission_controller: PermissionController
    # 启动时加载的用户级和项目级权限规则快照。
    permission_settings: LoadedPermissionSettings
    # 当前 Provider 和模型配置，供状态命令展示上下文上限。
    provider_config: ProviderConfig
    # 应用和模式命令共享的 Plan 开关。
    runtime_state: CommandRuntimeState
    # 当前命令的取消信号，传给压缩、恢复或 Agent 请求。
    cancellation: CancellationToken
    # 错误展示前需要替换的已知密钥。
    secrets: tuple[SecretValue, ...]
    # Skill 管理命令和动态 Skill handler 使用的协调对象；未启用 Skill
    # 系统的兼容调用方可以不传。
    skill_service: SkillService | None = None
    # /agent list、info 和 reload 使用的角色服务；未装配时为 None。
    agent_service: AgentService | None = None
    # /worktree 与 /status 读取的受管目录 Manager；未装配时为 None。
    worktree_manager: WorktreeManager | None = None


CommandHandler: TypeAlias = Callable[
    [CommandContext], Awaitable[CommandResult]
]


@dataclass(frozen=True, slots=True)
class Command:
    """登记一条命令的用户可见说明和实际执行入口。"""

    # 不含斜杠的正式名称。
    name: str
    # 不含斜杠的可选短名称。
    aliases: tuple[str, ...]
    # `/help` 列表中显示的一行说明。
    description: str
    # 参数错误或详情帮助中显示的完整用法。
    usage: str
    # 命令采用的执行通道。
    type: CommandType
    # 分发器命中后调用的异步函数。
    handler: CommandHandler
    # 整条命令缺少必需参数时显示的提示；无必需参数时为 None。
    arg_prompt: str | None = None
    # True 时仍可执行，但不出现在帮助和补全中。
    hidden: bool = False
    # True 表示该命令由 Skill Catalog 动态生成，/help 会显示来源标记。
    skill: bool = False
