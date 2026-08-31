"""配置加载、Provider 创建和 MCP 连接共用的数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath

from mycode.constants import (
    DEFAULT_COMPACTION_OUTPUT_TOKENS,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DEFAULT_TOOL_BATCH_SPILL_CHARS,
    DEFAULT_TOOL_RESULT_SPILL_CHARS,
)
from mycode.models.hooks import HookDefinition


class Protocol(str, Enum):
    """模型服务商使用的线上协议。"""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class ThinkingMode(str, Enum):
    """用户为 Anthropic 配置的思考模式。"""

    DISABLED = "false"
    ENABLED = "enabled"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True, repr=False)
class SecretValue:
    """只在显式调用 ``reveal`` 时暴露原文的敏感字符串。"""

    value: str

    # 读取密钥必须显式调用 reveal，使代码审查时所有暴露点都容易检索；
    # 普通打印、格式化和调试 repr 只会得到掩码。
    def reveal(self) -> str:
        """返回敏感值原文。

        Returns:
            仅供真正发送请求或启动进程时使用的敏感值原文。
        """
        return self.value

    def __repr__(self) -> str:
        """返回不会泄露原文的调试表示。

        Returns:
            使用固定掩码替代真实内容的字符串。
        """
        return "SecretValue('***')"

    def __str__(self) -> str:
        """返回不会泄露原文的普通字符串。

        Returns:
            固定的掩码字符串。
        """
        return "***"


@dataclass(frozen=True)
class ProviderConfig:
    """单个模型服务及其上下文预算。

    应用启动时从全部配置中选出一个实例。Provider 使用连接字段发送请求，
    ContextManager 使用四个预算字段决定何时保存工具结果和压缩历史。
    """

    name: str
    protocol: Protocol
    model: str
    base_url: str
    api_key: SecretValue
    thinking: ThinkingMode = ThinkingMode.DISABLED
    # 当前模型一次请求可接受的输入与输出总 Token 数。
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    # 摘要请求允许生成的最大 Token 数，分析草稿和正式摘要共同使用。
    compaction_output_tokens: int = DEFAULT_COMPACTION_OUTPUT_TOKENS
    # 单个工具正文超过多少 Unicode 字符时保存到工作区 artifact。
    tool_result_spill_chars: int = DEFAULT_TOOL_RESULT_SPILL_CHARS
    # 同一模型响应产生的工具正文合计超过多少字符时继续选择大结果落盘。
    tool_batch_spill_chars: int = DEFAULT_TOOL_BATCH_SPILL_CHARS


class McpTransportType(str, Enum):
    """MCP Server 支持的传输类型。"""

    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"


@dataclass(frozen=True, repr=False)
class ExpandedConfigValue:
    """保存配置展开结果及参与展开的敏感片段。"""

    # 真正传递给子进程或 HTTP 请求的完整值。
    value: str
    # 参与模板展开、需要在错误与输出中脱敏的值。
    secret_parts: tuple[SecretValue, ...] = ()

    def reveal(self) -> str:
        """返回展开后的完整配置值。

        Returns:
            仅供传输边界使用的展开后原文。
        """
        return self.value

    def __repr__(self) -> str:
        """返回不会泄露展开值的调试表示。

        Returns:
            使用固定掩码替代真实内容的字符串。
        """
        return "ExpandedConfigValue('***')"

    def __str__(self) -> str:
        """返回不会泄露展开值的普通字符串。

        Returns:
            固定的掩码字符串。
        """
        return "***"


@dataclass(frozen=True)
class StdioMcpServerConfig:
    """描述通过本地子进程连接的 MCP Server。"""

    # Server 在配置、工具命名空间和诊断中的稳定名称。
    name: str
    # 直接执行且不经过 Shell 拼接的程序名称或路径。
    command: str
    # 按原有边界传递给子进程的命令行参数。
    args: tuple[str, ...] = ()
    # 覆盖子进程继承环境的配置项。
    env: tuple[tuple[str, ExpandedConfigValue], ...] = ()
    # 固定的 stdio 传输标识。
    transport: McpTransportType = field(
        default=McpTransportType.STDIO,
        init=False,
    )


@dataclass(frozen=True)
class HttpMcpServerConfig:
    """描述通过 Streamable HTTP 连接的 MCP Server。"""

    # Server 在配置、工具命名空间和诊断中的稳定名称。
    name: str
    # Streamable HTTP MCP 端点的完整 URL。
    url: str
    # 发送给该 Server 的自定义 HTTP 请求头。
    headers: tuple[tuple[str, ExpandedConfigValue], ...] = ()
    # 固定的 Streamable HTTP 传输标识。
    transport: McpTransportType = field(
        default=McpTransportType.STREAMABLE_HTTP,
        init=False,
    )


McpServerConfig = StdioMcpServerConfig | HttpMcpServerConfig


@dataclass(frozen=True)
class AgentSettings:
    """保存独立子 Agent 的全局运行开关和后台容量。

    Attributes:
        auto_background_seconds: 前台子 Agent 运行多少秒后自动移交后台；``0``
            表示关闭自动移交。
        max_background_tasks: TaskManager 同时启动的队列 worker 数量。
        enable_verification: 是否把内置 Verification 角色加入可选择目录。
        agent_tool_timeout_seconds: 统一 Agent 工具从开始执行到返回最终文本或
            后台任务 ID 的最长秒数。
    """

    auto_background_seconds: float = 120.0
    max_background_tasks: int = 4
    enable_verification: bool = False
    agent_tool_timeout_seconds: float = 135.0

    def __post_init__(self) -> None:
        """校验四个配置值能直接用于计时器、队列和角色开关。

        Returns:
            四个字段合法且两个时间字段关系正确时不返回数据。

        Raises:
            ValueError: 时间、并发数或开关类型无效，或者 Agent 工具超时
                没有晚于已经启用的自动移交时间。
        """

        if (
            isinstance(self.auto_background_seconds, bool)
            or not isinstance(self.auto_background_seconds, (int, float))
            or self.auto_background_seconds < 0
        ):
            raise ValueError("Agent 自动移交秒数必须是非负数")
        if (
            isinstance(self.max_background_tasks, bool)
            or not isinstance(self.max_background_tasks, int)
            or self.max_background_tasks <= 0
        ):
            raise ValueError("Agent 后台并发数必须是正整数")
        if not isinstance(self.enable_verification, bool):
            raise ValueError("Verification 开关必须是布尔值")
        if (
            isinstance(self.agent_tool_timeout_seconds, bool)
            or not isinstance(
                self.agent_tool_timeout_seconds,
                (int, float),
            )
            or self.agent_tool_timeout_seconds <= 0
        ):
            raise ValueError("Agent 工具超时秒数必须是正数")
        if (
            self.auto_background_seconds > 0
            and self.agent_tool_timeout_seconds
            <= self.auto_background_seconds
        ):
            raise ValueError(
                "Agent 工具超时 agent_tool_timeout_seconds="
                f"{self.agent_tool_timeout_seconds:g} 秒必须大于自动移交时间 "
                "auto_background_seconds="
                f"{self.auto_background_seconds:g} 秒"
            )


def _validate_worktree_relative_text(value: str, field_name: str) -> None:
    """检查 Worktree 初始化配置使用项目内的斜杠相对路径。

    Args:
        value: 配置中填写的路径或 ignored 模式文本。
        field_name: 报错时显示的完整配置字段名。

    Returns:
        文本非空、使用正斜杠且没有绝对路径或父目录段时不返回数据。

    Raises:
        ValueError: 文本为空、使用反斜杠、是绝对路径，或包含 ``.``/``..``
            路径段。
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 必须是非空字符串")
    if "\\" in value:
        raise ValueError(f"{field_name} 必须使用正斜杠分隔路径")
    raw_parts = value.split("/")
    if any(not part or part in {".", ".."} for part in raw_parts):
        raise ValueError(f"{field_name} 必须是项目内相对路径")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError(f"{field_name} 必须是项目内相对路径")


@dataclass(frozen=True)
class WorktreePathRule:
    """说明一个明确路径在新 Worktree 中如何复制或软链接。

    Attributes:
        path: 相对主仓库根目录的斜杠路径，例如 ``settings.local.json`` 或
            ``node_modules``。
        required: 动作失败时是否终止 Worktree 创建；``False`` 时只记录警告。
    """

    path: str
    required: bool = False

    def __post_init__(self) -> None:
        """校验路径规则可以直接交给初始化器执行。

        Returns:
            路径是项目内相对路径且必需标志为布尔值时不返回数据。

        Raises:
            ValueError: 路径格式或 ``required`` 类型无效。
        """

        _validate_worktree_relative_text(self.path, "WorktreePathRule.path")
        if not isinstance(self.required, bool):
            raise ValueError("WorktreePathRule.required 必须是布尔值")


@dataclass(frozen=True)
class WorktreeIgnoredRule:
    """说明哪些被 Git 忽略的运行文件需要复制到新 Worktree。

    Attributes:
        pattern: 使用正斜杠的 shell 风格项目内匹配模式，例如 ``.env.*``。
            初始化器只用它匹配 Git 已确认忽略的实际路径，不把它交给 Shell。
        required: 没有匹配项或复制失败时是否终止 Worktree 创建。
    """

    pattern: str
    required: bool = False

    def __post_init__(self) -> None:
        """校验 ignored 规则的模式文本和必需标志。

        Returns:
            模式是项目内相对文本且 ``required`` 为布尔值时不返回数据。

        Raises:
            ValueError: 模式格式或 ``required`` 类型无效。
        """

        _validate_worktree_relative_text(
            self.pattern,
            "WorktreeIgnoredRule.pattern",
        )
        if not isinstance(self.required, bool):
            raise ValueError("WorktreeIgnoredRule.required 必须是布尔值")


@dataclass(frozen=True)
class WorktreeSettings:
    """保存 Worktree 初始化和过期清理所需的应用配置。

    Attributes:
        stale_after_hours: 临时目录最后使用多久后才进入清理候选，单位为小时。
        cleanup_interval_seconds: 后台清理器两次扫描之间的秒数。
        copy_files: 从主仓库复制到新 Worktree 的明确文件或目录规则。
        symlink_directories: 从新 Worktree 链接到主仓库的大型依赖目录规则。
        copy_ignored: 在 Git 已忽略文件中筛选并复制内容的模式规则。
        hooks_path: 显式写入 Worktree 专属 ``core.hooksPath`` 的相对路径；
            ``None`` 表示沿用 Git 当前配置，不写新值。
    """

    stale_after_hours: float = 24.0
    cleanup_interval_seconds: float = 1800.0
    copy_files: tuple[WorktreePathRule, ...] = ()
    symlink_directories: tuple[WorktreePathRule, ...] = ()
    copy_ignored: tuple[WorktreeIgnoredRule, ...] = ()
    hooks_path: str | None = None

    def __post_init__(self) -> None:
        """校验清理时间、初始化规则和可选 Hooks 路径。

        Returns:
            所有字段能被 Manager 和初始化器直接使用时不返回数据。

        Raises:
            ValueError: 时间不是正数、规则元组类型错误，或 Hooks 路径无效。
        """

        for field_name in ("stale_after_hours", "cleanup_interval_seconds"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise ValueError(f"WorktreeSettings.{field_name} 必须是正数")
        for field_name, item_type in (
            ("copy_files", WorktreePathRule),
            ("symlink_directories", WorktreePathRule),
            ("copy_ignored", WorktreeIgnoredRule),
        ):
            value = getattr(self, field_name)
            if not isinstance(value, tuple) or not all(
                isinstance(item, item_type) for item in value
            ):
                raise ValueError(f"WorktreeSettings.{field_name} 类型无效")
        if self.hooks_path is not None:
            _validate_worktree_relative_text(
                self.hooks_path,
                "WorktreeSettings.hooks_path",
            )


@dataclass(frozen=True)
class AppConfig:
    """包含全部 Provider 与 MCP Server 的应用配置。"""

    # 当前会话选中的 Provider 名称。
    active: str
    # 两层配置合并后的全部 Provider。
    providers: tuple[ProviderConfig, ...]
    # 两层配置合并后的全部 MCP Server。
    mcp_servers: tuple[McpServerConfig, ...] = ()
    # 三层配置按用户、项目、本地顺序追加并完成校验后的 Hook。
    hooks: tuple[HookDefinition, ...] = ()
    # 独立子 Agent 的自动移交、后台并发和 Verification 开关。
    agents: AgentSettings = field(default_factory=AgentSettings)
    # Worktree 创建初始化、会话恢复和后台过期清理使用的配置。
    worktrees: WorktreeSettings = field(default_factory=WorktreeSettings)

    @property
    def active_provider(self) -> ProviderConfig:
        """返回当前会话选中的 Provider 配置。

        Returns:
            名称与 ``active`` 一致的 Provider 配置。
        """
        # loader 已保证 active 必然命中；这里仍保留防御性检查，使手工构造
        # AppConfig 的测试或第三方调用不会悄悄选到错误 Provider。
        for provider in self.providers:
            if provider.name == self.active:
                return provider
        raise ValueError(f"当前 Provider 不可用：{self.active}")

    @property
    def secrets(self) -> tuple[SecretValue, ...]:
        """汇总应用已知且需要统一脱敏的敏感值。

        Returns:
            按首次出现顺序去重后的敏感值元组。
        """
        candidates = [provider.api_key for provider in self.providers]
        for server in self.mcp_servers:
            values = (
                server.env
                if isinstance(server, StdioMcpServerConfig)
                else server.headers
            )
            for _, expanded in values:
                candidates.extend(expanded.secret_parts)

        seen: set[str] = set()
        unique: list[SecretValue] = []
        for secret in candidates:
            raw = secret.reveal()
            if raw and raw not in seen:
                seen.add(raw)
                unique.append(secret)
        return tuple(unique)
