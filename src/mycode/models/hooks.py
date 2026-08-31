"""Hook 配置和一次事件执行时共享的数据。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from re import Pattern


class HookEvent(str, Enum):
    """MyCode 会在这些确定的生命周期位置查找用户配置的 Hook。"""

    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    PRE_SEND = "pre_send"
    POST_RECEIVE = "post_receive"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    ERROR = "error"
    COMPACT = "compact"


class HookLayer(str, Enum):
    """一条 Hook 来自哪个配置层；枚举顺序不承担优先级判断。"""

    USER = "user"
    PROJECT = "project"
    LOCAL = "local"


class HookOperator(str, Enum):
    """一个条件可使用的四种字符串比较方式。"""

    EQUALS = "=="
    NOT_EQUALS = "!="
    REGEX = "=~"
    GLOB = "~="


class HookConditionMode(str, Enum):
    """多个原子条件是全部通过还是任意一个通过。"""

    ALL = "all"
    ANY = "any"


@dataclass(frozen=True)
class HookContext:
    """保存一次真实事件能提供给条件和动作的数据。

    `HookEngine` 在每个生命周期接入点接收该对象。工具字段只在工具
    事件中有值，消息和错误字段也只在对应事件中有值。

    Attributes:
        event: 当前正在派发的生命周期事件。
        tool_name: 模型本次调用的工具名；非工具事件为 None。
        tool_args: 模型传给工具的完整参数；非工具事件为 None。
        file_path: 文件工具参数中提取出的路径；没有路径时为 None。
        message: 本次请求、响应或工具结果的可读摘要。
        error: Agent 主流程准备对外报告的脱敏错误说明。
    """

    event: HookEvent
    tool_name: str | None = None
    tool_args: Mapping[str, object] | None = None
    file_path: str | None = None
    message: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class HookCondition:
    """保存启动阶段已经解析好的一个字段比较。

    Attributes:
        field: 从 `HookContext` 读取的字段名，例如 `tool` 或 `args.path`。
        operator: 运行时采用的比较方式。
        expected: YAML 条件中写下的目标字符串。
        compiled_regex: 正则条件预编译的匹配器；其他操作符为 None。
    """

    field: str
    operator: HookOperator
    expected: str
    compiled_regex: Pattern[str] | None = None


@dataclass(frozen=True)
class HookConditionGroup:
    """保存一条 Hook 的全部原子条件及其组合方式。

    Attributes:
        mode: 全部条件必须匹配，或任意一个条件匹配。
        conditions: 按配置声明顺序保存的原子条件。
    """

    mode: HookConditionMode
    conditions: tuple[HookCondition, ...]


@dataclass(frozen=True)
class HookTemplate:
    """保存动作文本及启动阶段识别出的变量名。

    Attributes:
        text: 用户在 YAML 中写下的原始文本。
        variables: 按出现顺序保存的合法变量，执行动作前逐个展开。
    """

    text: str
    variables: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommandHookAction:
    """描述需要交给当前平台 Shell 执行的一条命令。

    Attributes:
        command: 执行前需要展开上下文变量的 Shell 命令模板。
        timeout_seconds: 最长执行秒数；None 表示不单独设置动作超时。
    """

    command: HookTemplate
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class PromptHookAction:
    """描述需要放进当前 Agent 下一次模型请求的一次性提醒。

    Attributes:
        message: 展开后写入当前 scope 提示词管理器的消息模板。
    """

    message: HookTemplate


@dataclass(frozen=True)
class HttpHookAction:
    """描述一次 Hook 通知要发送的 HTTP 请求。

    Attributes:
        url: 展开后真正请求的地址。
        method: 已规范成大写的 HTTP 方法。
        headers: 按 YAML 声明顺序保存的请求头模板。
        body: 可选请求正文模板。
    """

    url: HookTemplate
    method: str
    headers: tuple[tuple[str, HookTemplate], ...] = ()
    body: HookTemplate | None = None


@dataclass(frozen=True)
class AgentHookAction:
    """保存未来交给子 Agent 的提示；当前版本触发时只记占位日志。

    Attributes:
        prompt: 未来对接子 Agent 时使用的提示模板；当前版本只校验并展开。
    """

    prompt: HookTemplate


HookAction = (
    CommandHookAction
    | PromptHookAction
    | HttpHookAction
    | AgentHookAction
)


@dataclass(frozen=True)
class HookSource:
    """指出一条 Hook 在哪份配置的哪个位置声明。

    Attributes:
        layer: 用户、项目或本地配置层。
        path: 实际读取的 YAML 文件路径。
        index: Hook 在该文件 `hooks` 列表中的零基位置。
        hook_id: 用户给出的 id；省略时使用可读的自动编号。
    """

    layer: HookLayer
    path: Path
    index: int
    hook_id: str

    @property
    def label(self) -> str:
        """返回日志和配置错误共用的规则位置说明。

        Returns:
            同时包含配置文件路径和 Hook id 的可读文本。
        """

        return f"{self.path} 的 Hook {self.hook_id}"


@dataclass(frozen=True)
class HookDefinition:
    """保存一条经过完整校验、可直接交给运行时执行的 Hook。

    Attributes:
        source: 规则所在配置层、文件和声明位置。
        event: 规则等待的生命周期事件。
        condition: 可选条件组；None 表示事件发生就执行。
        action: 命中后执行的 command、prompt、http 或 agent 动作。
        once: 当前 scope 成功一次后是否跳过后续触发。
        async_mode: 是否把动作放到后台执行而不等待结果。
        reject: 命中时是否拒绝 pre_tool_use 对应的工具调用。
    """

    source: HookSource
    event: HookEvent
    condition: HookConditionGroup | None
    action: HookAction
    once: bool = False
    async_mode: bool = False
    reject: bool = False


@dataclass(frozen=True)
class HookActionResult:
    """告诉 Hook 引擎一次动作是否成功以及可用的有限输出。

    Attributes:
        success: 动作是否按自身协议成功完成。
        output: 已限制长度的 stdout、响应正文或注入提示文本。
        error: 失败的短原因；不保存命令、请求头或原始异常正文。
    """

    success: bool
    output: str = ""
    error: str | None = None


@dataclass(frozen=True)
class HookDispatchResult:
    """告诉工具适配器本次 `pre_tool_use` 是否拒绝了调用。

    Attributes:
        rejected: True 表示工具调度器不得继续执行本次调用。
        rejection_reason: 回灌模型的拒绝原因；未拒绝时为 None。
    """

    rejected: bool = False
    rejection_reason: str | None = None
