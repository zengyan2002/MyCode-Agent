"""工具注册、调度和执行共用的数据模型。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from mycode.models.json_types import JsonObject
from mycode.models.messages import ToolCall


class ToolAccess(str, Enum):
    READ = "read"
    WRITE = "write"


class ToolSource(str, Enum):
    """标识工具定义由 MyCode、Skill 还是外部 MCP 提供。"""

    # MyCode 自带的文件、命令和搜索工具
    BUILTIN = "builtin"
    # MyCode 内部控制流程使用的工具，例如 tool_search
    SYSTEM = "system"
    # 外部 MCP Server 提供的工具
    MCP = "mcp"
    # 目录型 Skill 通过 tool.json 提供的专属工具
    SKILL = "skill"


class ToolErrorCode(str, Enum):
    INVALID_ARGUMENTS = "invalid_arguments"
    UNKNOWN_TOOL = "unknown_tool"
    PATH_OUTSIDE_WORKSPACE = "path_outside_workspace"
    NOT_FOUND = "not_found"
    NOT_A_FILE = "not_a_file"
    ALREADY_EXISTS = "already_exists"
    INVALID_ENCODING = "invalid_encoding"
    INVALID_PATTERN = "invalid_pattern"
    NO_MATCH = "no_match"
    MULTIPLE_MATCHES = "multiple_matches"
    COMMAND_FAILED = "command_failed"
    IO_ERROR = "io_error"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"
    REMOTE_ERROR = "remote_error"


@dataclass(frozen=True)
class ToolDefinition:
    """保存发送给模型的工具名称、说明、参数格式和读写类别。"""

    # 工具名，也是模型返回 ToolCall 时使用的名字。
    name: str
    # 告诉模型什么时候应该使用该工具。
    description: str
    # 定义工具接收参数的 JSON Schema。
    input_schema: JsonObject
    # 本地可信的只读或写入分类。
    access: ToolAccess


@dataclass
class ToolActivationState:
    """保存单个 Agent 通过 tool_search 激活的 MCP 工具名。

    Attributes:
        active_mcp_names: 当前 Agent 后续 Provider 请求可以看到的 MCP 工具
            名。每个 Agent 都创建自己的集合，不能与父 Agent 或其他后台
            任务共享。
    """

    active_mcp_names: set[str] = field(default_factory=set)

    def activate(self, names: frozenset[str] | set[str]) -> None:
        """把一次 tool_search 命中的 MCP 工具加入当前 Agent 状态。

        Args:
            names: 已经由 ToolRegistry 确认存在的 MCP 工具名。

        Returns:
            不返回数据；集合会原地加入这些名字。
        """

        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("激活的 MCP 工具名必须是非空字符串")
        self.active_mcp_names.update(names)

    def reset(self) -> None:
        """清除当前 Agent 通过延迟搜索激活的全部 MCP 工具。

        Returns:
            不返回数据；调用后下一轮不会再因旧搜索暴露 MCP 工具。
        """

        self.active_mcp_names.clear()


@dataclass(frozen=True)
class ToolView:
    """描述某一轮 Agent 可以看见和调用哪些工具。

    SkillRuntime 先填写活动 Skill 和业务白名单，ToolRegistry 再根据当前
    MCP 激活状态解析出 visible_tool_names。Scheduler 使用这个最终名字
    集合复查模型返回的调用。
    """

    # 本轮已经激活、因此可以暴露专属工具的 Skill 名。
    active_skill_names: frozenset[str] = frozenset()
    # None 表示不限制业务工具；集合表示精确允许的业务工具名。
    business_allowlist: frozenset[str] | None = None
    # 当前 Agent 已激活的 MCP 工具名；Registry 不再读取全局可变状态。
    active_mcp_names: frozenset[str] = frozenset()
    # 作用于 BUILTIN、SYSTEM、MCP、SKILL 全部来源的最终白名单。
    final_allowlist: frozenset[str] | None = None
    # 作用于全部来源、且任何下层都不能恢复的最终禁止工具名。
    denied_tool_names: frozenset[str] = frozenset()
    # Provider 实际收到的工具名；解析 ToolView 后才会有值。
    visible_tool_names: frozenset[str] = frozenset()

    def resolved(
        self,
        visible_tool_names: frozenset[str],
    ) -> "ToolView":
        """复制当前条件，并写入 Provider 实际收到的工具名。

        Args:
            visible_tool_names: ToolRegistry 过滤后准备发送给模型的名字。

        Returns:
            带执行校验快照的新 ToolView。
        """

        return ToolView(
            active_skill_names=self.active_skill_names,
            business_allowlist=self.business_allowlist,
            active_mcp_names=self.active_mcp_names,
            final_allowlist=self.final_allowlist,
            denied_tool_names=self.denied_tool_names,
            visible_tool_names=visible_tool_names,
        )


@dataclass(frozen=True)
class ToolExecutionPolicy:
    """保存执行器和权限层需要知道的工具运行信息。

    ToolRegistry 为每个工具创建一项。普通工具通常只有 source；Skill
    专属工具还会记录所属 Skill、真实命令、超时和来源路径。
    """

    # 工具由内置代码、系统流程、MCP 还是 Skill 提供。
    source: ToolSource
    # Skill 专属工具所属的 Skill 名；其他来源为 None。
    skill_name: str | None = None
    # project、user 或 builtin；其他来源为 None。
    skill_origin: str | None = None
    # 外部 Skill 首次信任提示显示的入口路径。
    source_path: Path | None = None
    # 专属工具实际启动的命令数组。
    command: tuple[str, ...] = ()
    # 当前工具自己的超时；None 表示使用 Executor 全局超时。系统工具和
    # Skill 专属工具都可以在注册时提供该值。
    timeout_seconds: float | None = None
    # Skill stdout 上限；非 Skill 工具为 None。
    max_output_bytes: int | None = None

# ToolCall 是模型原始意图；ToolInvocation 在进入本地调度后补充可信的访问
# 分类、模型请求序号和调用顺序。不要直接相信模型自行声明读写属性。
@dataclass(frozen=True)
class ToolInvocation:
    call: ToolCall
    # 访问分类来自本地注册表，调度屏障和 Plan 模式都依赖它。
    access: ToolAccess
    # 产生该工具调用的 Agent 模型请求序号，从 1 开始。
    model_call_number: int
    # 在本次模型响应中的零基索引，用于并发完成后恢复原顺序。
    call_index: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.model_call_number, bool)
            or not isinstance(self.model_call_number, int)
            or self.model_call_number <= 0
        ):
            raise ValueError("工具调用的模型请求序号必须为正数")
        if (
            isinstance(self.call_index, bool)
            or not isinstance(self.call_index, int)
            or self.call_index < 0
        ):
            raise ValueError("工具调用索引不能为负数")

#Python 中把字典、列表等数据转成 JSON 格式字符串的函数
def _compact_json(payload: JsonObject) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


"""
工具执行结束
    ↓
生成 ToolExecutionResult
    ↓
检查字段组合是否合法
    ↓
转换成 JSON
    ↓
发送给模型
"""
@dataclass(frozen=True)
class ToolExecutionResult:
    tool_call_id: str
    tool_name: str
    success: bool
    content: str
    error_code: ToolErrorCode | None
    error_message: str | None
    timed_out: bool
    # truncated 表示 content/metadata 因预算缩减，不能等同于执行失败。
    truncated: bool
    # 截断前的 UTF-8 字节数，未截断时通常等于正文大小。
    original_size_bytes: int
    # 从进入执行器到形成结果的单调时钟耗时。
    duration_ms: int
    # 只保存退出码和计数等小型 JSON 数据，仍受完整结果上限保护。
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.success and (
            self.error_code is not None or self.error_message is not None
        ):
            raise ValueError("成功的工具结果不能包含错误信息")
        if not self.success and (
            self.error_code is None or not self.error_message
        ):
            raise ValueError("失败的工具结果必须包含错误码和错误消息")
        if self.duration_ms < 0:
            raise ValueError("工具结果耗时不能为负数")
        if self.original_size_bytes < 0:
            raise ValueError("工具结果大小不能为负数")

    # 先构造协议中立 JSON 对象；OpenAI/Anthropic Provider 再决定如何把
    # 序列化字符串包装成各自的 tool result 消息。
    def _payload(self) -> JsonObject:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "success": self.success,
            "content": self.content,
            "error_code": (
                self.error_code.value if self.error_code is not None else None
            ),
            "error_message": self.error_message,
            "timed_out": self.timed_out,
            "truncated": self.truncated,
            "original_size_bytes": self.original_size_bytes,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    def to_model_json(self) -> str:
        """把工具结果序列化为发送给模型的 JSON。

        过长的正文在调用本函数前已经由上下文管理器保存到 artifact，
        这里仅把最终要写入对话的结果转换成紧凑 JSON。
        """

        return _compact_json(self._payload())
