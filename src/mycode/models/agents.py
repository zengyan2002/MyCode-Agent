"""独立子 Agent 的角色定义、委派请求、运行结果和后台任务模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from mycode.models.messages import ChatMessage
from mycode.models.permissions import PermissionMode
from mycode.models.prompts import PromptContext, RuntimeInstruction
from mycode.models.tools import ToolView
from mycode.models.teams import TeamActorContext
from mycode.models.skills import SkillDefinition
from mycode.models.worktrees import (
    WorkspaceAssignment,
    WorkspaceIsolationMode,
    WorktreeFinishReport,
)

if TYPE_CHECKING:
    from mycode.agent.cancellation import CancellationToken


class AgentSource(str, Enum):
    """说明一份角色定义来自哪个配置层。"""

    PROJECT = "project"
    USER = "user"
    BUILTIN = "builtin"


class AgentPermissionMode(str, Enum):
    """保存角色文件声明的权限模式，``inherit`` 会在创建子 Agent 时解析。"""

    INHERIT = "inherit"
    STRICT = "strict"
    DEFAULT = "default"
    ALLOW = "allow"


class AgentDiagnosticLevel(str, Enum):
    """表示角色加载诊断需要以提示还是错误展示。"""

    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class AgentDefinition:
    """一份已经通过格式校验、可以用来创建子 Agent 的角色定义。

    Attributes:
        name: 展示给模型和用户的正式角色名。
        description: 告诉主 Agent 这个角色适合处理什么任务。
        tools: 角色工具白名单；``None`` 表示角色层不额外收窄，空集合表示
            该角色不能使用任何工具。
        disallowed_tools: 在白名单结果上继续移除的工具名。
        model: 角色指定的 Provider 模型名；``None`` 表示继承父 Agent 模型。
        max_model_calls: 子 Agent 最多向 Provider 发出的模型请求数；``None``
            表示使用系统默认值。
        permission_mode: 角色声明的权限模式，包括运行时才解析的 ``inherit``。
        default_background: 未在调用参数中指定时，该角色是否默认在后台执行。
        source: 定义来自项目、用户还是内置资源。
        entry_path: 实际读取的 Markdown 文件绝对路径。
        prompt_body: 去掉 YAML frontmatter 后、伴随子 Agent 全生命周期的系统指令。
        revision: 根据完整 Markdown 内容计算的 SHA-256 十六进制摘要。
        isolation: 定义式子 Agent 默认使用独立 Worktree 还是共享调用方目录。
    """

    name: str
    description: str
    tools: frozenset[str] | None
    disallowed_tools: frozenset[str]
    model: str | None
    max_model_calls: int | None
    permission_mode: AgentPermissionMode
    default_background: bool
    source: AgentSource
    entry_path: Path
    prompt_body: str
    revision: str
    isolation: WorkspaceIsolationMode = WorkspaceIsolationMode.WORKTREE

    def __post_init__(self) -> None:
        """拒绝不能安全进入角色目录的空白字段和非法限制。

        Returns:
            校验通过时不返回数据，冻结实例保持原值。

        Raises:
            ValueError: 任一文本字段为空、集合类型错误、模型调用上限非正、背景值
                不是布尔值，或枚举字段不是对应枚举成员。
        """

        for field_name, value in (
            ("name", self.name),
            ("description", self.description),
            ("prompt_body", self.prompt_body),
            ("revision", self.revision),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Agent {field_name} 必须是非空字符串")
        if self.tools is not None and not isinstance(self.tools, frozenset):
            raise ValueError("Agent tools 必须是 frozenset 或 None")
        if not isinstance(self.disallowed_tools, frozenset):
            raise ValueError("Agent disallowed_tools 必须是 frozenset")
        tool_sets = (self.disallowed_tools,) if self.tools is None else (
            self.tools,
            self.disallowed_tools,
        )
        if any(
            not isinstance(name, str) or not name.strip()
            for names in tool_sets
            for name in names
        ):
            raise ValueError("Agent 工具名必须是非空字符串")
        if self.model is not None and (
            not isinstance(self.model, str) or not self.model.strip()
        ):
            raise ValueError("Agent model 必须是非空字符串或 None")
        if self.max_model_calls is not None and (
            isinstance(self.max_model_calls, bool)
            or not isinstance(self.max_model_calls, int)
            or self.max_model_calls <= 0
        ):
            raise ValueError("Agent max_model_calls 必须是正整数或 None")
        if not isinstance(self.permission_mode, AgentPermissionMode):
            raise ValueError("Agent permission_mode 无效")
        if not isinstance(self.default_background, bool):
            raise ValueError("Agent default_background 必须是布尔值")
        if not isinstance(self.source, AgentSource):
            raise ValueError("Agent source 无效")
        if not isinstance(self.isolation, WorkspaceIsolationMode):
            raise ValueError("Agent isolation 无效")
        if not isinstance(self.entry_path, Path) or not self.entry_path.is_absolute():
            raise ValueError("Agent entry_path 必须是绝对路径")

    @property
    def key(self) -> str:
        """返回目录查找使用的大小写无关名称。

        Returns:
            对正式角色名执行 ``casefold`` 后的字符串。
        """

        return self.name.casefold()


@dataclass(frozen=True)
class AgentDiagnostic:
    """记录一份角色候选为何没有生效。

    Attributes:
        path: 产生问题的 Markdown 文件绝对路径。
        agent_name: 能从文件位置或 YAML 中识别出的角色名；无法识别时为 ``None``。
        level: 用户界面展示该问题时使用的严重级别。
        message: 不含 traceback、可以直接展示给用户的具体原因。
    """

    path: Path
    agent_name: str | None
    level: AgentDiagnosticLevel
    message: str

    def __post_init__(self) -> None:
        """校验诊断路径、角色名、级别和可展示原因。

        Returns:
            字段可用于用户诊断时不返回数据。

        Raises:
            ValueError: 路径不是绝对路径，或名称、级别、原因无效。
        """

        if not self.path.is_absolute():
            raise ValueError("Agent 诊断路径必须是绝对路径")
        if self.agent_name is not None and not self.agent_name.strip():
            raise ValueError("Agent 诊断名称不能是空字符串")
        if not isinstance(self.level, AgentDiagnosticLevel):
            raise ValueError("Agent 诊断级别无效")
        if not self.message.strip():
            raise ValueError("Agent 诊断必须包含原因")


@dataclass(frozen=True)
class AgentCandidate:
    """保存 Loader 扫描到的一份有效定义或一条失败诊断。

    Attributes:
        source: 候选所在的配置层。
        entry_path: 被扫描的 Markdown 文件绝对路径。
        definition: 解析成功时得到的角色定义，失败时为 ``None``。
        diagnostic: 解析失败或同层冲突时的原因，成功时为 ``None``。
    """

    source: AgentSource
    entry_path: Path
    definition: AgentDefinition | None
    diagnostic: AgentDiagnostic | None

    def __post_init__(self) -> None:
        """保证候选只携带成功定义或失败诊断中的一种。

        Returns:
            字段组合合法时不返回数据。

        Raises:
            ValueError: 来源、路径无效，或定义与诊断同时存在/同时缺失。
        """

        if not isinstance(self.source, AgentSource):
            raise ValueError("Agent 候选来源无效")
        if not self.entry_path.is_absolute():
            raise ValueError("Agent 候选路径必须是绝对路径")
        if (self.definition is None) == (self.diagnostic is None):
            raise ValueError("Agent 候选必须且只能包含定义或诊断之一")


@dataclass(frozen=True)
class AgentCatalogSnapshot:
    """一次完整角色扫描的不可变结果。

    Attributes:
        definitions: 以规范化角色名为键的当前有效定义。
        candidates: 每个规范化角色名对应的全部候选，顺序就是覆盖优先级。
        diagnostics: 本次扫描产生的全部问题，供 reload 和管理命令展示。
    """

    definitions: Mapping[str, AgentDefinition]
    candidates: Mapping[str, tuple[AgentCandidate, ...]]
    diagnostics: tuple[AgentDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        """规范化并冻结角色定义与候选映射。

        Returns:
            不返回数据；校验通过后两个映射会替换为只读视图。

        Raises:
            ValueError: 调用方传入的映射键尚未做大小写规范化。
        """

        normalized_definitions = {
            key.casefold(): value for key, value in self.definitions.items()
        }
        normalized_candidates = {
            key.casefold(): tuple(value) for key, value in self.candidates.items()
        }
        if any(key != key.casefold() for key in self.definitions):
            raise ValueError("Agent 定义目录的键必须已经规范化")
        if any(key != key.casefold() for key in self.candidates):
            raise ValueError("Agent 候选目录的键必须已经规范化")
        object.__setattr__(
            self,
            "definitions",
            MappingProxyType(normalized_definitions),
        )
        object.__setattr__(
            self,
            "candidates",
            MappingProxyType(normalized_candidates),
        )


@dataclass(frozen=True)
class AgentToolRequest:
    """保存模型调用统一 ``Agent`` 工具时提交的原始参数。

    Attributes:
        prompt: 子 Agent 必须完成的完整任务说明。
        description: 主 Agent 给这次委派写的简短用途说明。
        name: 后台列表中显示的任务名；未填写时由协调层生成。
        subagent_type: 定义式角色名；未填写表示走 Fork 路径。
        model: 本次定义式调用覆盖的模型名；未填写时按角色或父模型决定。
        run_in_background: 调用方明确指定的前后台方式；``None`` 表示使用
            角色默认值，Fork 路径最终仍会强制后台。
        team_name: 创建长期成员时指定的当前团队名称；普通委派为 None。
        backend: 成员后端偏好；只有 team_name 非空时可以填写。
        plan_mode_required: 成员修改前是否需要 Lead 审批计划；普通委派为 None。
    """

    prompt: str
    description: str
    name: str | None = None
    subagent_type: str | None = None
    model: str | None = None
    run_in_background: bool | None = None
    team_name: str | None = None
    backend: str | None = None
    plan_mode_required: bool | None = None

    def __post_init__(self) -> None:
        """校验模型提交的任务文字和可选委派参数。

        Returns:
            所有字段符合 Agent 工具 Schema 时不返回数据。

        Raises:
            ValueError: 必填文字为空、可选文字是空字符串，或后台标志
                不是布尔值。
        """

        for field_name, value in (
            ("prompt", self.prompt),
            ("description", self.description),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Agent 工具的 {field_name} 必须是非空字符串")
        for field_name, value in (
            ("name", self.name),
            ("subagent_type", self.subagent_type),
            ("model", self.model),
            ("team_name", self.team_name),
            ("backend", self.backend),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"Agent 工具的 {field_name} 不能是空字符串")
        if self.run_in_background is not None and not isinstance(
            self.run_in_background,
            bool,
        ):
            raise ValueError("Agent 工具的 run_in_background 必须是布尔值")
        if self.plan_mode_required is not None and not isinstance(
            self.plan_mode_required, bool
        ):
            raise ValueError("Agent 工具的 plan_mode_required 必须是布尔值")
        if self.team_name is None:
            if self.backend is not None or self.plan_mode_required is not None:
                raise ValueError("backend 和 plan_mode_required 只能用于团队成员")
        else:
            if self.name is None or self.subagent_type is None:
                raise ValueError("创建团队成员必须填写 name 和 subagent_type")
            if self.run_in_background is not None:
                raise ValueError("团队成员是长期运行实例，不能填写 run_in_background")
            if self.backend not in {None, "auto", "tmux", "iterm2", "in-process"}:
                raise ValueError("团队成员 backend 取值无效")


class IndependentAgentOrigin(str, Enum):
    """说明独立运行实例由哪条产品路径创建。"""

    DEFINITION = "definition"
    FORK = "fork"
    SKILL_FORK = "skill_fork"


@dataclass(frozen=True)
class IndependentAgentSpec:
    """后台排队和运行时装配共同使用的一次性冻结输入。

    Attributes:
        run_id: 本次独立运行的唯一标识。
        session_id: 发起委派的主会话标识，用于查询和会话清理。
        name: 后台列表和诊断中显示的运行名称。
        description: 说明这次运行为什么被创建。
        origin: 定义式、普通 Fork 或 Skill Fork。
        task_prompt: 交给子 Agent 的任务正文。
        initial_messages: 新对话启动前需要复制的消息；定义式通常为空。
        prompt: 首次 Provider 请求使用的稳定提示和继承运行时指令。
        inherited_runtime: 从父请求冻结复制的运行时指令。
        initial_tool_names: Fork 可继承的父工具名；``None`` 表示不做父集合交集。
        role: 定义式使用的角色；Fork 和 Skill Fork 通常为 ``None``。
        model_override: 本次 Provider 请求使用的临时模型名。
        max_model_calls: 本次独立运行最多向 Provider 发出的模型请求数。
        permission_mode: 创建时已经解析完成的具体权限模式。
        background: 运行是否处于非交互后台模式。
        tool_view: 多层工具策略计算出的全来源白名单和禁止集合。
        skill: Skill fork 启动时需要激活的 Skill 定义；其他来源为 ``None``。
        skill_arguments: 替换 fork Skill 正文中 ``$ARGUMENTS`` 的原始参数。
        workspace: 准备完成后冻结给这次运行的目录、分支和租约；排队准备前
            允许为 ``None``。
        team_actor: 长期团队成员本轮使用的可信身份；普通 SubAgent 为 None。
    """

    run_id: str
    session_id: str
    name: str
    description: str
    origin: IndependentAgentOrigin
    task_prompt: str
    initial_messages: tuple[ChatMessage, ...]
    prompt: PromptContext
    inherited_runtime: tuple[RuntimeInstruction, ...]
    initial_tool_names: frozenset[str] | None
    role: AgentDefinition | None
    model_override: str | None
    max_model_calls: int
    permission_mode: PermissionMode
    background: bool
    tool_view: ToolView = field(default_factory=ToolView)
    skill: SkillDefinition | None = None
    skill_arguments: str = ""
    workspace: WorkspaceAssignment | None = None
    team_actor: TeamActorContext | None = None

    def __post_init__(self) -> None:
        """校验排队和运行装配依赖的全部冻结输入。

        Returns:
            来源、权限、模型调用上限、提示和 Skill 字段一致时不返回数据。

        Raises:
            ValueError: 标识或任务为空、字段类型错误、模型调用上限非正，或只有
                Skill fork 才能携带的 SkillDefinition 与来源不匹配。
        """

        for field_name, value in (
            ("run_id", self.run_id),
            ("session_id", self.session_id),
            ("name", self.name),
            ("description", self.description),
            ("task_prompt", self.task_prompt),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"独立 Agent 的 {field_name} 必须是非空字符串")
        if not isinstance(self.origin, IndependentAgentOrigin):
            raise ValueError("独立 Agent origin 无效")
        if not isinstance(self.initial_messages, tuple):
            raise ValueError("独立 Agent initial_messages 必须是元组")
        if not isinstance(self.prompt, PromptContext):
            raise ValueError("独立 Agent prompt 必须是 PromptContext")
        if not isinstance(self.inherited_runtime, tuple) or not all(
            isinstance(item, RuntimeInstruction) for item in self.inherited_runtime
        ):
            raise ValueError("独立 Agent inherited_runtime 必须是运行时指令元组")
        if self.initial_tool_names is not None and not isinstance(
            self.initial_tool_names,
            frozenset,
        ):
            raise ValueError("独立 Agent initial_tool_names 必须是 frozenset 或 None")
        if self.model_override is not None and not self.model_override.strip():
            raise ValueError("独立 Agent model_override 不能是空字符串")
        if (
            isinstance(self.max_model_calls, bool)
            or not isinstance(self.max_model_calls, int)
            or self.max_model_calls <= 0
        ):
            raise ValueError("独立 Agent max_model_calls 必须是正整数")
        if not isinstance(self.permission_mode, PermissionMode):
            raise ValueError("独立 Agent permission_mode 无效")
        if not isinstance(self.background, bool):
            raise ValueError("独立 Agent background 必须是布尔值")
        if not isinstance(self.tool_view, ToolView):
            raise ValueError("独立 Agent tool_view 类型无效")
        if self.origin is IndependentAgentOrigin.SKILL_FORK:
            if not isinstance(self.skill, SkillDefinition):
                raise ValueError("Skill fork 必须包含 SkillDefinition")
        elif self.skill is not None:
            raise ValueError("只有 Skill fork 可以包含 SkillDefinition")
        if not isinstance(self.skill_arguments, str):
            raise ValueError("Skill fork 参数必须是字符串")
        if self.workspace is not None and not isinstance(
            self.workspace,
            WorkspaceAssignment,
        ):
            raise ValueError("独立 Agent workspace 类型无效")
        if self.team_actor is not None and not isinstance(
            self.team_actor, TeamActorContext
        ):
            raise ValueError("独立 Agent team_actor 类型无效")


class BackgroundTaskStatus(str, Enum):
    """表示一个后台子 Agent 当前处于队列、运行还是终态。"""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"

    @property
    def terminal(self) -> bool:
        """判断该状态是否已经不会继续变化。

        Returns:
            完成、阶段性完成、失败或取消时返回 ``True``；排队或运行时返回 ``False``。
        """

        return self in {
            BackgroundTaskStatus.COMPLETED,
            BackgroundTaskStatus.PARTIAL,
            BackgroundTaskStatus.FAILED,
            BackgroundTaskStatus.CANCELLED,
            BackgroundTaskStatus.INTERRUPTED,
        }


@dataclass
class AgentUsage:
    """累计一次子 Agent 运行实际消耗的模型、工具和时间统计。

    Attributes:
        model_calls: 已完成的 Provider 请求次数。
        input_tokens: Provider 报告的总输入 token 数。
        cached_input_tokens: Provider 报告的缓存命中 token 总数；所有请求均
            未报告该字段时为 ``None``，报告过零也保留为 ``0``。
        output_tokens: Provider 报告的总输出 token 数。
        tool_calls: 实际进入工具调度器的调用数量。
        duration_ms: 从运行开始到结束的单调时钟耗时，单位为毫秒。
    """

    model_calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int | None = None
    output_tokens: int = 0
    tool_calls: int = 0
    duration_ms: int = 0

    def __post_init__(self) -> None:
        """拒绝负数、布尔值和非法缓存 token 统计。

        Returns:
            所有统计均可加总时不返回数据。

        Raises:
            ValueError: 任一计数不是非负整数，或缓存计数不是 ``None``/
                非负整数。
        """

        values = (
            self.model_calls,
            self.input_tokens,
            self.output_tokens,
            self.tool_calls,
            self.duration_ms,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("Agent 用量统计必须是非负整数")
        if self.cached_input_tokens is not None and (
            isinstance(self.cached_input_tokens, bool)
            or not isinstance(self.cached_input_tokens, int)
            or self.cached_input_tokens < 0
        ):
            raise ValueError("Agent 缓存输入统计必须是非负整数或 None")


@dataclass(frozen=True)
class AgentRunResult:
    """独立子 Agent 结束后交给前台调用方或后台任务管理器的结果。

    Attributes:
        status: 完成、阶段性完成、失败或取消终态。
        final_text: 完成或阶段性完成时模型返回的正式文本。
        partial_text: 失败或取消前已经生成、仍可用于诊断的可见文本。
        error: 失败或取消原因；成功时必须为 ``None``。
        usage: 这次运行累计的模型、工具和时间统计。
        workspace_report: Worktree 收尾后实际删除或保留目录的报告；共享旧调用
            或尚未接入工作区时为 ``None``。
    """

    status: BackgroundTaskStatus
    final_text: str | None
    partial_text: str | None
    error: str | None
    usage: AgentUsage
    workspace_report: WorktreeFinishReport | None = None

    def __post_init__(self) -> None:
        """保证终态与最终文本、部分文本、错误和用量互相一致。

        Returns:
            完成或失败结果字段合法时不返回数据。

        Raises:
            ValueError: 使用非终态，成功缺少文本，失败缺少错误，或用量
                类型无效。
        """

        if not self.status.terminal:
            raise ValueError("AgentRunResult 必须使用终态")
        if self.status in {
            BackgroundTaskStatus.COMPLETED,
            BackgroundTaskStatus.PARTIAL,
        }:
            if (
                not self.final_text
                or self.partial_text is not None
                or self.error is not None
            ):
                raise ValueError("完成或阶段性完成结果必须只包含正式文本")
        else:
            if self.final_text is not None or not self.error:
                raise ValueError("失败或取消结果必须包含错误且不能包含最终文本")
        if self.partial_text is not None and not self.partial_text:
            raise ValueError("Agent 部分结果不能是空字符串")
        if not isinstance(self.usage, AgentUsage):
            raise ValueError("AgentRunResult usage 类型无效")
        if self.workspace_report is not None and not isinstance(
            self.workspace_report,
            WorktreeFinishReport,
        ):
            raise ValueError("AgentRunResult workspace_report 类型无效")


@dataclass(frozen=True)
class TaskMetadata:
    """在接管一个已运行句柄时补充后台任务列表需要的文字信息。

    Attributes:
        task_id: 后台任务的唯一标识。
        name: 任务列表展示的短名称。
        description: 任务列表展示的用途说明。
        source: 创建任务的入口，例如 ``agent`` 或 ``verification``。
        session_id: 拥有该任务的主会话标识。
    """

    task_id: str
    name: str
    description: str
    source: str
    session_id: str

    def __post_init__(self) -> None:
        """校验任务列表展示和会话归属所需的五个文本字段。

        Returns:
            所有字段均为非空字符串时不返回数据。

        Raises:
            ValueError: 任一字段为空或不是字符串。
        """

        for field_name, value in vars(self).items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"任务元数据 {field_name} 必须是非空字符串")


@dataclass
class BackgroundTaskRecord:
    """TaskManager 在进程内更新的一条后台任务记录。

    Attributes:
        task_id: 查询、停止和通知使用的唯一任务 ID。
        name: 后台任务列表展示的短名称。
        description: 说明该任务正在做什么。
        source: 创建任务的产品入口。
        session_id: 拥有任务的主会话 ID。
        status: 当前排队、运行或终态。
        result: 任务结束后保存的结果；排队和运行时为 ``None``。
        created_at: 任务进入 TaskManager 的时间。
        started_at: worker 或已接管句柄开始运行的时间。
        ended_at: 任务进入终态的时间。
        cancellation: 停止任务时触发的协作式取消令牌。
    """

    task_id: str
    name: str
    description: str
    source: str
    session_id: str
    status: BackgroundTaskStatus
    result: AgentRunResult | None
    created_at: datetime
    started_at: datetime | None
    ended_at: datetime | None
    cancellation: CancellationToken

    def __post_init__(self) -> None:
        """校验内部记录的元数据、状态、结果和时间点。

        Returns:
            记录可以进入 TaskManager 时不返回数据。

        Raises:
            ValueError: 元数据为空，或状态与结果、时间字段不一致。
        """

        TaskMetadata(
            task_id=self.task_id,
            name=self.name,
            description=self.description,
            source=self.source,
            session_id=self.session_id,
        )
        _validate_task_times(
            self.status,
            self.result,
            self.created_at,
            self.started_at,
            self.ended_at,
        )


@dataclass(frozen=True)
class BackgroundTaskSnapshot:
    """Task 工具返回给模型的不可变后台任务视图。

    Attributes:
        task_id: 可交给 TaskGet 或 TaskStop 的任务 ID。
        name: 任务短名称。
        description: 任务用途说明。
        source: 创建任务的入口。
        session_id: 拥有任务的会话 ID。
        status: 生成快照时观察到的任务状态。
        result: 终态任务的运行结果；非终态为 ``None``。
        created_at: 任务创建时间。
        started_at: 任务开始时间，尚在队列时为 ``None``。
        ended_at: 任务结束时间，尚未结束时为 ``None``。
    """

    task_id: str
    name: str
    description: str
    source: str
    session_id: str
    status: BackgroundTaskStatus
    result: AgentRunResult | None
    created_at: datetime
    started_at: datetime | None
    ended_at: datetime | None

    def __post_init__(self) -> None:
        """校验对外任务快照的元数据、状态、结果和时间点。

        Returns:
            快照字段可以被 Task 工具展示时不返回数据。

        Raises:
            ValueError: 元数据为空，或状态与结果、时间字段不一致。
        """

        TaskMetadata(
            task_id=self.task_id,
            name=self.name,
            description=self.description,
            source=self.source,
            session_id=self.session_id,
        )
        _validate_task_times(
            self.status,
            self.result,
            self.created_at,
            self.started_at,
            self.ended_at,
        )


@dataclass(frozen=True)
class TaskNotification:
    """后台任务结束后注入主对话的精简通知。

    Attributes:
        task_id: 已结束任务的 ID。
        session_id: 应接收通知的主会话 ID。
        status: 任务的完成、失败或取消终态。
        result_text: 经过统一脱敏后的最终文本、部分文本或错误说明。
        usage: 任务累计用量，不包含中间消息和工具正文。
        workspace_report: 子 Agent 目录最终被删除、保留或释放的报告。
    """

    task_id: str
    session_id: str
    status: BackgroundTaskStatus
    result_text: str
    usage: AgentUsage
    workspace_report: WorktreeFinishReport | None = None

    def __post_init__(self) -> None:
        """保证通知只携带所属会话可消费的终态结果。

        Returns:
            ID、终态、结果文字和用量均有效时不返回数据。

        Raises:
            ValueError: ID 或结果为空、状态不是终态，或用量类型无效。
        """

        if not self.task_id.strip() or not self.session_id.strip():
            raise ValueError("任务通知必须包含任务和会话 ID")
        if not self.status.terminal:
            raise ValueError("任务通知只能携带终态")
        if not self.result_text.strip():
            raise ValueError("任务通知必须包含可展示结果")
        if not isinstance(self.usage, AgentUsage):
            raise ValueError("任务通知 usage 类型无效")
        if self.workspace_report is not None and not isinstance(
            self.workspace_report,
            WorktreeFinishReport,
        ):
            raise ValueError("任务通知 workspace_report 类型无效")


@dataclass(frozen=True)
class AgentReloadReport:
    """汇总一次逐角色 reload 对当前目录造成的变化。

    Attributes:
        added: 本次新增并生效的正式角色名。
        updated: revision 发生变化并成功替换的角色名。
        removed: 所有来源均已删除、因此从目录移除的角色名。
        retained: 新候选损坏、继续使用旧定义的角色名。
        diagnostics: 本次扫描和安装产生的具体问题。
    """

    added: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    retained: tuple[str, ...] = ()
    diagnostics: tuple[AgentDiagnostic, ...] = ()


def _validate_task_times(
    status: BackgroundTaskStatus,
    result: AgentRunResult | None,
    created_at: datetime,
    started_at: datetime | None,
    ended_at: datetime | None,
) -> None:
    """校验后台任务状态、结果和三个时间点是否互相一致。

    Args:
        status: 当前任务状态。
        result: 终态运行结果或 ``None``。
        created_at: 任务创建时间。
        started_at: 任务开始时间或 ``None``。
        ended_at: 任务结束时间或 ``None``。

    Returns:
        校验通过时不返回数据。

    Raises:
        ValueError: 时间先后矛盾，或终态与结果字段不匹配。
    """

    if not isinstance(status, BackgroundTaskStatus):
        raise ValueError("后台任务状态无效")
    for value in (created_at, started_at, ended_at):
        if value is not None and not isinstance(value, datetime):
            raise ValueError("后台任务时间必须是 datetime 或 None")
    if started_at is not None and started_at < created_at:
        raise ValueError("后台任务开始时间不能早于创建时间")
    if ended_at is not None:
        lower_bound = started_at or created_at
        if ended_at < lower_bound:
            raise ValueError("后台任务结束时间不能早于开始时间")
    if status is BackgroundTaskStatus.QUEUED:
        if started_at is not None or ended_at is not None or result is not None:
            raise ValueError("排队任务不能包含开始、结束时间或结果")
    elif status is BackgroundTaskStatus.RUNNING:
        if started_at is None or ended_at is not None or result is not None:
            raise ValueError("运行任务必须有开始时间且不能有结束时间或结果")
    else:
        if started_at is None or ended_at is None or result is None:
            raise ValueError("终态任务必须包含开始时间、结束时间和结果")
        if result.status is not status:
            raise ValueError("后台任务状态必须与运行结果状态一致")
