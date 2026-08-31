"""协议中立的工具契约与通用结果辅助结构。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from mycode.models.json_types import JsonObject, JsonValue
from mycode.models.tools import ToolDefinition, ToolErrorCode
from mycode.models.tools import ToolActivationState
from mycode.tools.file_cache import AgentFileCache
from mycode.worktrees.binding import WorkspaceBinding, shared_workspace_binding

if TYPE_CHECKING:
    from mycode.models.teams import TeamActorContext
    from mycode.skills.resources import SkillResourceAccess
    from mycode.skills.load_tool import SkillLoadRouter


@dataclass(frozen=True)
class ToolContext:
    """保存一个 Agent 的工作区绑定、缓存和 Skill 工具运行状态。

    Attributes:
        workspace: 工具每次执行时读取的工作区绑定。为兼容现有外部调用，
            构造时仍接受绝对 ``Path``，并立即转换成共享绑定。
        output_limit_bytes: 兼容旧调用方的正数字段；工具不再据此截断正文。
        user_memory_root: ``read_file`` 允许读取的用户记忆目录。
        skill_resources: 当前会话已激活目录型 Skill 的只读虚拟路径。
        skill_load_router: LoadSkill 查找主服务或 fork 运行器使用的路由。
        skill_load_scope: ``main`` 修改主会话，``fork`` 只修改当前独立任务。
        file_cache: 当前 Agent 独享的文件正文缓存，父子 Agent 不共享。
        tool_activation: 当前 Agent 通过 tool_search 激活的 MCP 工具名。
        team_actor: 本地运行时确认的团队身份。普通主会话和 SubAgent 为
            ``None``；Team Lead 或成员运行时填入，工具参数不能覆盖它。
    """

    workspace: WorkspaceBinding | Path
    # 兼容旧构造代码和外部调用方；新工具实现不得据此丢弃正文。
    output_limit_bytes: int = 65_536
    # 只供 read_file 解析 ``~/.mycode/memory/<文件名>``；其他工具不会使用。
    user_memory_root: Path | None = None
    # 当前会话已激活目录型 Skill 的只读虚拟路径；其他工具不会使用。
    skill_resources: SkillResourceAccess | None = None
    # LoadSkill 根据该路由找到主 SkillService 或当前 fork 运行器。
    skill_load_router: SkillLoadRouter | None = None
    # main 表示修改主会话 Runtime；fork 表示只修改当前独立任务。
    skill_load_scope: str = "main"
    # 当前 Agent 自己的文件正文缓存；父子 Agent 不能共享该实例。
    file_cache: AgentFileCache = field(default_factory=AgentFileCache)
    # 当前 Agent 自己通过 tool_search 激活的 MCP 工具名。
    tool_activation: ToolActivationState = field(
        default_factory=ToolActivationState
    )
    team_actor: TeamActorContext | None = None

    def __post_init__(self) -> None:
        """校验工具上下文，并把旧的绝对路径参数转换成共享绑定。

        Returns:
            上下文包含有效绑定、路径、范围、缓存和激活状态时不返回数据。

        Raises:
            ValueError: 工作区、记忆路径、输出上限、Skill 范围、缓存或激活
                状态类型无效。
        """

        if isinstance(self.workspace, Path):
            if not self.workspace.is_absolute():
                raise ValueError("工具工作区根目录必须是绝对路径")
            object.__setattr__(
                self,
                "workspace",
                shared_workspace_binding(self.workspace),
            )
        elif not isinstance(self.workspace, WorkspaceBinding):
            raise ValueError("工具 workspace 必须是 WorkspaceBinding 或绝对 Path")
        if (
            self.user_memory_root is not None
            and not self.user_memory_root.is_absolute()
        ):
            raise ValueError("用户记忆根目录必须是绝对路径")
        if self.output_limit_bytes <= 0:
            raise ValueError("工具输出上限必须为正数")
        if self.skill_load_scope not in {"main", "fork"}:
            raise ValueError("Skill 加载范围只能是 main 或 fork")
        if not isinstance(self.file_cache, AgentFileCache):
            raise ValueError("工具文件缓存类型无效")
        if not isinstance(self.tool_activation, ToolActivationState):
            raise ValueError("工具激活状态类型无效")
        if self.team_actor is not None:
            from mycode.models.teams import TeamActorContext

            if not isinstance(self.team_actor, TeamActorContext):
                raise ValueError("工具团队身份类型无效")

    @property
    def workspace_root(self) -> Path:
        """读取下一次工具操作应使用的工作区根目录。

        Returns:
            调用属性时绑定快照中的绝对根目录。具体工具应只读取一次并在该次
            调用内复用，避免主会话恰好切换目录时混用两个根目录。
        """

        binding = self.workspace
        if not isinstance(binding, WorkspaceBinding):
            raise RuntimeError("ToolContext 尚未完成工作区绑定初始化")
        return binding.snapshot().root

    def set_team_actor(self, actor: TeamActorContext | None) -> None:
        """在创建、恢复、接管或删除团队后替换本地可信身份。

        Args:
            actor: 新的 Lead/成员身份；退出团队或切换到普通会话时传 None。

        Returns:
            不返回数据。该方法只供应用生命周期调用，模型工具参数不能触达。

        Raises:
            ValueError: actor 不是 TeamActorContext 或 None。
        """

        if actor is not None:
            from mycode.models.teams import TeamActorContext

            if not isinstance(actor, TeamActorContext):
                raise ValueError("工具团队身份类型无效")
        object.__setattr__(self, "team_actor", actor)

# 每个工具执行结束后都会返回一个 ToolOutput，记录执行是否成功、输出正文、错误信息和截断情况。
# ToolExecutor 随后会补上调用 ID、工具名和执行耗时，再把它转换成完整的 ToolExecutionResult。
@dataclass(frozen=True)
class ToolOutput:
    """保存一次工具执行后最直接的结果"""
    # 工具是否执行成功
    success: bool
    # 工具产生的文本内容；执行失败时也可以保存错误现场或部分输出
    content: str = ""
    # 执行失败时的错误类型；成功时必须为 None
    error_code: ToolErrorCode | None = None
    # 执行失败时给用户和模型看的错误说明；成功时必须为 None
    error_message: str | None = None
    # 退出码、结果数量等少量附加信息；大段内容应放在 content 中
    metadata: JsonObject = field(default_factory=dict)
    # content 是否因为超过输出限制而被截断
    truncated: bool = False
    # 截断前正文的 UTF-8 字节数；未截断时就是当前正文的字节数
    original_size_bytes: int = 0

    def __post_init__(self) -> None:
        if self.success and (
            self.error_code is not None or self.error_message is not None
        ):
            raise ValueError("成功的工具输出不能包含错误信息")
        if not self.success and (
            self.error_code is None or not self.error_message
        ):
            raise ValueError("失败的工具输出必须包含错误码和错误消息")
        if self.original_size_bytes < 0:
            raise ValueError("工具输出大小不能为负数")

    @classmethod
    def ok(
        cls,
        content: str = "",
        *,
        metadata: JsonObject | None = None,
        truncated: bool = False,
        original_size_bytes: int | None = None,
    ) -> "ToolOutput":
        # 未显式给出原始大小时，正文就是完整输出；发生截断的工具必须传入
        # 截断前大小，才能让模型判断缺失数据量。
        size = (
            len(content.encode("utf-8"))
            if original_size_bytes is None
            else original_size_bytes
        )
        return cls(
            success=True,
            content=content,
            metadata=metadata or {},
            truncated=truncated,
            original_size_bytes=size,
        )

    @classmethod
    def fail(
        cls,
        error_code: ToolErrorCode,
        error_message: str,
        *,
        content: str = "",
        metadata: JsonObject | None = None,
        truncated: bool = False,
        original_size_bytes: int | None = None,
    ) -> "ToolOutput":
        size = (
            len(content.encode("utf-8"))
            if original_size_bytes is None
            else original_size_bytes
        )
        return cls(
            success=False,
            content=content,
            error_code=error_code,
            error_message=error_message,
            metadata=metadata or {},
            truncated=truncated,
            original_size_bytes=size,
        )


class ToolFailure(Exception):
    """表示工具预期内的执行失败，执行器会把错误代码和说明返回给模型。"""

    def __init__(self, code: ToolErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class Tool(Protocol):
    # 新工具只需提供模型可见的定义并实现异步 execute。注册、输入校验、
    # 超时控制、异常转换和返回值包装由公共执行链处理。
    @property
    def definition(self) -> ToolDefinition:
        """返回工具名称、用途说明和模型调用时必须遵循的 JSON 输入格式。"""

    async def execute(
        self,
        arguments: Mapping[str, JsonValue],
        context: ToolContext,
    ) -> ToolOutput:
        """执行一次已经通过 Schema 校验的工具调用。"""
