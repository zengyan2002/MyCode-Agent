"""把统一 Agent 工具请求分流到定义式、Fork、前台或后台执行。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING
from uuid import uuid4

from mycode.agents.catalog import AgentCatalog
from mycode.agents.loader import AgentLoader
from mycode.agents.prompts import (
    build_fork_messages,
    definition_role_section,
    subagent_constraints_section,
)
from mycode.agent.finalization import strip_model_budget_instructions
from mycode.agents.runtime import AgentRunHandle, IndependentAgentRuntimeBuilder
from mycode.agents.snapshots import ParentRunRecorder
from mycode.agents.tasks import TaskManager
from mycode.agents.tool_policy import build_child_tool_view
from mycode.constants import DEFAULT_MAX_MODEL_CALLS
from mycode.models.agents import (
    AgentPermissionMode,
    AgentDefinition,
    AgentReloadReport,
    AgentToolRequest,
    BackgroundTaskSnapshot,
    BackgroundTaskStatus,
    IndependentAgentOrigin,
    IndependentAgentSpec,
    TaskMetadata,
)
from mycode.models.teams import BackendPreference, SpawnTeammateRequest, TeamActorContext
from mycode.models.permissions import PermissionMode
from mycode.models.prompts import PromptContext
from mycode.agent.instructions import RuntimeInstructionManager
from mycode.permissions.policy import PermissionController
from mycode.tools.base import ToolOutput
from mycode.models.tools import ToolErrorCode
from mycode.agents.workspaces import AgentWorkspaceService

if TYPE_CHECKING:
    from mycode.teams.service import TeamService


class AgentService:
    """协调角色查询、父快照、独立运行和后台 TaskManager。

    AgentTool 是唯一模型入口，只负责 Schema 和参数转换；定义式/Fork 分流、
    权限快照、前台超时移交和 reload 都集中在本类。

    Attributes:
        _loader: 显式 reload 和应用启动时扫描角色文件的 Loader。
        _catalog: Agent 工具、主提示和管理命令共用的当前角色目录。
        _runtime_builder: 为每次委派创建独立运行状态的装配器。
        _tasks: 排队、接管、查询和通知后台运行的 TaskManager。
        _parent_recorder: Fork 创建时读取的主 Agent 实际请求与响应快照。
        _foreground_handle: 当前可以被 ESC 或超时移交的前台运行句柄。
    """

    def __init__(
        self,
        loader: AgentLoader,
        catalog: AgentCatalog,
        runtime_builder: IndependentAgentRuntimeBuilder,
        task_manager: TaskManager,
        parent_recorder: ParentRunRecorder,
        parent_permissions: PermissionController,
        session_id_getter: Callable[[], str],
        base_stable_prompt: str,
        workspace_service: AgentWorkspaceService,
        *,
        auto_background_seconds: float = 120.0,
        instruction_manager: RuntimeInstructionManager | None = None,
        team_service: TeamService | None = None,
    ) -> None:
        """连接 Agent 工具执行所需的真实应用对象。

        Args:
            loader: 扫描项目、用户和内置角色的 Loader。
            catalog: 保存当前有效角色的 Catalog。
            runtime_builder: 为每次委派创建独立运行对象的 Builder。
            task_manager: 管理排队、运行、结果和通知的后台管理器。
            parent_recorder: 主 AgentTurnRunner 写入的实际请求快照。
            parent_permissions: 创建定义式运行时解析 ``inherit`` 的当前模式。
            session_id_getter: 每次委派时取得当前主会话 ID 的函数。
            base_stable_prompt: 已包含应用固定规则和项目指令的主稳定提示。
            workspace_service: 为每次定义式、Fork 和 fork Skill 运行准备固定目录。
            auto_background_seconds: 前台运行自动移交后台的秒数；``0`` 关闭。
            instruction_manager: 主 Agent 的运行时指令管理器；提供时会在
                启动和 reload 后刷新轻量角色目录。
            team_service: 可选 Agent Team 服务；提供后带 team_name 的 Agent
                请求会创建长期成员，普通委派仍走原路径。

        Returns:
            不返回数据；服务随后由 AgentTool、``/agent`` 和应用 UI 共用。
        """

        if auto_background_seconds < 0:
            raise ValueError("Agent 自动移交秒数不能为负数")
        self._loader = loader
        self._catalog = catalog
        self._runtime_builder = runtime_builder
        self._tasks = task_manager
        self._parent_recorder = parent_recorder
        self._parent_permissions = parent_permissions
        self._session_id_getter = session_id_getter
        self._base_stable_prompt = base_stable_prompt
        if not isinstance(workspace_service, AgentWorkspaceService):
            raise ValueError("workspace_service 类型无效")
        self._workspace_service = workspace_service
        self._auto_background_seconds = float(auto_background_seconds)
        self._instruction_manager = instruction_manager
        self._team_service = team_service
        # 只有定义式前台委派会设置这两个字段；Agent 工具是写工具，同一批
        # 调用按顺序执行，因此任一时刻最多只有一个可移交句柄。
        self._foreground_handle: AgentRunHandle | None = None
        self._foreground_adoption: asyncio.Event | None = None
        # 每个主会话独立递增，未显式命名的任务使用 agent-0001 等稳定名称。
        self._name_sequence: dict[str, int] = {}
        self._refresh_catalog_instruction()

    def request_foreground_adoption(self) -> bool:
        """请求把当前正在等待的定义式子 Agent 移交后台。

        Returns:
            存在可移交前台句柄时返回 ``True`` 并唤醒 ``delegate``；当前
            没有前台子 Agent、或移交已经发生时返回 ``False``。
        """

        event = self._foreground_adoption
        if self._foreground_handle is None or event is None or event.is_set():
            return False
        event.set()
        return True

    @property
    def task_manager(self) -> TaskManager:
        """返回 Agent 和 Task 工具共用的后台任务管理器。

        Returns:
            当前进程唯一的 TaskManager。
        """

        return self._tasks

    def initialize_catalog(
        self,
        known_tool_names: frozenset[str],
    ) -> AgentReloadReport:
        """在应用完成 MCP 和 Skill 注册后首次加载角色目录。

        Args:
            known_tool_names: 当前共享 ToolRegistry 已注册的全部工具名。

        Returns:
            初次扫描新增的角色和逐角色诊断。损坏定义不会阻止其他角色
            进入目录。
        """

        self._catalog.set_known_tool_names(known_tool_names)
        report = self._catalog.install_reload(self._loader.scan())
        self._refresh_catalog_instruction()
        return report

    async def delegate(
        self,
        request: AgentToolRequest,
        *,
        team_actor: TeamActorContext | None = None,
    ) -> ToolOutput:
        """执行一次定义式或 Fork 委派，并返回工具可回灌的结果。

        Args:
            request: AgentTool 已校验的 prompt、description 和可选参数。
            team_actor: ToolContext 提供的可信团队身份；普通委派为 None。

        Returns:
            前台完成时返回子 Agent 最终文本；后台启动或移交时返回任务 ID；
            角色不存在、父快照缺失或运行失败时返回结构化工具失败。
        """

        if request.team_name is not None:
            if self._team_service is None:
                return ToolOutput.fail(
                    ToolErrorCode.BLOCKED, "当前应用没有启用 Agent Team"
                )
            if team_actor is None or team_actor.actor_kind != "lead":
                return ToolOutput.fail(
                    ToolErrorCode.BLOCKED, "只有当前 Team Lead 能创建成员"
                )
            try:
                member = await self._team_service.spawn_member(
                    team_actor,
                    SpawnTeammateRequest(
                        team_name=request.team_name,
                        name=request.name or "",
                        role_name=request.subagent_type or "",
                        prompt=request.prompt,
                        model_override=request.model,
                        backend=BackendPreference(request.backend or "auto"),
                        plan_mode_required=bool(request.plan_mode_required),
                    ),
                )
            except Exception as exc:
                return ToolOutput.fail(ToolErrorCode.INTERNAL_ERROR, str(exc))
            return ToolOutput.ok(
                "team_member_started "
                f"agent_id={member.agent_id} name={member.name!r} "
                f"backend={member.backend.value}"
            )

        try:
            spec = self._build_spec(request)
            spec = await self._workspace_service.prepare(spec)
        except (KeyError, RuntimeError, ValueError) as exc:
            return ToolOutput.fail(ToolErrorCode.INVALID_ARGUMENTS, str(exc))

        if spec.background:
            try:
                snapshot = await self._tasks.launch(spec)
            except Exception as exc:
                await self._workspace_service.abandon(
                    spec,
                    f"后台任务入队失败：{exc}",
                )
                return ToolOutput.fail(ToolErrorCode.INTERNAL_ERROR, str(exc))
            return ToolOutput.ok(self._launched_text(snapshot))

        try:
            runner = self._runtime_builder.build(spec)
        except Exception as exc:
            await self._workspace_service.abandon(
                spec,
                f"子 Agent 装配失败：{exc}",
            )
            return ToolOutput.fail(ToolErrorCode.INTERNAL_ERROR, str(exc))
        handle = runner.start()
        adoption = asyncio.Event()
        self._foreground_handle = handle
        self._foreground_adoption = adoption
        adoption_waiter = asyncio.create_task(adoption.wait())
        timeout_waiter: asyncio.Task[None] | None = None
        if self._auto_background_seconds > 0:
            timeout_waiter = asyncio.create_task(
                asyncio.sleep(self._auto_background_seconds)
            )
        waiters: set[asyncio.Task[object]] = {
            handle.task,  # type: ignore[arg-type]
            adoption_waiter,  # type: ignore[arg-type]
        }
        if timeout_waiter is not None:
            waiters.add(timeout_waiter)  # type: ignore[arg-type]
        try:
            done, _ = await asyncio.wait(
                waiters,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if handle.task in done:
                result = await handle.wait()
            else:
                snapshot = await self._tasks.adopt(
                    handle,
                    TaskMetadata(
                        task_id=spec.run_id,
                        name=spec.name,
                        description=spec.description,
                        source=spec.origin.value,
                        session_id=spec.session_id,
                    ),
                )
                return ToolOutput.ok(self._launched_text(snapshot))
        except asyncio.CancelledError:
            # Ctrl+C 取消的是仍属于前台调用的 handle；已经移交的 handle 会
            # 在上面的 return 之前由 TaskManager 获得独立所有权。
            handle.cancel()
            await asyncio.gather(handle.task, return_exceptions=True)
            raise
        finally:
            if self._foreground_handle is handle:
                self._foreground_handle = None
                self._foreground_adoption = None
            adoption_waiter.cancel()
            if timeout_waiter is not None:
                timeout_waiter.cancel()
            await asyncio.gather(
                adoption_waiter,
                *(() if timeout_waiter is None else (timeout_waiter,)),
                return_exceptions=True,
            )

        if result.status in {
            BackgroundTaskStatus.COMPLETED,
            BackgroundTaskStatus.PARTIAL,
        }:
              usage = result.usage
              workspace = result.workspace_report
              metadata = {
                  "status": result.status.value,
                  "model_calls": usage.model_calls,
                  "input_tokens": usage.input_tokens,
                  "cached_input_tokens": usage.cached_input_tokens,
                  "output_tokens": usage.output_tokens,
                  "tool_calls": usage.tool_calls,
                  "duration_ms": usage.duration_ms,
              }
              if workspace is not None:
                  metadata.update(
                      {
                          "workspace_path": str(workspace.workspace.root),
                          "workspace_branch": workspace.workspace.branch,
                          "workspace_action": workspace.action.value,
                          "workspace_reason": workspace.reason,
                      }
                  )
              return ToolOutput.ok(
                  result.final_text or "子 Agent 已完成",
                  metadata=metadata,
              )
        return ToolOutput.fail(
            ToolErrorCode.CANCELLED
            if result.status is BackgroundTaskStatus.CANCELLED
            else ToolErrorCode.INTERNAL_ERROR,
            result.error or "子 Agent 运行失败",
            content=result.partial_text or "",
        )

    def format_list(self) -> str:
        """整理 `/agent list` 的角色名称、来源、说明和默认运行方式。

        Returns:
            一行一个角色的用户可见文字；没有角色时返回明确提示。
        """

        definitions = self._catalog.list()
        if not definitions:
            return "当前没有可用 Agent 角色"
        return "\n".join(
            f"- {item.name} [{item.source.value}] "
            f"[{'后台' if item.default_background else '前台'}]：{item.description}"
            for item in definitions
        )

    def format_info(self, name: str) -> str:
        """整理一个角色的生效配置，但不展示完整系统提示正文。

        Args:
            name: 用户输入的角色名。

        Returns:
            来源路径、模型、模型调用次数、权限和工具白黑名单的多行文字。

        Raises:
            KeyError: 角色不存在。
        """

        role = self._catalog.get(name)
        if role is None:
            raise KeyError(f"未知 Agent 角色：{name}")
        tools = "不额外限制" if role.tools is None else ", ".join(sorted(role.tools)) or "无"
        denied = ", ".join(sorted(role.disallowed_tools)) or "无"
        return "\n".join(
            (
                f"名称：{role.name}",
                f"说明：{role.description}",
                f"来源：{role.source.value}",
                f"路径：{role.entry_path}",
                f"模型：{role.model or 'inherit'}",
                f"最大模型调用次数：{role.max_model_calls or '系统默认'}",
                f"权限模式：{role.permission_mode.value}",
                f"默认运行：{'后台' if role.default_background else '前台'}",
                f"工具白名单：{tools}",
                f"工具黑名单：{denied}",
            )
        )

    def reload(self) -> AgentReloadReport:
        """重新扫描三层角色，并逐角色安装可用更新。

        Returns:
            新增、更新、删除、保留和诊断组成的 AgentReloadReport。
        """

        scanned = self._loader.reload(self._catalog.snapshot)
        report = self._catalog.install_reload(scanned)
        self._refresh_catalog_instruction()
        return report

    def _refresh_catalog_instruction(self) -> None:
        """把当前有效角色的名字和用途写入主 Agent 运行时指令。

        Returns:
            不返回数据；未提供 instruction_manager 时保持兼容调用方不变。
        """

        if self._instruction_manager is None:
            return
        roles = self._catalog.list()
        content = ""
        if roles:
            content = "\n".join(
                (
                    "可通过 Agent 工具委派的预定义角色：",
                    *(f"- {role.name}: {role.description}" for role in roles),
                    "固定角色任务应填写 subagent_type；临时且依赖当前对话的任务留空走 Fork。",
                )
            )
        self._instruction_manager.set_agent_catalog(content)

    def _build_spec(self, request: AgentToolRequest) -> IndependentAgentSpec:
        """把 Agent 工具原始参数解析成可排队的冻结运行输入。

        Args:
            request: 已通过 AgentTool Schema 和模型校验的调用参数。

        Returns:
            不持有主会话可变对象的 IndependentAgentSpec。

        Raises:
            ValueError: 指定的定义式角色不存在。
            RuntimeError: Fork 时当前父请求快照尚未完整。
        """

        session_id = self._session_id_getter()
        run_id = uuid4().hex
        if request.subagent_type is None:
            snapshot = self._parent_recorder.snapshot()
            messages = build_fork_messages(snapshot, request.prompt)
            origin = IndependentAgentOrigin.FORK
            background = True
            role = None
            prompt = snapshot.request.prompt
            inherited_runtime = strip_model_budget_instructions(prompt.runtime)
            initial_tools = snapshot.tool_view.visible_tool_names
            max_model_calls = DEFAULT_MAX_MODEL_CALLS
            model_override = request.model
            name = request.name or self._next_name(session_id)
        else:
            role = self._catalog.get(request.subagent_type)
            if role is None:
                available = ", ".join(
                    item.name for item in self._catalog.list()
                )
                raise ValueError(
                    f"未知 Agent 角色：{request.subagent_type}。"
                    f"当前可用：{available or '无'}"
                )
            messages = ()
            origin = IndependentAgentOrigin.DEFINITION
            background = (
                role.default_background
                if request.run_in_background is None
                else request.run_in_background
            )
            role_sections = (
                subagent_constraints_section(),
                definition_role_section(role),
            )
            appended = "".join(
                f"## {section.name}\n\n{section.content.strip()}\n\n"
                for section in role_sections
            )
            prompt = PromptContext(self._base_stable_prompt + appended)
            inherited_runtime = ()
            initial_tools = None
            max_model_calls = role.max_model_calls or DEFAULT_MAX_MODEL_CALLS
            model_override = request.model or role.model
            name = request.name or self._next_name(session_id)

        permission_mode = self._resolve_permission_mode(role)
        tool_view = build_child_tool_view(
            origin=origin,
            parent_visible_names=initial_tools,
            background=background,
            role=role,
        )
        return IndependentAgentSpec(
            run_id=run_id,
            session_id=session_id,
            name=name,
            description=request.description,
            origin=origin,
            task_prompt=request.prompt,
            initial_messages=messages,
            prompt=prompt,
            inherited_runtime=inherited_runtime,
            initial_tool_names=initial_tools,
            role=role,
            model_override=model_override,
            max_model_calls=max_model_calls,
            permission_mode=permission_mode,
            background=background,
            tool_view=tool_view,
        )

    def _resolve_permission_mode(
        self,
        role: AgentDefinition | None,
    ) -> PermissionMode:
        """把角色定义层的 ``inherit`` 快照成当前父权限模式。

        Args:
            role: 定义式 AgentDefinition；Fork 没有角色时为 ``None``。

        Returns:
            可直接创建 PermissionController 的具体 PermissionMode。
        """

        if role is None or role.permission_mode is AgentPermissionMode.INHERIT:
            return self._parent_permissions.mode
        return PermissionMode(role.permission_mode.value)

    def _next_name(self, session_id: str) -> str:
        """为一个主会话生成下一个未显式命名的任务名。

        Args:
            session_id: 本次委派所属的当前主会话 ID。

        Returns:
            从 ``agent-0001`` 开始、在该会话内单调递增的名称。
        """

        sequence = self._name_sequence.get(session_id, 0) + 1
        self._name_sequence[session_id] = sequence
        return f"agent-{sequence:04d}"

    @staticmethod
    def _launched_text(snapshot: BackgroundTaskSnapshot) -> str:
        """生成后台启动后立即回灌主模型的稳定说明。

        Args:
            snapshot: TaskManager 刚返回的 queued 或 running 快照。

        Returns:
            含 ``async_launched``、任务 ID、名称和状态的单行文字。
        """

        return (
            "async_launched "
            f"task_id={snapshot.task_id} name={snapshot.name!r} "
            f"status={snapshot.status.value}"
        )
