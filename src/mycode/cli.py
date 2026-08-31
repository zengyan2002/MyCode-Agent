"""控制台入口及运行依赖组装。"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from rich.console import Console

from mycode.agent.conversation import Conversation
from mycode.agent.environment import EnvironmentCollector
from mycode.agent.instructions import RuntimeInstructionManager
from mycode.agent.loop import AgentLoop
from mycode.agent.system_prompt import PromptAssembler
from mycode.agents.agent_tool import AgentTool
from mycode.agents.catalog import AgentCatalog
from mycode.agents.loader import AgentLoader
from mycode.agents.parser import AgentParser
from mycode.agents.runtime import IndependentAgentRuntimeBuilder
from mycode.agents.service import AgentService
from mycode.agents.snapshots import ParentRunRecorder
from mycode.agents.task_tools import (
    TaskGetTool,
    TaskListTool,
    TaskStopTool,
)
from mycode.agents.tasks import TaskManager, TaskNotificationInbox
from mycode.app.application import ChatApplication
from mycode.app.terminal_ui import create_terminal_ui
from mycode.commands import CommandCompleter, create_builtin_registry as create_command_registry
from mycode.constants import TOOL_TIMEOUT_SECONDS
from mycode.context import ArtifactStore, ContextManager
from mycode.errors import ConfigError, redact_secrets
from mycode.hooks.actions import HookActionRunner
from mycode.hooks.adapters import PostToolHookObserver, PreToolHookInterceptor
from mycode.hooks.engine import HookEngine
from mycode.mcp import McpManager
from mycode.memory import MemoryExtractionWorker, MemoryStore
from mycode.permissions import (
    PermissionController,
    PermissionInterceptor,
    PermissionPolicy,
    PermissionRuleResolver,
)
from mycode.providers.factory import create_provider
from mycode.providers.runner import ProviderRequestRunner
from mycode.providers.transport import HttpTransport
from mycode.persistence import ProjectInstructionLoader, SessionManager
from mycode.settings import LocalPermissionStore, load_permission_settings
from mycode.settings.loader import load_startup_config
from mycode.models.agents import AgentCatalogSnapshot
from mycode.models.teams import TeammateBackend
from mycode.models.tools import ToolSource
from mycode.skills.catalog import SkillCatalog
from mycode.skills.fork import SkillForkRunner
from mycode.skills.load_tool import LoadSkillTool, SkillLoadRouter
from mycode.skills.loader import SkillLoader
from mycode.skills.parser import SkillParser
from mycode.skills.resources import SkillResourceAccess
from mycode.skills.runtime import SkillRuntime
from mycode.skills.service import SkillService
from mycode.skills.trust import SkillTrustInterceptor, SkillTrustStore
from mycode.tools import (
    ToolContext,
    ToolExecutor,
    create_builtin_registry as create_tool_registry,
)
from mycode.tools.interceptors import PlanOnlyInterceptor
from mycode.tools.scheduler import ToolScheduler
from mycode.worktrees.binding import shared_workspace_binding
from mycode.worktrees.cleanup import WorktreeCleanupService
from mycode.worktrees.git import GitWorktreeBackend
from mycode.worktrees.initializer import WorktreeInitializer
from mycode.worktrees.manager import WorktreeManager
from mycode.worktrees.state import WorktreeStateStore
from mycode.agents.workspaces import AgentWorkspaceService
from mycode.teams.backends.detection import BackendDetector
from mycode.teams.backends.base import TeammateLaunch
from mycode.teams.backends.in_process import InProcessBackend
from mycode.teams.backends.iterm2 import ITerm2Backend
from mycode.teams.backends.tmux import TmuxBackend
from mycode.teams.host import TeammateHost
from mycode.teams.integration import TeamIntegrationService
from mycode.teams.mailbox import TeamMailbox
from mycode.teams.message_tool import SendMessageTool
from mycode.teams.policy import (
    CoordinatorCommandInterceptor,
    CoordinatorCommandObserver,
    CoordinatorFileInterceptor,
    TeamActorInterceptor,
    build_team_tool_view,
)
from mycode.teams.runtime import (
    TeamAgentWorkspaceService,
    TeamMemberRuntimeFactory,
)
from mycode.teams.service import TeamService
from mycode.teams.store import TeamStateStore
from mycode.teams.supervisor import TeammateSupervisor
from mycode.teams.task_tools import (
    TeamTaskClaimTool,
    TeamTaskCreateTool,
    TeamTaskGetTool,
    TeamTaskListTool,
    TeamTaskUpdateTool,
)
from mycode.teams.tasks import TeamTaskBoard
from mycode.teams.team_tools import (
    TeamCreateTool,
    TeamDeleteTool,
    TeamGetTool,
    TeamMemberStopTool,
    TeamTakeoverTool,
)


_SESSION_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")


def _parse_startup_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """解析启动参数并在创建网络客户端之前校验恢复会话 ID。

    Args:
        argv: 不含程序名的命令行参数；``None`` 时由 argparse 读取进程参数。

    Returns:
        包含可选 ``resume`` 会话 ID 的命名空间。

    Raises:
        SystemExit: 用户请求帮助，或参数/会话 ID 格式无效。
    """

    parser = argparse.ArgumentParser(prog="mycode")
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="恢复指定会话及其最后使用的 Worktree",
    )
    parser.add_argument(
        "--team-host",
        nargs=3,
        metavar=("TEAM_ID", "AGENT_ID", "GENERATION"),
        help=argparse.SUPPRESS,
    )
    parsed = parser.parse_args(argv)
    if parsed.resume is not None and parsed.team_host is not None:
        parser.error("--resume 不能与内部 --team-host 同时使用")
    if parsed.resume is not None and _SESSION_ID.fullmatch(parsed.resume) is None:
        parser.error("--resume 必须使用 YYYYMMDD-HHMMSS-xxxx 格式的会话 ID")
    if parsed.team_host is not None:
        try:
            generation = int(parsed.team_host[2])
        except ValueError:
            parser.error("内部 Host generation 必须是正整数")
        if generation < 1:
            parser.error("内部 Host generation 必须是正整数")
    return parsed


async def _run_team_host(startup_args: argparse.Namespace) -> int:
    """在独立终端窗格中恢复一个长期团队成员并等待邮箱唤醒。

    Args:
        startup_args: 已通过 ``_parse_startup_args`` 校验的内部 Host 参数，
            其中包含 team ID、agent ID 和正整数 generation。

    Returns:
        Host 正常收到退出请求时返回 ``0``；配置、租约、会话恢复或运行失败
        时返回 ``1``。
    """

    assert startup_args.team_host is not None
    team_id, agent_id, generation_text = startup_args.team_host
    lease = os.environ.get("MYCODE_TEAM_LEASE", "")
    root_text = os.environ.get("MYCODE_TEAM_ROOT", "")
    if not lease or not root_text:
        Console(stderr=True).print(
            "[错误] 内部团队 Host 缺少租约或主工作区路径",
            style="bold red",
            markup=False,
        )
        return 1
    workspace_root = Path(root_text).resolve(strict=True)
    team_store = TeamStateStore(workspace_root)
    snapshot = team_store.load_team(team_id)
    member = next(
        (item for item in snapshot.members if item.agent_id == agent_id),
        None,
    )
    if member is None:
        Console(stderr=True).print(
            f"[错误] 团队成员不存在：{agent_id}",
            style="bold red",
            markup=False,
        )
        return 1

    try:
        config = load_startup_config()
        permission_settings = load_permission_settings(workspace_root)
    except ConfigError as exc:
        Console(stderr=True).print(
            f"[错误] {redact_secrets(str(exc))}",
            style="bold red",
            markup=False,
        )
        return 1

    transport = HttpTransport()
    try:
        provider_config = config.active_provider
        request_runner = ProviderRequestRunner(
            create_provider(provider_config, transport)
        )
        command_registry = create_command_registry()
        ui = create_terminal_ui(
            secrets=config.secrets,
            command_completer=CommandCompleter(command_registry),
        )
        registry = create_tool_registry()
        registry.register(LoadSkillTool(), source=ToolSource.SYSTEM)
        team_tasks = TeamTaskBoard(team_store)
        team_mailbox = TeamMailbox(team_store)
        permission_controller = PermissionController(permission_settings)
        permission_store = LocalPermissionStore(permission_settings.local_path)
        hook_engine = HookEngine(
            config.hooks,
            HookActionRunner(workspace_root),
        )
        worktree_git = GitWorktreeBackend(workspace_root)
        worktree_manager = WorktreeManager(
            workspace_root,
            shared_workspace_binding(workspace_root),
            worktree_git,
            WorktreeInitializer(workspace_root, config.worktrees, worktree_git),
            WorktreeStateStore(workspace_root),
        )
        team_integration = TeamIntegrationService(
            workspace_root,
            team_store,
            worktree_manager,
        )

        async def validate_member_completion(actor, request) -> None:
            """验证独立 Host 准备提交的任务完成记录。

            Args:
                actor: 成员运行时注入的可信 TeamActorContext。
                request: 成员本次 TeamTaskUpdate 的显式字段。

            Returns:
                提交归属、Worktree 和结果检查通过时不返回数据。
            """

            await team_integration.validate_task_completion(
                actor,
                request.task_id,
                commit_hashes=request.commit_hashes,
                result=request.result,
            )

        for team_tool in (
            TeamTaskListTool(team_tasks),
            TeamTaskGetTool(team_tasks),
            TeamTaskClaimTool(team_tasks),
            TeamTaskUpdateTool(
                team_tasks,
                completion_validator=validate_member_completion,
            ),
            SendMessageTool(team_mailbox),
        ):
            registry.register(team_tool, source=ToolSource.SYSTEM)
        runtime_builder = IndependentAgentRuntimeBuilder(
            request_runner,
            registry,
            hook_engine,
            provider_config,
            workspace_root,
            permission_settings,
            permission_controller,
            permission_store,
            ui,
            SkillCatalog(),
            TeamAgentWorkspaceService(worktree_manager),
            secrets=config.secrets,
        )
        agent_loader = AgentLoader.from_workspace(
            AgentParser(),
            member.worktree_path,
            enable_verification=config.agents.enable_verification,
        )
        agent_catalog = AgentCatalog(agent_loader.scan())
        stable_prompt = PromptAssembler(
            project_instructions=ProjectInstructionLoader(
                member.worktree_path
            ).load().content
        ).build()
        runtime_factory = TeamMemberRuntimeFactory(
            store=team_store,
            mailbox=team_mailbox,
            catalog=agent_catalog,
            runtime_builder=runtime_builder,
            request_runner=request_runner,
            provider_config=provider_config,
            stable_prompt=stable_prompt,
            parent_permissions=permission_controller,
        )
        host = TeammateHost(team_store, team_mailbox, runtime_factory)

        async def wait_for_wake() -> None:
            """阻塞读取窗格标准输入，收到一行后让 Host 检查邮箱。

            Returns:
                tmux 或 iTerm2 adapter 发送换行后返回，不产生数据。
            """

            line = await asyncio.to_thread(sys.stdin.readline)
            if line == "":
                raise RuntimeError("成员 Host 的终端输入已经关闭")

        launch = TeammateLaunch(
            workspace_root=workspace_root,
            worktree_path=member.worktree_path,
            team_id=team_id,
            agent_id=agent_id,
            generation=int(generation_text),
            lease_token=lease,
            prompt=team_store.load_runtime_prompt(team_id, agent_id),
        )
        await host(launch, wait_for_wake)
        return 0
    except Exception as exc:
        Console(stderr=True).print(
            f"[错误] 团队成员 Host 失败：{redact_secrets(str(exc), config.secrets)}",
            style="bold red",
            markup=False,
        )
        return 1
    finally:
        await transport.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    """装配 MyCode 并运行终端应用。

    Args:
        argv: 不含程序名的启动参数；测试可传空列表，控制台入口传 ``None``。

    Returns:
        ``0`` 表示正常退出，``1`` 表示关闭资源失败，``2`` 表示参数或配置错误。
    """

    try:
        startup_args = _parse_startup_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    if startup_args.team_host is not None:
        return asyncio.run(_run_team_host(startup_args))
    # 这里是整个程序的“装配根”：各层只声明自己需要的依赖，只有入口知道
    # 具体使用哪一种 UI、网络传输、Provider 和工具实现。集中组装可以避免
    # 业务模块主动创建全局对象，也方便测试替换其中任意组件。
    workspace_root = Path.cwd().resolve()
    workspace_binding = shared_workspace_binding(workspace_root)
    try:
        # 启动阶段先加载并校验配置，失败时不创建网络客户端。
        config = load_startup_config()
        permission_settings = load_permission_settings(workspace_root)
    except ConfigError as exc:
        # 配置错误属于可预期的启动失败，统一返回状态码 2。
        Console(stderr=True).print(
            f"[错误] {redact_secrets(str(exc))}",
            style="bold red",
            markup=False,
            highlight=False,
        )
        return 2

    # Provider 工厂把配置中的协议枚举映射到具体适配器；上层 Agent 只依赖
    # 统一 Provider 接口，不需要知道请求头或 SSE 数据格式。
    provider_config = config.active_provider
    loaded_instructions = ProjectInstructionLoader(workspace_root).load()
    # UI 与网络传输都在整个终端会话中复用。UI 需要所有已配置密钥只用于
    # 本地展示脱敏；它不会读取密钥原文以外的 Provider 私有状态。
    command_registry = create_command_registry()
    ui = create_terminal_ui(
        secrets=config.secrets,
        command_completer=CommandCompleter(command_registry),
    )
    transport = HttpTransport()
    provider = create_provider(provider_config, transport)

    memory_store = MemoryStore(
        workspace_root,
        Path.home(),
        secrets=config.secrets,
    )

    # 注册顺序会原样暴露给模型，因此内置注册表负责提供稳定的工具列表。
    registry = create_tool_registry()
    # LoadSkill 是 Agent 运行基础设施。主会话和 fork 共用工具定义，但由
    # 各自 ToolContext 选择实际修改哪个 Skill Runtime。
    skill_load_router = SkillLoadRouter()
    registry.register(LoadSkillTool(), source=ToolSource.SYSTEM)
    # 创建MCP管理器对象
    mcp_manager = (
        McpManager(
            config.mcp_servers,
            secrets=config.secrets,
        )
        if config.mcp_servers
        else None
    )

    # 主会话持有自己的 Skill 资源和信任状态。独立运行由
    # IndependentAgentRuntimeBuilder 为每次调用重新创建这些对象。
    main_skill_resources = SkillResourceAccess()
    main_skill_trust = SkillTrustStore()

    # 所有内置工具都在当前项目目录内运行；read_file 还可以读取用户记忆目录中的单个文件
    tool_context = ToolContext(
        workspace=workspace_binding,
        user_memory_root=memory_store.user_memory_root,
        skill_resources=main_skill_resources,
        skill_load_router=skill_load_router,
        skill_load_scope="main",
    )
    # TeamStateStore 只保存当前主工作区的团队数据。它在工具调度器之前
    # 创建，是因为每次团队工具和 Coordinator 命令执行前都要重新读取
    # Actor generation，不能信任模型上一轮看到的旧状态。
    team_store = TeamStateStore(workspace_root)

    # Executor 负责一次调用的校验、超时和异常隔离；Scheduler 再负责一批调用的并发与顺序，两层职责不要混在具体工具实现中。
    executor = ToolExecutor(
        registry,
        tool_context,
        timeout_seconds=TOOL_TIMEOUT_SECONDS,
    )

    # 保存当前会话的消息历史，Provider 发送请求时再转换成对应的接口格式
    conversation = Conversation()

    # 准备本次运行使用的 artifact 存储。过长的工具输出会在脱敏后写入`.mycode/artifacts/<随机 ID>/`，执行 /clear 或正常退出时删除该目录
    artifact_store = ArtifactStore(
        workspace_root,
        uuid.uuid4().hex,
        secrets=config.secrets,
    )

    # 负责发送模型请求，并处理模型返回的事件流
    request_runner = ProviderRequestRunner(provider)

    # 创建当前会话的上下文管理器，负责保存过长的工具输出、估算模型请求大小、并在对话接近上下文上限时用摘要替换较早消息，保留近期完整消息
    context_manager = ContextManager(
        request_runner,
        conversation,
        provider_config,
        artifact_store,
    )

    # 准备会话管理器，把对话消息写入 `.mycode/sessions/` 下的 JSONL 文件
    # 并在新建或恢复会话时同步更新内存消息和上下文摘要状态
    session_manager = SessionManager(
        workspace_root,
        conversation,
        context_manager,
    )
    # 删除超过保留天数的非当前会话，并返回实际删除数量
    cleaned_sessions = session_manager.cleanup_expired(
        datetime.now().astimezone()
    )
    # 创建新的空会话
    session_manager.create_new()

    # Catalog 跨会话复用；RuntimeInstructionManager 和 SkillRuntime 只保存
    # 当前主会话的活动 SOP、资源范围和旁路元数据。
    skill_catalog = SkillCatalog()
    instruction_manager = RuntimeInstructionManager(
        EnvironmentCollector(workspace_binding)
    )
    skill_runtime = SkillRuntime(
        skill_catalog,
        instruction_manager,
        main_skill_resources,
        main_skill_trust,
        session_manager,
    )

    # 创建记忆提取任务。每个正常结束的对话回合都会交给它分析
    # 它调用模型找出值得跨会话保留的信息，并把笔记写入记忆目录
    memory_worker = MemoryExtractionWorker(
        request_runner,
        memory_store,
        secrets=config.secrets,
    )

    # 保存当前权限模式、仅本会话有效的授权规则，以及最新的本地权限规则
    permission_controller = PermissionController(permission_settings)

    # 创建分层权限规则选择器
    permission_resolver = PermissionRuleResolver(
        permission_settings.user,
        permission_settings.project,
    )
    permission_policy = PermissionPolicy(
        permission_resolver,
        permission_controller,
    )
    permission_store = LocalPermissionStore(
        permission_settings.local_path
    )
    permission_interceptor = PermissionInterceptor(
        registry,
        tool_context,
        permission_policy,
        permission_controller,
        ui,
        permission_store,
    )

    # 三层配置已在启动加载阶段完成校验。主会话和 fork 共用规则与动作资源，
    # 但各自的 once、提示词和后台任务由独立 HookRunScope 保存。
    hook_actions = HookActionRunner(workspace_root)
    hook_engine = HookEngine(config.hooks, hook_actions)
    pre_tool_hooks = PreToolHookInterceptor(hook_engine)
    post_tool_hooks = PostToolHookObserver(hook_engine)

    stable_prompt = PromptAssembler(
        project_instructions=loaded_instructions.content
    ).build()
    worktree_git = GitWorktreeBackend(workspace_root)
    worktree_manager = WorktreeManager(
        workspace_root,
        workspace_binding,
        worktree_git,
        WorktreeInitializer(workspace_root, config.worktrees, worktree_git),
        WorktreeStateStore(workspace_root),
    )
    workspace_service = AgentWorkspaceService(worktree_manager)
    worktree_cleanup = WorktreeCleanupService(
        worktree_manager,
        config.worktrees,
    )
    team_integration = TeamIntegrationService(
        workspace_root,
        team_store,
        worktree_manager,
    )

    ## 创建工具调度器，负责安排工具的执行顺序与并发，并在 Plan 模式下拦截写工具。
    scheduler = ToolScheduler(
        registry,
        executor,
        interceptors=(
            PlanOnlyInterceptor(),
            TeamActorInterceptor(
                team_store,
                lambda: tool_context.team_actor,
            ),
            CoordinatorCommandInterceptor(
                lambda: tool_context.team_actor,
                lambda team_id: team_store.load_team(team_id).integration,
                team_integration,
            ),
            CoordinatorFileInterceptor(
                workspace_root,
                lambda: tool_context.team_actor,
                lambda team_id: team_store.load_team(team_id).integration,
            ),
            SkillTrustInterceptor(registry, ui, main_skill_trust),
            pre_tool_hooks,
            permission_interceptor,
        ),
        observers=(
            post_tool_hooks,
            CoordinatorCommandObserver(
                lambda: tool_context.team_actor,
                team_integration,
            ),
        ),
    )
    agent_runtime_builder = IndependentAgentRuntimeBuilder(
        request_runner,
        registry,
        hook_engine,
        provider_config,
        workspace_root,
        permission_settings,
        permission_controller,
        permission_store,
        ui,
        skill_catalog,
        workspace_service,
        user_memory_root=memory_store.user_memory_root,
        secrets=config.secrets,
    )
    fork_runner = SkillForkRunner(
        agent_runtime_builder,
        permission_controller,
        lambda: session_manager.current_id,
        stable_prompt,
        workspace_service,
    )
    skill_loader = SkillLoader.from_workspace(
        SkillParser(reserved_names=command_registry.static_names),
        workspace_root,
    )
    skill_service = SkillService(
        skill_loader,
        skill_catalog,
        skill_runtime,
        fork_runner,
        command_registry,
        registry,
        lambda: session_manager.history,
    )
    skill_load_router.bind_main(skill_service)

    # Agent 服务和稳定工具先创建；角色定义等 MCP 与 Skill 完成注册后，
    # 再由 ChatApplication 使用完整工具表逐项校验并安装。
    agent_loader = AgentLoader.from_workspace(
        AgentParser(),
        workspace_root,
        enable_verification=config.agents.enable_verification,
    )
    agent_catalog = AgentCatalog(AgentCatalogSnapshot({}, {}, ()))
    team_tasks = TeamTaskBoard(team_store)
    team_mailbox = TeamMailbox(team_store)
    team_workspace_service = TeamAgentWorkspaceService(worktree_manager)
    team_runtime_builder = IndependentAgentRuntimeBuilder(
        request_runner,
        registry,
        hook_engine,
        provider_config,
        workspace_root,
        permission_settings,
        permission_controller,
        permission_store,
        ui,
        skill_catalog,
        team_workspace_service,
        user_memory_root=memory_store.user_memory_root,
        secrets=config.secrets,
    )
    team_runtime_factory = TeamMemberRuntimeFactory(
        store=team_store,
        mailbox=team_mailbox,
        catalog=agent_catalog,
        runtime_builder=team_runtime_builder,
        request_runner=request_runner,
        provider_config=provider_config,
        stable_prompt=stable_prompt,
        parent_permissions=permission_controller,
    )
    teammate_host = TeammateHost(
        team_store,
        team_mailbox,
        team_runtime_factory,
    )
    in_process_backend = InProcessBackend(teammate_host)

    def create_member_session(team_id: str) -> str:
        """在团队自己的 sessions 目录创建一个空成员会话。

        Args:
            team_id: 新成员所属团队 ID，用来定位持久化目录。

        Returns:
            新建 JSONL 会话的稳定 session ID。

        Raises:
            SessionError: 目录或会话文件无法创建。
        """

        member_conversation = Conversation()
        member_artifacts = ArtifactStore(
            workspace_root,
            f"team-session-{uuid.uuid4().hex}",
            secrets=config.secrets,
        )
        member_context = ContextManager(
            request_runner,
            member_conversation,
            provider_config,
            member_artifacts,
        )
        member_sessions = SessionManager(
            workspace_root,
            member_conversation,
            member_context,
            sessions_dir=team_store.team_dir(team_id) / "sessions",
        )
        try:
            return member_sessions.create_new()
        finally:
            member_sessions.close()

    teammate_supervisor = TeammateSupervisor(
        workspace_root=workspace_root,
        store=team_store,
        tasks=team_tasks,
        worktrees=worktree_manager,
        detector=BackendDetector(),
        adapters={
            TeammateBackend.TMUX: TmuxBackend(),
            TeammateBackend.ITERM2: ITerm2Backend(),
            TeammateBackend.IN_PROCESS: in_process_backend,
        },
        session_creator=create_member_session,
    )

    async def wake_team_member(
        team_id: str,
        member_id: str,
        reason: str,
    ) -> None:
        """把邮箱的显式唤醒请求转交给对应成员后端。

        Args:
            team_id: 收件成员所属团队 ID。
            member_id: 需要读取新消息的成员 ID。
            reason: 邮箱记录的唤醒原因，当前仅用于调用链可读性。

        Returns:
            后端接受唤醒时不返回数据；忙碌成员无需重复唤醒。
        """

        del reason
        await teammate_supervisor.wake(team_id, member_id)

    team_mailbox.set_wake_handler(wake_team_member)
    team_service = TeamService(
        store=team_store,
        tasks=team_tasks,
        supervisor=teammate_supervisor,
        integration=team_integration,
        sessions=session_manager,
        worktrees=worktree_manager,
        confirm_takeover=ui.confirm,
        actor_setter=tool_context.set_team_actor,
    )
    parent_recorder = ParentRunRecorder()
    task_manager = TaskManager(
        agent_runtime_builder,
        max_concurrency=config.agents.max_background_tasks,
        sanitize=lambda text: redact_secrets(text, config.secrets),
    )
    notification_inbox = TaskNotificationInbox()
    agent_service = AgentService(
        agent_loader,
        agent_catalog,
        agent_runtime_builder,
        task_manager,
        parent_recorder,
        permission_controller,
        lambda: session_manager.current_id,
        stable_prompt,
        workspace_service,
        auto_background_seconds=config.agents.auto_background_seconds,
        instruction_manager=instruction_manager,
        team_service=team_service,
    )
    # 注册顺序是模型看到的稳定顺序：统一委派入口在前，四个任务工具随后。
    # AgentService 会在自动移交时间到达时返回任务 ID；这里的独立超时更长，
    # 只在服务既没有完成也没有移交时提供外层最终上限。
    registry.register(
        AgentTool(agent_service),
        source=ToolSource.SYSTEM,
        timeout_seconds=config.agents.agent_tool_timeout_seconds,
    )
    registry.register(
        TaskListTool(task_manager, lambda: session_manager.current_id),
        source=ToolSource.SYSTEM,
    )
    registry.register(
        TaskGetTool(task_manager, lambda: session_manager.current_id),
        source=ToolSource.SYSTEM,
    )
    registry.register(
        TaskStopTool(task_manager, lambda: session_manager.current_id),
        source=ToolSource.SYSTEM,
    )
    for team_tool in (
        TeamCreateTool(team_service),
        TeamGetTool(team_service),
        TeamDeleteTool(team_service),
        TeamTakeoverTool(team_service),
        TeamMemberStopTool(team_service),
        TeamTaskCreateTool(team_service),
        TeamTaskListTool(team_tasks),
        TeamTaskGetTool(team_tasks),
        TeamTaskClaimTool(team_tasks),
        TeamTaskUpdateTool(team_tasks, team_service),
        SendMessageTool(team_mailbox),
    ):
        registry.register(team_tool, source=ToolSource.SYSTEM)

    agent = AgentLoop(
        session_manager,
        registry,
        scheduler,
        hook_engine,
        request_runner=request_runner,
        instruction_manager=instruction_manager,
        context_manager=context_manager,
        stable_prompt=stable_prompt,
        memory_store=memory_store,
        memory_worker=memory_worker,
        secrets=config.secrets,
        skill_runtime=skill_runtime,
        tool_activation=tool_context.tool_activation,
        parent_recorder=parent_recorder,
        task_manager=task_manager,
        notification_inbox=notification_inbox,
        worktree_manager=worktree_manager,
        team_tool_view_resolver=lambda base: build_team_tool_view(
            tool_context.team_actor,
            base=base,
            conflicted_files=(
                ()
                if tool_context.team_actor is None
                else tuple(
                    str(path)
                    for path in team_store.load_team(
                        tool_context.team_actor.team_id
                    ).integration.conflicted_files
                )
            ),
        ),
        team_message_drain=lambda: (
            ()
            if tool_context.team_actor is None
            else team_mailbox.drain_for_agent(tool_context.team_actor)
        ),
        team_session_refresh=team_service.restore_for_lead,
    )

    startup_notices = [
        f"项目指令警告：{warning.path}：{warning.reason}"
        for warning in loaded_instructions.warnings
    ]
    if cleaned_sessions:
        startup_notices.append(
            f"已清理 {cleaned_sessions} 个超过 30 天未活动的会话"
        )

    # ChatApplication 是最外层交互循环，负责命令和 UI；AgentLoop 才负责
    # 模型请求、工具回灌和历史提交。
    application = ChatApplication(
        agent=agent,
        ui=ui,
        transport=transport,
        provider_config=provider_config,
        command_registry=command_registry,
        memory_store=memory_store,
        permission_controller=permission_controller,
        permission_settings=permission_settings,
        secrets=config.secrets,
        mcp_manager=mcp_manager,
        mcp_registry=registry,
        session_manager=session_manager,
        memory_worker=memory_worker,
        startup_notices=startup_notices,
        skill_service=skill_service,
        agent_service=agent_service,
        task_manager=task_manager,
        notification_inbox=notification_inbox,
        worktree_manager=worktree_manager,
        worktree_cleanup=worktree_cleanup,
        team_service=team_service,
        resume_session_id=startup_args.resume,
    )

    # 状态码：0 正常退出；1 资源关闭失败；2 配置加载失败。普通运行期错误
    # 会结束当前轮并返回输入提示符，通常不会让整个进程退出。
    return application.run()
