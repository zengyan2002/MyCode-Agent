"""终端交互应用：协调全屏 UI、命令、Agent 回合与安全取消。"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass

from mycode.agent.cancellation import CancellationToken
from mycode.agent.loop import AgentLoop
from mycode.app.terminal_ui import TerminalUI
from mycode.app.ui_models import UIAction, UIActionKind
from mycode.commands.dispatcher import CommandDispatcher
from mycode.commands.models import (
    AgentSubmission,
    CommandContext,
    CommandResult,
    CommandRuntimeState,
    ParsedCommand,
    SkillSubmission,
)
from mycode.commands.parser import parse_command
from mycode.commands.registry import CommandRegistry
from mycode.errors import MyCodeError, redact_secrets
from mycode.mcp.manager import McpManager
from mycode.models.config import ProviderConfig, SecretValue
from mycode.models.events import (
    AgentErrorCode,
    AgentErrorEvent,
    AgentRunOptions,
    AgentWarningEvent,
    FinalReplyEvent,
)
from mycode.models.messages import AssistantMessage, TextBlock, UserMessage
from mycode.models.permissions import LoadedPermissionSettings, PermissionMode
from mycode.permissions.policy import PermissionController
from mycode.providers.transport import HttpTransport
from mycode.tools.registry import ToolRegistry
from mycode.models.memory import MemoryWorkerStatus, MemoryWorkerStatusKind
from mycode.memory.store import MemoryStore
from mycode.memory.worker import MemoryExtractionWorker
from mycode.persistence.sessions import SessionManager
from mycode.models.skills import SkillInvocation
from mycode.skills.service import SkillService
from mycode.agents.service import AgentService
from mycode.agents.tasks import (
    TaskManager,
    TaskNotificationInbox,
    notification_message,
)
from mycode.worktrees.cleanup import WorktreeCleanupService
from mycode.worktrees.manager import WorktreeManager
from mycode.teams.service import TeamService


@dataclass(frozen=True, slots=True)
class _TurnOutcome:
    """回合任务的安全结果，避免进程级信号越过任务边界。"""

    # True 表示用户取消了当前前台任务，UI 应显示取消提示。
    cancelled: bool = False
    # 后台任务无法直接抛给事件循环的进程信号或已知业务错误。
    failure: BaseException | None = None
    # 命令任务完成后交给应用处理的退出或 Agent 提交请求
    command_result: CommandResult | None = None


class ChatApplication:
    """协调终端输入、统一命令任务、Agent 回合和资源关闭。"""

    def __init__(
        self,
        *,
        agent: AgentLoop,
        ui: TerminalUI,
        transport: HttpTransport,
        provider_config: ProviderConfig,
        command_registry: CommandRegistry,
        memory_store: MemoryStore,
        permission_controller: PermissionController,
        permission_settings: LoadedPermissionSettings,
        session_manager: SessionManager,
        secrets: Iterable[SecretValue] = (),
        mcp_manager: McpManager | None = None,
        mcp_registry: ToolRegistry | None = None,
        memory_worker: MemoryExtractionWorker | None = None,
        startup_notices: Iterable[str] = (),
        skill_service: SkillService | None = None,
        agent_service: AgentService | None = None,
        task_manager: TaskManager | None = None,
        notification_inbox: TaskNotificationInbox | None = None,
        worktree_manager: WorktreeManager | None = None,
        worktree_cleanup: WorktreeCleanupService | None = None,
        team_service: TeamService | None = None,
        resume_session_id: str | None = None,
    ) -> None:
        """连接终端应用一轮输入会实际使用的运行组件。

        Args:
            agent: 处理普通文本和 inline Skill 的主 AgentLoop。
            ui: 读取用户动作并展示命令、模型和工具事件的终端界面。
            transport: 应用退出时需要关闭的 Provider HTTP 传输。
            provider_config: 状态栏和 /status 展示的当前模型配置。
            command_registry: 帮助、补全和分发共用的命令注册表。
            memory_store: /memory 和 Agent 运行时索引使用的长期记忆存储。
            permission_controller: 当前会话生效的权限模式和临时规则。
            permission_settings: 启动时读到的用户级、项目级权限配置。
            session_manager: 保存主会话用户、助手和工具消息的管理器。
            secrets: 错误展示前需要遮盖的配置密钥。
            mcp_manager: 可选的 MCP 连接和发现管理器。
            mcp_registry: MCP 工具最终注册到的共享工具表。
            memory_worker: 正常回合结束后提取长期笔记的后台任务。
            startup_notices: UI 启动后逐条展示的非阻塞说明。
            skill_service: 可选的动态 Skill 调用、热读和 reload 服务。
            agent_service: `/agent` 查询和热重载使用的角色服务。
            task_manager: 当前进程的后台子 Agent 队列；退出时统一取消并关闭。
            notification_inbox: 忙碌时暂存通知、下一次模型请求边界排空的
                会话收件箱。
            worktree_manager: 子 Agent 隔离、会话目录绑定和恢复使用的 Manager。
            worktree_cleanup: 启动扫描和周期清理过期临时目录的后台服务。
            team_service: 可选的长期团队生命周期服务；应用启动时恢复 Lead，
                关闭时只暂停同进程成员，不删除团队数据。
            resume_session_id: CLI ``--resume`` 指定的旧会话 ID；未提供时保留
                启动阶段创建的新会话。

        Returns:
            None。run 被调用后才启动 UI、MCP、Skill 扫描和输入循环。

        Raises:
            ValueError: 提供 MCP 管理器却没有提供对应工具注册表。
        """

        if mcp_manager is not None and mcp_registry is None:
            raise ValueError("配置 MCP 管理器时必须同时提供工具注册表")
        # 普通文本和 inline Skill 最终都由这个主循环处理。
        self._agent = agent
        # 所有用户输入、状态、错误和模型事件都通过同一界面展示。
        self._ui = ui
        # finally 中关闭该传输，释放 Provider HTTP 连接。
        self._transport = transport
        # 初始界面和状态命令读取当前 Provider 名、模型和窗口大小。
        self._provider_config = provider_config
        # 帮助、查找和补全共同使用的冻结命令注册表
        self._command_registry = command_registry
        # 统一处理未知命令、参数提示、Handler 异常和命令返回值
        self._command_dispatcher = CommandDispatcher(command_registry)
        # 供 /memory 读取长期记忆快照的真实存储
        self._memory_store = memory_store
        # 模式命令和工具权限拦截器共享当前会话权限状态。
        self._permission_controller = permission_controller
        # 供 /permission rules 展示 USER 和 PROJECT 启动快照
        self._permission_settings = permission_settings
        # 命令和前台任务报错时先替换这些敏感原文。
        self._secrets = tuple(secrets)
        # 可选的外部 MCP Server 连接与工具发现管理器。
        self._mcp_manager = mcp_manager
        # MCP 管理器把远端工具注册到 Agent 共用的集中注册表。
        self._mcp_registry = mcp_registry
        # 命令上下文、fork 回流和退出清理使用同一个主会话管理器。
        self._session_manager = session_manager
        # 有值时在应用启动和关闭期间管理自动笔记后台任务。
        self._memory_worker = memory_worker
        # 项目指令和清理提示会在 UI 可用后按顺序显示。
        self._startup_notices = tuple(startup_notices)
        # 动态斜杠命令和 LoadSkill 共用该服务；兼容测试可不启用 Skill 系统。
        self._skill_service = skill_service
        # 命令上下文读取同一服务，确保 reload 后 Agent 工具立即看到新目录。
        self._agent_service = agent_service
        # 退出路径关闭全部后台任务，避免 Provider 请求残留在事件循环中。
        self._task_manager = task_manager
        # TaskManager 的全局完成队列先由应用按 session 分发到这个收件箱。
        self._notification_inbox = notification_inbox
        self._worktree_manager = worktree_manager
        self._worktree_cleanup = worktree_cleanup
        self._team_service = team_service
        self._resume_session_id = resume_session_id
        # Shift+Tab、/plan、/do 和普通 Agent 回合共享的模式状态
        self._runtime_state = CommandRuntimeState()

    def _set_plan_only(self, enabled: bool) -> None:
        """由 Shift+Tab 修改共享模式并刷新当前 UI。

        Args:
            enabled: True 表示进入 Plan 模式，False 表示执行模式。

        Returns:
            None。
        """

        unchanged = self._runtime_state.plan_only is enabled
        self._runtime_state.plan_only = enabled
        self._ui.set_plan_mode(enabled)
        if enabled:
            message = (
                "Plan 模式保持开启" if unchanged else "Plan 模式已开启"
            )
        else:
            message = (
                "Plan 模式保持关闭" if unchanged else "Plan 模式已关闭"
            )
        self._ui.show_status(message)

    def _show_memory_status(self, status: MemoryWorkerStatus) -> None:
        """把一条后台笔记结果显示为普通状态或非阻塞警告。"""

        if status.kind is MemoryWorkerStatusKind.FAILED:
            self._ui.render_event(AgentWarningEvent(status.message))
        else:
            self._ui.show_status(status.message)

    async def _consume_turn(
        self,
        user_text: str,
        *,
        plan_only: bool,
        cancellation: CancellationToken,
        emit_user_event: bool = True,
    ) -> bool:
        if emit_user_event:
            stream = self._agent.stream_turn(
                user_text,
                options=AgentRunOptions(plan_only=plan_only),
                cancellation=cancellation,
            )
        else:
            stream = self._agent.stream_turn(
                user_text,
                options=AgentRunOptions(plan_only=plan_only),
                cancellation=cancellation,
                emit_user_event=False,
            )
        cancelled = False
        try:
            async for event in stream:
                self._ui.render_event(event)
                if (
                    isinstance(event, AgentErrorEvent)
                    and event.code is AgentErrorCode.CANCELLED
                ):
                    cancelled = True
        finally:
            await stream.aclose()
        return cancelled

    async def _consume_turn_safely(
        self,
        user_text: str,
        *,
        plan_only: bool,
        cancellation: CancellationToken,
        emit_user_event: bool = True,
    ) -> _TurnOutcome:
        """在异步任务内部收拢提供方抛出的进程级信号。"""

        try:
            cancelled = await self._consume_turn(
                user_text,
                plan_only=plan_only,
                cancellation=cancellation,
                emit_user_event=emit_user_event,
            )
        except (KeyboardInterrupt, SystemExit) as exc:
            return _TurnOutcome(failure=exc)
        return _TurnOutcome(cancelled=cancelled)

    def _start_turn(
        self,
        user_text: str,
        *,
        display_text: str | None = None,
        plan_only: bool | None = None,
        show_user_input: bool = True,
    ) -> tuple[asyncio.Task[_TurnOutcome], CancellationToken]:
        """启动一个普通或由提示词命令展开的 Agent 回合。

        Args:
            user_text: 实际写入会话并发送给 Agent 的完整正文。
            display_text: UI 展示的短文本；None 表示与实际正文相同。
            plan_only: 本轮显式模式；None 表示读取当前共享模式。
            show_user_input: False 时不调用 ``begin_turn``，也不让 Runner
                发出 UserMessageEvent；后台通知唤醒使用该值。

        Returns:
            正在运行的回合任务和可供 Ctrl+C 使用的取消令牌。
        """

        cancellation = CancellationToken()
        if show_user_input:
            self._ui.begin_turn(display_text or user_text)
        self._ui.set_busy(True)
        selected_plan = (
            self._runtime_state.plan_only
            if plan_only is None
            else plan_only
        )
        task = asyncio.create_task(
            self._consume_turn_safely(
                user_text,
                plan_only=selected_plan,
                cancellation=cancellation,
                emit_user_event=show_user_input,
            )
        )
        return task, cancellation

    def _start_pending_notification_turn(
        self,
    ) -> tuple[asyncio.Task[_TurnOutcome], CancellationToken] | None:
        """把当前会话尚未消费的后台通知启动为一个隐藏用户输入回合。

        Returns:
            有待处理通知时返回新回合任务和取消令牌；没有配置收件箱或当前
            会话队列为空时返回 ``None``。
        """

        inbox = self._notification_inbox
        if inbox is None:
            return None
        messages = inbox.drain_messages(self._session_manager.current_id)
        if not messages:
            return None
        text = "\n\n".join(message.content for message in messages)
        return self._start_turn(text, show_user_input=False)

    async def _dispatch_command_safely(
        self,
        parsed: ParsedCommand,
        cancellation: CancellationToken,
    ) -> _TurnOutcome:
        """构造真实命令上下文并执行统一分发器。

        Args:
            parsed: 已由公共解析器识别的斜杠命令。
            cancellation: 命令内压缩或恢复流程使用的取消令牌。

        Returns:
            包含命令结果、取消状态或进程级信号的任务结果。
        """

        context = CommandContext(
            invocation=parsed,
            registry=self._command_registry,
            agent=self._agent,
            ui=self._ui,
            session_manager=self._session_manager,
            memory_store=self._memory_store,
            permission_controller=self._permission_controller,
            permission_settings=self._permission_settings,
            provider_config=self._provider_config,
            runtime_state=self._runtime_state,
            cancellation=cancellation,
            secrets=self._secrets,
            skill_service=self._skill_service,
            agent_service=self._agent_service,
            worktree_manager=self._worktree_manager,
        )
        try:
            result = await self._command_dispatcher.dispatch(parsed, context)
        except (KeyboardInterrupt, SystemExit) as exc:
            return _TurnOutcome(failure=exc)
        return _TurnOutcome(
            cancelled=cancellation.is_cancelled,
            command_result=result,
        )

    def _start_command(
        self,
        parsed: ParsedCommand,
    ) -> tuple[asyncio.Task[_TurnOutcome], CancellationToken]:
        """启动一条不会自动进入普通 Agent Loop 的命令任务。

        Args:
            parsed: 已解析的命令名称、参数和原始短文本。

        Returns:
            正在运行的命令任务和可供 Ctrl+C 使用的取消令牌。
        """

        cancellation = CancellationToken()
        self._ui.set_busy(True)
        task = asyncio.create_task(
            self._dispatch_command_safely(parsed, cancellation)
        )
        return task, cancellation

    async def _consume_skill_safely(
        self,
        submission: SkillSubmission,
        cancellation: CancellationToken,
    ) -> _TurnOutcome:
        """执行 Skill 提交，并把 inline 或 fork 结果写入正确的历史。

        Args:
            submission: 动态命令 handler 生成的名字、参数和简短显示文字。
            cancellation: 用户按 Ctrl+C 时由应用触发的本次取消令牌。

        Returns:
            记录取消或失败的统一任务结果；成功时返回空结果。
        """

        service = self._skill_service
        if service is None:
            return _TurnOutcome(failure=MyCodeError("Skill 系统尚未初始化"))
        try:
            result = await service.invoke(
                SkillInvocation(
                    submission.name,
                    submission.arguments,
                    submission.display_text,
                ),
                cancellation,
            )
            if result.warning:
                self._ui.render_event(AgentWarningEvent(result.warning))
            if result.final_text is None:
                cancelled = await self._consume_turn(
                    result.display_text,
                    plan_only=self._runtime_state.plan_only,
                    cancellation=cancellation,
                )
                return _TurnOutcome(cancelled=cancelled)
            self._session_manager.append(
                (
                    UserMessage(result.display_text),
                    AssistantMessage((TextBlock(result.final_text),)),
                )
            )
            self._ui.render_event(FinalReplyEvent(result.final_text, 1))
            return _TurnOutcome()
        except (KeyboardInterrupt, SystemExit) as exc:
            return _TurnOutcome(failure=exc)
        except MyCodeError as exc:
            return _TurnOutcome(failure=exc)

    def _start_skill(
        self,
        submission: SkillSubmission,
    ) -> tuple[asyncio.Task[_TurnOutcome], CancellationToken]:
        """启动一条动态 Skill 命令的前台任务。

        Args:
            submission: handler 已按当前定义区分好的 inline 或 fork 提交。

        Returns:
            正在执行的任务和 Ctrl+C 可以触发的取消令牌。
        """

        cancellation = CancellationToken()
        self._ui.begin_turn(submission.display_text)
        self._ui.set_busy(True)
        task = asyncio.create_task(
            self._consume_skill_safely(submission, cancellation)
        )
        return task, cancellation

    async def _finish_turn(
        self,
        task: asyncio.Task[_TurnOutcome],
    ) -> CommandResult | None:
        """收尾一个 Agent 或命令任务，并返回可继续处理的命令结果。

        Args:
            task: 已完成或即将完成的统一应用任务。

        Returns:
            命令任务的退出或 Agent 提交请求；普通回合和失败返回 None。
        """

        command_result: CommandResult | None = None
        try:
            outcome = await task
            if outcome.cancelled or isinstance(
                outcome.failure,
                (KeyboardInterrupt, SystemExit),
            ):
                self._ui.show_status("本轮已取消")
            elif outcome.command_result is not None:
                command_result = outcome.command_result
            elif outcome.failure is not None:
                if isinstance(outcome.failure, MyCodeError):
                    self._ui.show_error(
                        redact_secrets(str(outcome.failure), self._secrets)
                    )
                else:
                    self._ui.show_error("恢复会话失败，当前会话保持不变")
        except asyncio.CancelledError:
            self._ui.show_status("本轮已取消")
        except KeyboardInterrupt:
            self._ui.show_status("本轮已取消")
        except MyCodeError as exc:
            self._ui.show_error(
                redact_secrets(str(exc), self._secrets)
            )
        except Exception:
            self._ui.show_error("发生未预期错误，本轮已终止")
        finally:
            handing_off = (
                command_result is not None
                and (
                    command_result.agent_submission is not None
                    or command_result.skill_submission is not None
                )
            )
            if not handing_off:
                self._ui.end_turn()
                self._ui.set_busy(False)
        return command_result

    async def _run_async(self) -> int:
        initial_permission = self._permission_controller.mode
        self._ui.configure(self._provider_config, initial_permission)
        self._ui.set_plan_mode(self._runtime_state.plan_only)
        worktree_cleanup_started = False
        if self._worktree_manager is not None:
            try:
                recovery = await self._worktree_manager.start(
                    resumed_session_id=self._resume_session_id,
                )
            except Exception as exc:
                self._ui.show_error(f"Worktree 状态启动失败：{exc}")
            else:
                for warning in recovery.warnings:
                    self._ui.show_error(f"Worktree 警告：{warning}")
                for interrupted in recovery.interrupted_tasks:
                    self._ui.show_status(
                        "已保留上次中断的子任务 Worktree："
                        f"{interrupted.worktree_name} ({interrupted.path})"
                    )
                if self._task_manager is not None:
                    self._task_manager.restore_interrupted(
                        recovery.interrupted_tasks
                    )
                if self._resume_session_id is None:
                    try:
                        resolution = (
                            await self._worktree_manager.resolve_session_binding(
                                self._session_manager.current_id
                            )
                        )
                        warnings = await self._agent.activate_workspace(
                            resolution.assignment
                        )
                    except Exception as exc:
                        self._ui.show_error(
                            f"当前目录未启用 Worktree 管理：{exc}"
                        )
                    else:
                        for warning in (*resolution.warnings, *warnings):
                            self._ui.show_error(f"Worktree 警告：{warning}")
                if recovery.state_trusted and self._worktree_cleanup is not None:
                    try:
                        await self._worktree_cleanup.start()
                    except Exception as exc:
                        self._ui.show_error(f"Worktree 后台清理启动失败：{exc}")
                    else:
                        worktree_cleanup_started = True
        if self._memory_worker is not None:
            self._memory_worker.start()
        if self._mcp_manager is not None:
            try:
                assert self._mcp_registry is not None
                startup_report = await self._mcp_manager.start(
                    self._mcp_registry
                )
            except Exception:
                self._ui.show_error(
                    "MCP 工具启动失败，将继续使用当前可用工具"
                )
            else:
                for issue in startup_report.issues:
                    self._ui.show_error(f"MCP 警告：{issue.message}")

        skill_startup_error: str | None = None
        if self._skill_service is not None:
            try:
                self._skill_service.scan_and_install()
            except MyCodeError as exc:
                skill_startup_error = redact_secrets(
                    str(exc),
                    self._secrets,
                )

        agent_startup_notices: list[str] = []
        if self._agent_service is not None and self._mcp_registry is not None:
            report = self._agent_service.initialize_catalog(
                self._mcp_registry.registered_names
            )
            agent_startup_notices.extend(
                f"Agent 定义警告：{item.path}：{item.message}"
                for item in report.diagnostics
            )

        await self._agent.start_hooks()

        if self._resume_session_id is not None:
            try:
                restored = await self._agent.restore_session(
                    self._resume_session_id,
                    CancellationToken(),
                )
            except Exception as exc:
                self._ui.show_error(
                    "启动恢复会话失败："
                    + redact_secrets(str(exc), self._secrets)
                )
            else:
                self._ui.clear_transcript()
                self._ui.show_status(f"已恢复会话：{restored.session_id}")
                for warning in restored.worktree_warnings:
                    self._ui.show_error(f"Worktree 恢复警告：{warning}")

        if self._team_service is not None and self._resume_session_id is None:
            try:
                _, team_reports = await self._team_service.restore_for_lead()
            except Exception as exc:
                self._ui.show_error(
                    "Agent Team 恢复失败："
                    + redact_secrets(str(exc), self._secrets)
                )
            else:
                for report in team_reports:
                    self._ui.show_status(f"Agent Team：{report}")

        ui_task = asyncio.create_task(self._ui.run_async())
        if skill_startup_error is not None:
            self._ui.show_error(
                f"Skill 启动校验失败：{skill_startup_error}"
            )
        for notice in self._startup_notices:
            self._ui.show_status(notice)
        for notice in agent_startup_notices:
            self._ui.show_error(notice)
        action_task: asyncio.Task[UIAction] | None = asyncio.create_task(
            self._ui.next_action()
        )
        memory_status_task = (
            asyncio.create_task(self._memory_worker.next_status())
            if self._memory_worker is not None
            else None
        )
        task_notification_task = (
            asyncio.create_task(self._task_manager.next_notification())
            if self._task_manager is not None
            else None
        )
        turn_task: asyncio.Task[_TurnOutcome] | None = None
        cancellation: CancellationToken | None = None
        exit_code = 2 if skill_startup_error is not None else 0
        should_exit = skill_startup_error is not None

        try:
            while not should_exit:
                waiters: set[asyncio.Task[object]] = {
                    ui_task,  # type: ignore[arg-type]
                }
                if action_task is not None:
                    waiters.add(action_task)  # type: ignore[arg-type]
                if turn_task is not None:
                    waiters.add(turn_task)  # type: ignore[arg-type]
                if memory_status_task is not None:
                    waiters.add(memory_status_task)  # type: ignore[arg-type]
                if task_notification_task is not None:
                    waiters.add(task_notification_task)  # type: ignore[arg-type]
                done, _ = await asyncio.wait(
                    waiters,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if ui_task in done:
                    try:
                        await ui_task
                    except Exception:
                        self._ui.show_error(
                            "终端界面发生错误，应用正在退出"
                        )
                        exit_code = 1
                    should_exit = True

                if turn_task is not None and turn_task in done:
                    command_result = await self._finish_turn(turn_task)
                    turn_task = None
                    cancellation = None
                    if command_result is not None:
                        if command_result.exit_requested:
                            should_exit = True
                        elif command_result.agent_submission is not None:
                            submission = command_result.agent_submission
                            turn_task, cancellation = self._start_turn(
                                submission.prompt,
                                display_text=submission.display_text,
                                plan_only=submission.plan_only,
                            )
                        elif command_result.skill_submission is not None:
                            turn_task, cancellation = self._start_skill(
                                command_result.skill_submission
                            )
                    if turn_task is None:
                        pending_turn = self._start_pending_notification_turn()
                        if pending_turn is not None:
                            turn_task, cancellation = pending_turn

                if (
                    task_notification_task is not None
                    and task_notification_task in done
                ):
                    try:
                        notification = task_notification_task.result()
                    except asyncio.CancelledError:
                        task_notification_task = None
                    except Exception:
                        self._ui.render_event(
                            AgentWarningEvent("无法读取后台子 Agent 完成通知")
                        )
                        task_notification_task = None
                    else:
                        assert self._task_manager is not None
                        task_notification_task = asyncio.create_task(
                            self._task_manager.next_notification()
                        )
                        if (
                            notification.session_id
                            == self._session_manager.current_id
                        ):
                            if turn_task is not None:
                                if self._notification_inbox is not None:
                                    self._notification_inbox.put(notification)
                            else:
                                message = notification_message(notification)
                                turn_task, cancellation = self._start_turn(
                                    message.content,
                                    show_user_input=False,
                                )

                if (
                    memory_status_task is not None
                    and memory_status_task in done
                ):
                    try:
                        status = memory_status_task.result()
                    except asyncio.CancelledError:
                        memory_status_task = None
                    except Exception:
                        self._ui.render_event(
                            AgentWarningEvent("无法读取自动笔记后台状态")
                        )
                        memory_status_task = None
                    else:
                        self._show_memory_status(status)
                        memory_status_task = asyncio.create_task(
                            self._memory_worker.next_status()
                        )

                if action_task is not None and action_task in done:
                    try:
                        action = action_task.result()
                    except asyncio.CancelledError:
                        action = UIAction(UIActionKind.EXIT)
                    except Exception:
                        self._ui.show_error(
                            "终端输入发生错误，应用正在退出"
                        )
                        action = UIAction(UIActionKind.EXIT)
                        exit_code = 1
                    action_task = None

                    if action.kind is UIActionKind.EXIT:
                        should_exit = True
                    elif action.kind is UIActionKind.CANCEL:
                        if cancellation is not None:
                            cancellation.cancel()
                        else:
                            self._ui.show_status("当前没有运行中的请求")
                    elif action.kind is UIActionKind.ADOPT_BACKGROUND:
                        if (
                            self._agent_service is not None
                            and self._agent_service.request_foreground_adoption()
                        ):
                            self._ui.show_status(
                                "正在把前台子 Agent 移交后台，原任务会继续运行"
                            )
                        else:
                            self._ui.show_status(
                                "当前没有可移交到后台的子 Agent"
                            )
                    elif action.kind is UIActionKind.TOGGLE_PLAN:
                        if turn_task is None:
                            self._set_plan_only(
                                not self._runtime_state.plan_only
                            )
                        else:
                            self._ui.show_status(
                                "当前轮次执行中，暂不能切换 Plan 模式"
                            )
                    elif action.kind is UIActionKind.SUBMIT:
                        if turn_task is not None:
                            self._ui.show_status(
                                "当前轮次尚未结束，请稍候"
                            )
                        else:
                            assert action.text is not None
                            stripped = action.text.strip()
                            if stripped:
                                parsed = parse_command(stripped)
                                if parsed is None:
                                    turn_task, cancellation = self._start_turn(
                                        stripped
                                    )
                                else:
                                    turn_task, cancellation = self._start_command(
                                        parsed
                                    )

                if (
                    not should_exit
                    and action_task is None
                ):
                    action_task = asyncio.create_task(
                        self._ui.next_action()
                    )
        finally:
            if action_task is not None and not action_task.done():
                action_task.cancel()
                await asyncio.gather(
                    action_task,
                    return_exceptions=True,
                )
            if turn_task is not None:
                if cancellation is not None:
                    cancellation.cancel()
                await self._finish_turn(turn_task)
            if memory_status_task is not None:
                if memory_status_task.done() and not memory_status_task.cancelled():
                    try:
                        final_status = memory_status_task.result()
                    except Exception:
                        pass
                    else:
                        self._show_memory_status(final_status)
                else:
                    memory_status_task.cancel()
                    await asyncio.gather(
                        memory_status_task,
                        return_exceptions=True,
                    )
            if task_notification_task is not None:
                task_notification_task.cancel()
                await asyncio.gather(
                    task_notification_task,
                    return_exceptions=True,
                )
            if self._memory_worker is not None:
                try:
                    memory_finished = await self._memory_worker.close()
                except Exception:
                    self._ui.show_error("关闭自动笔记后台任务时发生错误")
                    exit_code = 1
                else:
                    if not memory_finished:
                        self._ui.show_error(
                            "自动笔记任务等待 10 秒后仍未完成，"
                            "部分记忆可能尚未保存"
                        )
                    while memory_finished:
                        try:
                            pending_status = await asyncio.wait_for(
                                self._memory_worker.next_status(),
                                timeout=0.001,
                            )
                        except TimeoutError:
                            break
                        self._show_memory_status(pending_status)
            if worktree_cleanup_started and self._worktree_cleanup is not None:
                try:
                    await self._worktree_cleanup.close()
                except Exception:
                    self._ui.show_error("关闭 Worktree 后台清理时发生错误")
                    exit_code = 1
            if self._task_manager is not None:
                try:
                    await self._task_manager.close()
                except Exception:
                    self._ui.show_error("关闭后台子 Agent 任务时发生错误")
                    exit_code = 1
            if self._team_service is not None:
                try:
                    await self._team_service.close_local_hosts()
                except Exception:
                    self._ui.show_error("暂停同进程团队成员时发生错误")
                    exit_code = 1
            if self._worktree_manager is not None:
                try:
                    await self._worktree_manager.close()
                except Exception:
                    self._ui.show_error("关闭 Worktree Manager 时发生错误")
                    exit_code = 1
            try:
                await self._agent.shutdown_hooks()
            except Exception:
                self._ui.show_error("关闭 Hook 资源时发生错误")
                exit_code = 1
            try:
                self._session_manager.close()
            except Exception:
                self._ui.show_error("关闭当前会话文件时发生错误")
                exit_code = 1
            try:
                close_agent = getattr(self._agent, "close", None)
                if callable(close_agent):
                    close_agent()
            except Exception:
                self._ui.show_error("清理当前会话 artifact 时发生错误")
                exit_code = 1
            self._ui.stop()
            if not ui_task.done():
                await asyncio.gather(ui_task, return_exceptions=True)
            if self._mcp_manager is not None:
                try:
                    close_report = await self._mcp_manager.aclose()
                except Exception:
                    self._ui.show_error("关闭 MCP 资源时发生错误")
                    exit_code = 1
                else:
                    if close_report.issues:
                        self._ui.show_error("部分 MCP 资源关闭失败")
                        exit_code = 1
            try:
                await self._transport.aclose()
            except Exception:
                self._ui.show_error("关闭网络资源时发生错误")
                exit_code = 1
        return exit_code

    def run(self) -> int:
        try:
            return asyncio.run(self._run_async())
        except KeyboardInterrupt:
            # 全屏模式会把 Ctrl+C 转换为动作；该分支只保护不支持按键绑定
            # 的兼容终端，资源清理由 _run_async 的 finally 完成。
            return 0
