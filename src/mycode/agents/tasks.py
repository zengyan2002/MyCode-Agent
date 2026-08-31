"""管理后台子 Agent 的 FIFO 队列、状态查询、取消和完成通知。"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import datetime

from mycode.agent.cancellation import CancellationToken
from mycode.agents.runtime import AgentRunHandle, IndependentAgentRuntimeBuilder
from mycode.models.agents import (
    AgentRunResult,
    AgentUsage,
    BackgroundTaskRecord,
    BackgroundTaskSnapshot,
    BackgroundTaskStatus,
    IndependentAgentSpec,
    TaskMetadata,
    TaskNotification,
)
from mycode.models.messages import UserMessage
from mycode.models.worktrees import (
    InterruptedTaskSummary,
    WorkspaceAssignment,
    WorkspaceIsolationMode,
    WorktreeFinishAction,
    WorktreeFinishReport,
    WorktreeTaskOutcome,
)


class TaskNotificationInbox:
    """按主会话保存尚未进入 Provider 请求的后台任务通知。

    Attributes:
        _pending: 键是主会话 ID，值是按任务完成到达顺序排列的通知队列。
            队列只存在于当前进程内，不会写入会话旁路文件。
    """

    def __init__(self) -> None:
        """创建尚无待处理通知的会话收件箱。

        Returns:
            不返回数据；后续由 ChatApplication 放入通知、AgentTurnRunner
            在 Provider 请求边界排空。
        """

        self._pending: dict[str, deque[TaskNotification]] = defaultdict(deque)

    def put(self, notification: TaskNotification) -> None:
        """把一条完成通知放到其所属会话队尾。

        Args:
            notification: TaskManager 已脱敏且带终态用量的通知。

        Returns:
            不返回数据；同一会话中的到达顺序保持不变。
        """

        self._pending[notification.session_id].append(notification)

    def drain_messages(self, session_id: str) -> tuple[UserMessage, ...]:
        """排空一个会话的通知并转换成合法用户消息。

        Args:
            session_id: 即将发送 Provider 请求的主会话 ID。

        Returns:
            每条通知对应一个 ``<task-notification>`` UserMessage；没有待处理
            通知时返回空元组。读取后这些通知不会再次返回。
        """

        queue = self._pending.pop(session_id, deque())
        return tuple(notification_message(item) for item in queue)

    def has_pending(self, session_id: str) -> bool:
        """判断指定会话是否有尚未消费的通知。

        Args:
            session_id: 当前或待恢复的主会话 ID。

        Returns:
            队列非空时返回 ``True``，否则返回 ``False``。
        """

        return bool(self._pending.get(session_id))

    def clear_session(self, session_id: str) -> None:
        """丢弃一个已销毁会话尚未送入模型的全部通知。

        Args:
            session_id: 被清空、切换或关闭的会话 ID。

        Returns:
            不返回数据；其他会话的通知保持不变。
        """

        self._pending.pop(session_id, None)


def notification_message(notification: TaskNotification) -> UserMessage:
    """把后台终态和用量写成主模型可解释的用户消息。

    Args:
        notification: 已由 TaskManager 脱敏的完成通知。

    Returns:
        正文包在 ``<task-notification>`` 标签中的 UserMessage，不含中间
        模型消息、thinking 或工具输出。
    """

    usage = notification.usage
    cached = (
        "未报告"
        if usage.cached_input_tokens is None
        else str(usage.cached_input_tokens)
    )
    workspace_lines: tuple[str, ...] = ()
    if notification.workspace_report is not None:
        report = notification.workspace_report
        workspace_lines = (
            "workspace:",
            f"  path: {report.workspace.root}",
            f"  branch: {report.workspace.branch or 'detached'}",
            f"  action: {report.action.value}",
            f"  reason: {report.reason}",
        )
    return UserMessage(
        "\n".join(
            (
                "<task-notification>",
                f"task_id: {notification.task_id}",
                f"status: {notification.status.value}",
                "result:",
                notification.result_text,
                "usage:",
                f"  model_calls: {usage.model_calls}",
                f"  input_tokens: {usage.input_tokens}",
                f"  cached_input_tokens: {cached}",
                f"  output_tokens: {usage.output_tokens}",
                f"  tool_calls: {usage.tool_calls}",
                f"  duration_ms: {usage.duration_ms}",
                *workspace_lines,
                "</task-notification>",
            )
        )
    )


class TaskManager:
    """在当前进程和会话内管理全部后台子 Agent。

    普通后台启动进入固定并发的 FIFO worker；前台运行移交时通过
    :meth:`adopt` 接管同一个 AgentRunHandle，不重启、不重新排队。

    Attributes:
        _builder: worker 启动排队 spec 时使用的独立运行装配器。
        _queue: 保存尚未开始的 ``(task_id, spec)``，按 FIFO 取出。
        _records: 当前进程内全部会话的可变任务记录。
        _handles: 已经运行且可以协作式取消的 AgentRunHandle。
        _notifications: 按任务到达终态的先后顺序保存的精简通知。
        _workers: 固定并发数量的后台队列消费者。
    """

    def __init__(
        self,
        builder: IndependentAgentRuntimeBuilder,
        *,
        max_concurrency: int = 4,
        sanitize: Callable[[str], str] | None = None,
    ) -> None:
        """创建任务管理器，但在首次 launch 前不启动 worker。

        Args:
            builder: 把冻结 spec 装配成真实独立运行器的 Builder。
            max_concurrency: 同时从 FIFO 队列启动的后台任务数量。
            sanitize: 通知写入结果前使用的统一脱敏函数；未传时原样保留。

        Returns:
            不返回数据；首次 ``launch`` 时才创建 worker。

        Raises:
            ValueError: 并发数不是正整数。
        """

        if (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency <= 0
        ):
            raise ValueError("后台任务并发数必须是正整数")
        self._builder = builder
        self._max_concurrency = max_concurrency
        self._sanitize = sanitize or (lambda text: text)
        self._records: dict[str, BackgroundTaskRecord] = {}
        self._handles: dict[str, AgentRunHandle] = {}
        # 保存尚未启动的 prepared spec，让排队取消时也能释放 Worktree 租约。
        self._queued_specs: dict[str, IndependentAgentSpec] = {}
        self._queue: asyncio.Queue[tuple[str, IndependentAgentSpec] | None] = (
            asyncio.Queue()
        )
        self._notifications: deque[TaskNotification] = deque()
        self._notification_ready = asyncio.Event()
        self._workers: list[asyncio.Task[None]] = []
        self._adopted_monitors: set[asyncio.Task[None]] = set()
        self._closed = False

    async def launch(
        self,
        spec: IndependentAgentSpec,
    ) -> BackgroundTaskSnapshot:
        """把一份尚未启动的 spec 放入后台 FIFO 队列。

        Args:
            spec: AgentService 已冻结的运行输入。

        Returns:
            状态为 ``queued`` 的初始任务快照。

        Raises:
            RuntimeError: 管理器已关闭或任务 ID 重复。
        """

        self._ensure_open()
        self._ensure_workers()
        if spec.run_id in self._records:
            raise RuntimeError(f"后台任务 ID 已存在：{spec.run_id}")
        created = datetime.now().astimezone()
        record = BackgroundTaskRecord(
            task_id=spec.run_id,
            name=spec.name,
            description=spec.description,
            source=spec.origin.value,
            session_id=spec.session_id,
            status=BackgroundTaskStatus.QUEUED,
            result=None,
            created_at=created,
            started_at=None,
            ended_at=None,
            cancellation=CancellationToken(),
        )
        self._records[spec.run_id] = record
        self._queued_specs[spec.run_id] = spec
        await self._queue.put((spec.run_id, spec))
        return self._snapshot(record)

    async def adopt(
        self,
        handle: AgentRunHandle,
        metadata: TaskMetadata,
    ) -> BackgroundTaskSnapshot:
        """接管一个已经在前台运行的句柄并让它在后台继续。

        Args:
            handle: 正在执行的原 AgentRunHandle。
            metadata: Task 列表所需的名字、说明、来源和会话 ID。

        Returns:
            状态为 ``running`` 的快照。接管不会占用 FIFO worker 名额。

        Raises:
            RuntimeError: 管理器关闭、ID 不一致或任务 ID 已存在。
        """

        self._ensure_open()
        if handle.run_id != metadata.task_id:
            raise RuntimeError("接管句柄 ID 与任务元数据不一致")
        if metadata.task_id in self._records:
            raise RuntimeError(f"后台任务 ID 已存在：{metadata.task_id}")
        now = datetime.now().astimezone()
        handle.move_to_background()
        record = BackgroundTaskRecord(
            task_id=metadata.task_id,
            name=metadata.name,
            description=metadata.description,
            source=metadata.source,
            session_id=metadata.session_id,
            status=BackgroundTaskStatus.RUNNING,
            result=None,
            created_at=now,
            started_at=now,
            ended_at=None,
            cancellation=handle.cancellation,
        )
        self._records[metadata.task_id] = record
        self._handles[metadata.task_id] = handle
        monitor = asyncio.create_task(self._monitor_handle(record, handle))
        self._adopted_monitors.add(monitor)
        monitor.add_done_callback(self._adopted_monitors.discard)
        return self._snapshot(record)

    def list(self, session_id: str) -> tuple[BackgroundTaskSnapshot, ...]:
        """列出一个主会话拥有的全部后台任务。

        Args:
            session_id: 当前主会话 ID。

        Returns:
            按创建时间和任务 ID 排序的不可变快照元组。
        """

        records = (
            record
            for record in self._records.values()
            if record.session_id == session_id
        )
        return tuple(
            self._snapshot(record)
            for record in sorted(
                records,
                key=lambda item: (item.created_at, item.task_id),
            )
        )

    def get(self, session_id: str, task_id: str) -> BackgroundTaskSnapshot:
        """查询当前会话中的一个后台任务。

        Args:
            session_id: 当前主会话 ID。
            task_id: Agent 工具启动时返回的任务 ID。

        Returns:
            查询时刻的不可变任务快照。

        Raises:
            KeyError: 任务不存在或属于其他会话。
        """

        record = self._records.get(task_id)
        if record is None or record.session_id != session_id:
            raise KeyError(f"当前会话不存在后台任务：{task_id}")
        return self._snapshot(record)

    def restore_interrupted(
        self,
        summaries: tuple[InterruptedTaskSummary, ...],
    ) -> tuple[BackgroundTaskSnapshot, ...]:
        """把上次进程遗留的子任务导入为只读 ``interrupted`` 终态。

        Args:
            summaries: WorktreeManager 启动时从可信状态文件恢复出的任务摘要。

        Returns:
            本次新导入的任务快照。已有同 ID 记录不会被覆盖，也不会重新排队
            或发起模型请求。

        Raises:
            ValueError: 摘要集合或其中元素类型无效。
        """

        if not isinstance(summaries, tuple) or not all(
            isinstance(item, InterruptedTaskSummary) for item in summaries
        ):
            raise ValueError("interrupted summaries 类型无效")
        restored: list[BackgroundTaskSnapshot] = []
        now = datetime.now().astimezone()
        for summary in summaries:
            task_id = summary.task_id or f"interrupted-{summary.worktree_name}"
            if task_id in self._records:
                continue
            assignment = WorkspaceAssignment(
                root=summary.path,
                isolation=WorkspaceIsolationMode.WORKTREE,
                worktree_name=summary.worktree_name,
                branch=summary.branch,
                base_commit=summary.base_commit,
                lease_id=f"interrupted:{task_id}",
            )
            report = WorktreeFinishReport(
                workspace=assignment,
                action=WorktreeFinishAction.RETAINED,
                terminal_status=WorktreeTaskOutcome.INTERRUPTED,
                changes=None,
                reason=summary.reason,
            )
            result = AgentRunResult(
                status=BackgroundTaskStatus.INTERRUPTED,
                final_text=None,
                partial_text=None,
                error=(
                    f"{summary.reason}；模型执行没有恢复。"
                    f"成果目录保留在 {summary.path}"
                ),
                usage=AgentUsage(),
                workspace_report=report,
            )
            record = BackgroundTaskRecord(
                task_id=task_id,
                name=summary.worktree_name,
                description="从上次进程恢复的中断子任务",
                source="recovery",
                session_id=summary.session_id,
                status=BackgroundTaskStatus.INTERRUPTED,
                result=result,
                created_at=now,
                started_at=now,
                ended_at=now,
                cancellation=CancellationToken(),
            )
            self._records[task_id] = record
            restored.append(self._snapshot(record))
            self._notifications.append(
                TaskNotification(
                    task_id=task_id,
                    session_id=summary.session_id,
                    status=BackgroundTaskStatus.INTERRUPTED,
                    result_text=self._sanitize(result.error or summary.reason),
                    usage=result.usage,
                    workspace_report=report,
                )
            )
        if restored:
            self._notification_ready.set()
        return tuple(restored)

    async def stop(
        self,
        session_id: str,
        task_id: str,
    ) -> BackgroundTaskSnapshot:
        """取消当前会话中排队或运行的后台任务。

        Args:
            session_id: 当前主会话 ID。
            task_id: 要停止的任务 ID。

        Returns:
            请求取消后的任务快照；已处于终态时原样返回。
        """

        record = self._record_for(session_id, task_id)
        if record.status.terminal:
            return self._snapshot(record)
        handle = self._handles.get(task_id)
        if handle is not None:
            handle.cancel()
            return self._snapshot(record)
        # 尚未被 worker 取走的条目不能从 asyncio.Queue 中间删除；先把记录
        # 设为取消，worker 取到后会跳过，不改变其他排队任务的相对顺序。
        spec = self._queued_specs.pop(task_id, None)
        workspace_report = (
            await self._builder.abandon(spec, "后台任务在开始前被取消")
            if spec is not None
            else None
        )
        result = AgentRunResult(
            BackgroundTaskStatus.CANCELLED,
            None,
            None,
            "后台任务在开始前被取消",
            AgentUsage(),
            workspace_report,
        )
        self._finish(record, result)
        await self._notify(record)
        return self._snapshot(record)

    async def cancel_session(self, session_id: str) -> None:
        """取消并移除一个主会话的全部后台任务。

        Args:
            session_id: 被清空、切换或关闭的主会话 ID。

        Returns:
            所有运行句柄结束、该会话记录移除后返回。
        """

        task_ids = [
            record.task_id
            for record in self._records.values()
            if record.session_id == session_id
        ]
        for task_id in task_ids:
            record = self._records[task_id]
            if not record.status.terminal:
                await self.stop(session_id, task_id)
        self._discard_queued_session(session_id)
        waits = [
            self._handles[task_id].task
            for task_id in task_ids
            if task_id in self._handles
        ]
        if waits:
            await asyncio.gather(*waits, return_exceptions=True)
        for task_id in task_ids:
            self._records.pop(task_id, None)
            self._handles.pop(task_id, None)
        self._notifications = deque(
            item
            for item in self._notifications
            if item.session_id != session_id
        )
        if self._notifications:
            self._notification_ready.set()
        else:
            self._notification_ready.clear()

    async def next_notification(self) -> TaskNotification:
        """等待下一条后台任务完成通知。

        Returns:
            含终态、脱敏结果和用量的 TaskNotification。
        """

        while True:
            if self._notifications:
                notification = self._notifications.popleft()
                if not self._notifications:
                    self._notification_ready.clear()
                return notification
            await self._notification_ready.wait()

    async def close(self) -> None:
        """取消全部任务、停止 worker，并释放进程内记录。

        Returns:
            所有运行和监控协程结束后返回；不会写持久化文件。
        """

        if self._closed:
            return
        self._closed = True
        sessions = {record.session_id for record in self._records.values()}
        for session_id in sessions:
            await self.cancel_session(session_id)
        for _ in self._workers:
            await self._queue.put(None)
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        if self._adopted_monitors:
            await asyncio.gather(
                *tuple(self._adopted_monitors),
                return_exceptions=True,
            )
        self._workers.clear()
        self._records.clear()
        self._handles.clear()
        self._queued_specs.clear()
        self._notifications.clear()
        self._notification_ready.clear()

    def _ensure_open(self) -> None:
        """拒绝在管理器关闭后创建或接管新的后台任务。

        Returns:
            管理器仍可用时不返回数据。

        Raises:
            RuntimeError: ``close`` 已经开始或完成。
        """

        if self._closed:
            raise RuntimeError("后台任务管理器已经关闭")

    def _ensure_workers(self) -> None:
        """在首次排队时创建配置数量的 FIFO worker。

        Returns:
            不返回数据；worker 已存在时不重复创建。
        """

        if not self._workers:
            self._workers = [
                asyncio.create_task(self._worker())
                for _ in range(self._max_concurrency)
            ]

    def _discard_queued_session(self, session_id: str) -> None:
        """从尚未启动的 FIFO 条目中移除指定会话的冻结 spec。

        Args:
            session_id: 已清空、切换或关闭的主会话 ID。

        Returns:
            不返回数据；其他会话条目的相对顺序保持不变。
        """

        retained: list[tuple[str, IndependentAgentSpec] | None] = []
        while True:
            try:
                queued = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._queue.task_done()
            if queued is None or queued[1].session_id != session_id:
                retained.append(queued)
        for queued in retained:
            self._queue.put_nowait(queued)

    async def _worker(self) -> None:
        """按 FIFO 取出 spec，启动运行器并等待结束。

        Returns:
            收到 ``close`` 放入的停止标记后返回，不产生业务结果。
        """

        while True:
            queued = await self._queue.get()
            try:
                if queued is None:
                    return
                task_id, spec = queued
                self._queued_specs.pop(task_id, None)
                record = self._records.get(task_id)
                if record is None or record.status is not BackgroundTaskStatus.QUEUED:
                    continue
                try:
                    runner = self._builder.build(spec)
                    handle = runner.start()
                except Exception as exc:
                    try:
                        workspace_report = await self._builder.abandon(
                            spec,
                            f"后台 Runner 装配失败：{exc}",
                        )
                    except Exception:
                        workspace_report = None
                    result = AgentRunResult(
                        BackgroundTaskStatus.FAILED,
                        None,
                        None,
                        str(exc) or type(exc).__name__,
                        AgentUsage(),
                        workspace_report,
                    )
                    self._finish(record, result)
                    await self._notify(record)
                    continue
                record.status = BackgroundTaskStatus.RUNNING
                record.started_at = datetime.now().astimezone()
                record.cancellation = handle.cancellation
                self._handles[task_id] = handle
                await self._monitor_handle(record, handle)
            finally:
                self._queue.task_done()

    async def _monitor_handle(
        self,
        record: BackgroundTaskRecord,
        handle: AgentRunHandle,
    ) -> None:
        """等待一个运行句柄，并把结果写入记录和通知队列。

        Args:
            record: 需要更新的 TaskManager 内部记录。
            handle: worker 启动或前台移交的原运行句柄。

        Returns:
            句柄结束、记录更新且终态通知入队后返回。
        """

        result = await handle.wait()
        self._finish(record, result)
        self._handles.pop(record.task_id, None)
        await self._notify(record)

    def _finish(
        self,
        record: BackgroundTaskRecord,
        result: AgentRunResult,
    ) -> None:
        """把一次子 Agent 的终态、结果和结束时间写回任务记录。

        Args:
            record: TaskManager 内与运行 ID 对应的可变记录。
            result: 子 Agent 返回的完成、失败或取消结果。

        Returns:
            不返回数据；调用后 ``record`` 已处于终态。
        """

        record.status = result.status
        record.result = result
        record.ended_at = datetime.now().astimezone()
        if record.started_at is None:
            record.started_at = record.created_at

    async def _notify(self, record: BackgroundTaskRecord) -> None:
        """为一个终态任务生成脱敏通知并按到达顺序入队。

        Args:
            record: 已经写入 ``AgentRunResult`` 的终态任务记录。

        Returns:
            通知进入应用监听队列后返回，不等待主 Agent 消费。
        """

        assert record.result is not None
        result = record.result
        text = result.final_text or result.partial_text or result.error or "任务结束"
        self._notifications.append(
            TaskNotification(
                task_id=record.task_id,
                session_id=record.session_id,
                status=record.status,
                result_text=self._sanitize(text),
                usage=result.usage,
                workspace_report=result.workspace_report,
            )
        )
        self._notification_ready.set()

    def _record_for(self, session_id: str, task_id: str) -> BackgroundTaskRecord:
        """取得当前会话拥有的内部任务记录。

        Args:
            session_id: 发起查询或取消的主会话 ID。
            task_id: Agent 工具返回的后台任务 ID。

        Returns:
            TaskManager 内部可变的 ``BackgroundTaskRecord``。

        Raises:
            KeyError: 任务不存在或属于其他会话。
        """

        record = self._records.get(task_id)
        if record is None or record.session_id != session_id:
            raise KeyError(f"当前会话不存在后台任务：{task_id}")
        return record

    @staticmethod
    def _snapshot(record: BackgroundTaskRecord) -> BackgroundTaskSnapshot:
        """复制可变内部记录为调用方不能修改的快照。

        Args:
            record: TaskManager 当前保存的可变记录。

        Returns:
            包含同一时刻字段值的 BackgroundTaskSnapshot。
        """

        return BackgroundTaskSnapshot(
            task_id=record.task_id,
            name=record.name,
            description=record.description,
            source=record.source,
            session_id=record.session_id,
            status=record.status,
            result=record.result,
            created_at=record.created_at,
            started_at=record.started_at,
            ended_at=record.ended_at,
        )
