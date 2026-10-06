"""运行一个长期成员的事件循环，并在每轮边界消费持久化邮箱。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime

from mycode.models.teams import (
    TeamActorContext,
    TeamMessageKind,
    TeammateState,
)
from mycode.teams.backends.base import TeammateLaunch, WakeWaiter
from mycode.teams.mailbox import TeamMailbox
from mycode.teams.runtime import TeamRuntimeLoader
from mycode.teams.store import TeamStateStore
from mycode.teams.tasks import TeamTaskBoard


_MAILBOX_POLL_SECONDS = 0.2


class TeammateHost:
    """代表一个成员进程内持续存在、但只在收到事件时调用模型的循环。

    Attributes:
        store: 校验 generation、租约并写成员状态的团队 Store。
        mailbox: 读取和确认成员邮箱的持久化入口。
        runtime_loader: 按 team/member 恢复成员会话与 Agent 组件的函数。
    """

    def __init__(
        self,
        store: TeamStateStore,
        mailbox: TeamMailbox,
        runtime_loader: TeamRuntimeLoader,
    ) -> None:
        """保存成员 Host 运行所需的三个生产组件。

        Args:
            store: 团队和成员状态持久化入口。
            mailbox: 成员点对点消息和 cursor 入口。
            runtime_loader: 恢复该成员完整对话和执行组件的异步函数。

        Returns:
            不返回数据；调用实例时才开始事件循环。
        """

        self.store = store
        self.mailbox = mailbox
        self.runtime_loader = runtime_loader
        self.tasks = TeamTaskBoard(store)

    async def __call__(
        self,
        launch: TeammateLaunch,
        wait_for_wake: WakeWaiter,
    ) -> None:
        """恢复成员并处理首次提示、邮箱消息和后续显式唤醒。

        Args:
            launch: Supervisor 生成的 team/member/generation、租约和工作区。
            wait_for_wake: 在成员空闲时等待一次唤醒的异步函数。
                独立进程 Host 传入等待标准输入的函数。

        Returns:
            收到 shutdown_request 或 task 被取消后结束，不删除会话和成员。
        """

        actor = TeamActorContext(
            team_id=launch.team_id,
            actor_id=launch.agent_id,
            actor_kind="member",
            generation=launch.generation,
        )
        runtime = None
        wake_task: asyncio.Task[None] | None = None
        try:
            # 在恢复对话前先验证租约，避免旧 Host 读取并继续写入新一代会话。
            self.store.update_member(
                actor,
                launch.agent_id,
                lambda member: replace(
                    member, state=TeammateState.STARTING, updated_at=_now()
                ),
                lease_token=launch.lease_token,
            )
            runtime = await self.runtime_loader(launch.team_id, launch.agent_id)
            self.store.update_member(
                actor,
                launch.agent_id,
                lambda member: replace(
                    member, state=TeammateState.RUNNING, updated_at=_now(), last_active_at=_now()
                ),
                lease_token=launch.lease_token,
            )
            # 新建或恢复成员时也检查已有任务，覆盖任务先于成员创建的情况。
            if not self.tasks.pending_scans(actor):
                available = tuple(view.task.task_id for view in self.tasks.list(actor) if view.claimable)
                if available:
                    self.tasks.open_scan(actor.team_id, available, (actor.actor_id,))
            pending_prompt = launch.prompt.strip()
            first_prompt_pending = bool(pending_prompt)
            while True:
                messages = self.mailbox.read_unread(actor)
                scans = self.tasks.pending_scans(actor)
                if not pending_prompt and not messages and not scans:
                    self.store.update_member(
                        actor,
                        launch.agent_id,
                        lambda member: replace(
                            member, state=TeammateState.IDLE, updated_at=_now()
                        ),
                        lease_token=launch.lease_token,
                    )
                    if wake_task is None:
                        wake_task = asyncio.create_task(wait_for_wake())
                    while True:
                        # 跨进程发送者可能没有后端回调；邮箱中的 wake 消息
                        # 保留了通知。超时检查不取消已有的标准输入等待。
                        done, _ = await asyncio.wait(
                            (wake_task,), timeout=_MAILBOX_POLL_SECONDS
                        )
                        if done:
                            completed_wait = wake_task
                            wake_task = None
                            completed_wait.result()
                            messages = self.mailbox.read_unread(actor)
                            break
                        messages = self.mailbox.read_unread(actor)
                        scans = self.tasks.pending_scans(actor)
                        if scans or any(message.wake for message in messages):
                            break
                    scans = self.tasks.pending_scans(actor)
                    if not messages and not scans:
                        continue
                shutdown = any(
                    item.kind is TeamMessageKind.SHUTDOWN_REQUEST for item in messages
                )
                parts = [pending_prompt] if pending_prompt else []
                parts.extend(
                    f"团队消息[{item.summary}]：{item.body}" for item in messages
                )
                if not shutdown:
                    available = {view.task.task_id for view in self.tasks.list(actor) if view.claimable}
                    for scan in scans:
                        candidates = tuple(task_id for task_id in scan.task_ids if task_id in available)
                        if candidates:
                            parts.append(
                                f"任务认领检查：round_id={scan.round_id}，候选任务={', '.join(candidates)}。"
                                "请先用 TeamTaskList/TeamTaskGet 查看任务，按角色和优先级选择合适任务，"
                                "调用 TeamTaskClaim（带上本轮 round_id）成功后再执行。"
                                "如果认领失败，刷新任务列表，不要执行未认领任务。"
                                "每次只能持有一个 working 任务；无合适任务则说明原因后待命。"
                            )
                pending_prompt = ""
                if parts:
                    self.store.update_member(
                        actor,
                        launch.agent_id,
                        lambda member: replace(
                            member, state=TeammateState.RUNNING, updated_at=_now(), last_active_at=_now()
                        ),
                        lease_token=launch.lease_token,
                    )
                    await runtime.run("\n\n".join(parts))
                    if first_prompt_pending:
                        # 只有模型回合成功结束后才清空落盘指令。Host 在此前
                        # 崩溃时，恢复流程仍能重新取得原始任务说明。
                        self.store.clear_runtime_prompt(
                            launch.team_id,
                            launch.agent_id,
                        )
                        first_prompt_pending = False
                    self.mailbox.acknowledge(actor, messages)
                if not shutdown:
                    for scan in scans:
                        self.tasks.finish_scan(scan.round_id, actor.actor_id, team_id=actor.team_id)
                self.store.update_member(
                    actor,
                    launch.agent_id,
                    lambda member: replace(
                        member, state=TeammateState.IDLE, updated_at=_now(), last_active_at=_now()
                    ),
                    lease_token=launch.lease_token,
                )
                if shutdown:
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            try:
                self.store.update_member(
                    actor,
                    launch.agent_id,
                    lambda member: replace(
                        member, state=TeammateState.FAILED, updated_at=_now()
                    ),
                    lease_token=launch.lease_token,
                )
            except Exception:
                pass
            raise
        finally:
            if wake_task is not None:
                wake_task.cancel()
                await asyncio.gather(wake_task, return_exceptions=True)
            if runtime is not None:
                runtime.close()


def _now() -> datetime:
    """返回带本地时区的当前时间，供成员状态写入使用。

    Returns:
        可以跨进程序列化和比较的带时区 ``datetime``。
    """

    return datetime.now().astimezone()
