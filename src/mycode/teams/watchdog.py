"""Lead 独立续租，成员在模型和工具执行期间也检查租约。"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
from dataclasses import asdict
from datetime import datetime

from mycode.models.teams import TeamActorContext, TeamWatchdogSettings
from mycode.teams.backends.base import TeammateLaunch
from mycode.teams.locks import ExclusiveFileLock
from mycode.teams.store import TeamStateStore, TeamStoreError


class LeadHeartbeat:
    """持有一个团队、一个 Lead 代数的运行锁与心跳后台任务。"""
    def __init__(self, store: TeamStateStore, actor: TeamActorContext, settings: TeamWatchdogSettings):
        self.store, self.actor, self.settings = store, actor, settings
        self.owner_instance_id = secrets.token_hex(16)
        self.lead_session_id = store.load_team(actor.team_id).team.lead_session_id
        self.sequence = 0
        self.lock = ExclusiveFileLock(store.team_dir(actor.team_id) / "locks" / f"lead-runtime-{actor.generation}.lock",
            self.owner_instance_id, max_attempts=1)
        self.task: asyncio.Task | None = None

    def payload(self, *, released=False, reason=None):
        return {"lead_session_id": self.lead_session_id, "lead_generation": self.actor.generation,
            "owner_instance_id": self.owner_instance_id, "sequence": self.sequence,
            "updated_at": datetime.now().astimezone().isoformat(), "released": released,
            "release_reason": reason, "settings": asdict(self.settings)}

    def start(self) -> None:
        """取得本代数运行锁、发布第一份心跳，然后独立续租。"""
        self.lock.acquire()
        try:
            self.store.publish_lead_runtime(self.actor, self.payload(), claim=True)
            self.task = asyncio.create_task(self._run())
        except BaseException:
            self.lock.release()
            raise

    async def _run(self):
        while True:
            await asyncio.sleep(self.settings.heartbeat_interval)
            self.sequence += 1
            try:
                self.store.publish_lead_runtime(self.actor, self.payload())
            except TeamStoreError:
                # 未写成功的心跳不延长成员租约；暂时 I/O 故障可以在下轮恢复。
                logging.getLogger(__name__).warning("Lead 心跳未能续租，成员将按期限自行停止")

    async def close(self, reason="application_shutdown") -> None:
        """停止续租并发布释放意图；身份已更换时不覆盖新 owner。"""
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        try:
            self.store.publish_lead_runtime(self.actor, self.payload(released=True, reason=reason))
        except TeamStoreError:
            pass
        finally:
            self.lock.release()


class MemberWatchdog:
    """观察固定 launch 身份的心跳推进，用本进程单调时间判断租约。"""
    def __init__(self, store: TeamStateStore, launch: TeammateLaunch):
        self.store, self.launch = store, launch
        self.ready = asyncio.Event()
        self.reason: str | None = None

    def identity_reason(self, member, owner):
        launch = self.launch
        if (member.runtime_generation != launch.generation or member.lease_token_hash !=
                hashlib.sha256(launch.lease_token.encode()).hexdigest()):
            return "member_lease_revoked"
        if member.stop_request_generation == launch.generation:
            return member.last_stop_reason or "explicit_stop"
        if not launch.owner_instance_id:
            return "member_lease_revoked"
        if owner:
            if (owner.get("owner_instance_id") != launch.owner_instance_id
                    or owner.get("lead_generation") != launch.owner_lead_generation
                    or owner.get("lead_session_id") != launch.owner_lead_session_id):
                return "lead_replaced"
            if owner.get("released"):
                return owner.get("release_reason") or "application_shutdown"
        return None

    async def run(self) -> str:
        settings = self.launch.watchdog
        loop = asyncio.get_running_loop()
        started = advanced = loop.time()
        sequence = None
        while True:
            now = loop.time()
            try:
                member = self.store.read_member(self.launch.team_id, self.launch.agent_id)
                owner = self.store.read_lead_runtime(self.launch.team_id)
                reason = self.identity_reason(member, owner)
                # 接管可以先更新 team.json，心跳尚未写入；旧成员也必须停止。
                team = self.store.load_team(self.launch.team_id).team
                if team.lead_generation != self.launch.owner_lead_generation:
                    reason = "lead_replaced"
                if reason:
                    self.reason = reason
                    return reason
                current = owner.get("sequence")
                if current is not None:
                    if sequence is not None and current > sequence:
                        advanced = now
                        self.ready.set()
                    sequence = current if sequence is None or current > sequence else sequence
            except (TeamStoreError, OSError):
                pass  # 读取失败不续租，也不立即把暂时故障当作 Lead 死亡。
            if not self.ready.is_set() and now - started >= settings.startup_grace:
                self.reason = "lead_startup_timeout"
                return self.reason
            if self.ready.is_set() and now - advanced >= settings.lease_timeout:
                self.reason = "lead_lease_expired"
                return self.reason
            await asyncio.sleep(settings.poll_interval)
