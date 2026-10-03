"""团队状态和邮箱共用的操作系统级跨进程文件锁。"""

from __future__ import annotations

import json
import os
import random
import secrets
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO


class TeamLockError(RuntimeError):
    """锁争用超时或文件操作失败。"""


@dataclass(frozen=True, slots=True)
class LockOwner:
    """记录持锁进程、调用者、时间和所有权标识，用于诊断。"""

    pid: int
    actor: str
    created_at: datetime
    token: str


class ExclusiveFileLock:
    """Windows 使用 msvcrt 字节锁，POSIX 使用 flock。

    锁文件保持存在，禁止在释放时删除，避免不同进程锁住不同文件实例。
    所有权由打开的文件句柄决定，进程崩溃后由系统释放，不靠 PID 猜测。
    """

    def __init__(
        self, path: Path, actor: str, *, max_attempts: int = 10,
        min_delay_seconds: float = 0.005, max_delay_seconds: float = 0.1,
    ) -> None:
        if not path.is_absolute():
            raise ValueError("团队锁路径必须是绝对路径")
        if max_attempts <= 0:
            raise ValueError("锁重试次数必须为正数")
        if min_delay_seconds < 0 or max_delay_seconds < min_delay_seconds:
            raise ValueError("锁重试等待范围无效")
        self.path = path
        self.actor = actor
        self.max_attempts = max_attempts
        self.min_delay_seconds = min_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self._owner: LockOwner | None = None
        self._handle: BinaryIO | None = None

    def acquire(self) -> LockOwner:
        """有限退避抢锁；仅在取得独占锁后写入诊断信息。"""
        if self._handle is not None:
            raise TeamLockError("同一个锁实例不能重复获取")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Windows 允许锁定 EOF 以外的字节，因此无需在抢锁前初始化文件。
            handle = self.path.open("a+b")
        except OSError as exc:
            raise TeamLockError(f"无法打开锁 {self.path.name}：{exc}") from exc
        try:
            for attempt in range(self.max_attempts):
                try:
                    handle.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if attempt + 1 == self.max_attempts:
                        raise TeamLockError(
                            f"锁 {self.path.name} 在 {self.max_attempts} 次尝试后仍不可用"
                        ) from exc
                    time.sleep(random.uniform(self.min_delay_seconds, self.max_delay_seconds))
            owner = LockOwner(os.getpid(), self.actor, datetime.now().astimezone(), secrets.token_hex(16))
            payload = json.dumps(dict(pid=owner.pid, actor=owner.actor,
                                      created_at=owner.created_at.isoformat(), token=owner.token),
                                 ensure_ascii=False).encode("utf-8")
            handle.seek(0)
            handle.truncate()
            handle.write(payload)
            handle.flush()
            self._handle = handle
            self._owner = owner
            return owner
        except BaseException:
            handle.close()
            raise

    def release(self) -> None:
        """释放系统锁并关闭句柄，保留锁文件供后续进程使用。"""
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            raise TeamLockError(f"无法释放锁 {self.path.name}：{exc}") from exc
        finally:
            handle.close()
            self._handle = None
            self._owner = None

    def __enter__(self) -> LockOwner:
        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()
