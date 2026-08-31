"""团队 JSON 和 JSONL 文件共用的跨进程独占锁。"""

from __future__ import annotations

import json
import os
import random
import secrets
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


class TeamLockError(RuntimeError):
    """表示锁被活进程占用、记录损坏或有限重试已经耗尽。"""


@dataclass(frozen=True, slots=True)
class LockOwner:
    """保存锁文件中的进程、调用者和随机所有权标识。

    Attributes:
        pid: 创建锁文件的进程 ID。
        actor: 便于错误信息定位的团队调用者。
        created_at: 锁文件成功创建的带时区时间。
        token: 释放锁时必须再次匹配的随机值。
    """

    pid: int
    actor: str
    created_at: datetime
    token: str


def _process_alive(pid: int) -> bool | None:
    """判断锁记录中的本机进程是否仍存在。

    Args:
        pid: 锁文件记录的本机进程 ID。

    Returns:
        进程存在时返回 True；系统明确报告不存在时返回 False；权限或平台
        原因导致无法确认时返回 None。
    """

    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    except OSError:
        return None
    return True


class ExclusiveFileLock:
    """通过独占创建文件保护一个跨进程更新区段。

    Attributes:
        path: 实际创建和删除的锁文件绝对路径。
        actor: 锁争用错误中展示的调用者名称。
        max_attempts: 抢锁最多尝试次数；耗尽后直接失败。
        min_delay_seconds: 相邻重试之间最短随机等待。
        max_delay_seconds: 相邻重试之间最长随机等待。
    """

    def __init__(
        self,
        path: Path,
        actor: str,
        *,
        max_attempts: int = 10,
        min_delay_seconds: float = 0.005,
        max_delay_seconds: float = 0.1,
    ) -> None:
        """保存锁路径和有限退避参数。

        Args:
            path: 要独占创建的锁文件绝对路径。
            actor: 便于诊断的调用者名称。
            max_attempts: 包含首次尝试在内的最大抢锁次数。
            min_delay_seconds: 重试随机等待的下界。
            max_delay_seconds: 重试随机等待的上界。

        Returns:
            不返回数据；实际抢锁发生在 ``acquire`` 或进入 with 时。
        """

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

    def acquire(self) -> LockOwner:
        """在有限次数内取得锁，并清理能确认已退出进程留下的旧锁。

        Returns:
            本次成功写入锁文件的所有权记录。

        Raises:
            TeamLockError: 锁一直由活进程持有、持有者无法确认或锁文件损坏。
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)
        last_owner: LockOwner | None = None
        for attempt in range(1, self.max_attempts + 1):
            owner = LockOwner(
                pid=os.getpid(),
                actor=self.actor,
                created_at=datetime.now().astimezone(),
                token=secrets.token_hex(16),
            )
            payload = json.dumps(
                {
                    "pid": owner.pid,
                    "actor": owner.actor,
                    "created_at": owner.created_at.isoformat(),
                    "token": owner.token,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            try:
                with self.path.open("x", encoding="utf-8", newline="") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError:
                last_owner = self._read_owner()
                alive = _process_alive(last_owner.pid)
                if alive is False:
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        raise TeamLockError(
                            f"无法清理已退出进程留下的锁 {self.path.name}：{exc}"
                        ) from exc
                    continue
                if alive is None:
                    raise TeamLockError(
                        f"无法确认锁 {self.path.name} 的进程 {last_owner.pid} 是否仍在运行"
                    )
                if attempt < self.max_attempts:
                    time.sleep(random.uniform(self.min_delay_seconds, self.max_delay_seconds))
                    continue
                break
            except OSError as exc:
                raise TeamLockError(f"无法创建锁 {self.path.name}：{exc}") from exc
            self._owner = owner
            return owner
        detail = (
            f"，当前由 {last_owner.actor}（PID {last_owner.pid}）持有"
            if last_owner is not None
            else ""
        )
        raise TeamLockError(
            f"锁 {self.path.name} 在 {self.max_attempts} 次尝试后仍不可用{detail}"
        )

    def release(self) -> None:
        """只在磁盘 token 仍属于本实例时删除锁。

        Returns:
            成功删除锁或当前实例尚未持锁时不返回数据。

        Raises:
            TeamLockError: 锁已经被替换，或读取、删除失败。
        """

        owner = self._owner
        if owner is None:
            return
        current = self._read_owner()
        if current.token != owner.token:
            raise TeamLockError(f"锁 {self.path.name} 已由其他持有者替换")
        try:
            self.path.unlink()
        except OSError as exc:
            raise TeamLockError(f"无法释放锁 {self.path.name}：{exc}") from exc
        finally:
            self._owner = None

    def _read_owner(self) -> LockOwner:
        """读取并校验当前锁文件。

        Returns:
            锁文件记录的所有者。

        Raises:
            TeamLockError: 文件无法读取或字段格式无效。
        """

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return LockOwner(
                pid=int(raw["pid"]),
                actor=str(raw["actor"]),
                created_at=datetime.fromisoformat(str(raw["created_at"])),
                token=str(raw["token"]),
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise TeamLockError(f"锁文件 {self.path.name} 内容无效：{exc}") from exc

    def __enter__(self) -> LockOwner:
        """取得锁并把所有权记录交给 with 代码块。

        Returns:
            本次成功写入锁文件的进程、线程、nonce 和创建时间。

        Raises:
            TeamLockError: 门限内无法取得锁或锁文件损坏。
        """

        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """离开 with 代码块时释放当前实例取得的锁。

        Args:
            exc_type: with 代码块抛出的异常类型；正常退出时为 ``None``。
            exc: with 代码块抛出的异常对象；正常退出时为 ``None``。
            traceback: with 代码块异常的回溯；正常退出时为 ``None``。

        Returns:
            锁释放后不返回数据，也不吞掉代码块中的异常。
        """

        self.release()
