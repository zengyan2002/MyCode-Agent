"""复制命令需要的普通文件，排除凭证、运行记录和链接目标。"""

from __future__ import annotations

import fnmatch
import os
import shutil
import stat
import threading
from dataclasses import dataclass
from pathlib import Path

from mycode.models.config import CommandSandboxSettings, SecretValue


_EXCLUDED = frozenset({".git", ".mycode", ".env", "config.local.yaml", ".ssh", ".aws",
    ".azure", ".docker", ".kube", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"})
_CHUNK = 65536


class SnapshotError(ValueError):
    """副本尚未准备好，不能执行命令。"""


@dataclass(frozen=True)
class SnapshotInfo:
    root: Path
    file_count: int
    total_bytes: int
    excluded_count: int
    excluded_relative_paths: tuple[str, ...]


def prepare_snapshot(root: Path, target: Path, settings: CommandSandboxSettings,
                     protected_paths: tuple[Path, ...], secrets: tuple[SecretValue, ...],
                     cancelled: threading.Event) -> SnapshotInfo:
    """分块复制当前工作区；调用方取消时停止复制，不留下可执行的半份输入。"""
    root = root.resolve(strict=True)
    protected = {os.path.normcase(str(p.resolve())) for p in protected_paths}
    secret_bytes = tuple(s.reveal().encode("utf-8") for s in secrets if s.reveal())
    overlap = max((len(s) - 1 for s in secret_bytes), default=0)
    limit = settings.snapshot_max_mb * 1024 * 1024
    count = total = excluded = scanned = 0
    names: list[str] = []

    def skip(relative: str) -> None:
        nonlocal excluded
        excluded += 1
        if len(names) < 20:
            names.append(relative)

    def copy_directory(source: Path, dest: Path) -> None:
        nonlocal count, total, scanned
        dest.mkdir(parents=True, exist_ok=True)
        with os.scandir(source) as entries:
            for entry in entries:
                if cancelled.is_set():
                    raise SnapshotError("工作副本准备已取消")
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                name = entry.name.casefold() if os.name == "nt" else entry.name
                match_path = relative.casefold() if os.name == "nt" else relative
                patterns = (p.casefold() if os.name == "nt" else p for p in settings.exclude)
                if (name in _EXCLUDED or name.startswith(".env")
                        or os.path.normcase(str(path.absolute())) in protected
                        or any(fnmatch.fnmatchcase(match_path, p) for p in patterns)):
                    skip(relative)
                    continue
                before = entry.stat(follow_symlinks=False)
                if (stat.S_ISLNK(before.st_mode)
                        or getattr(before, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024)):
                    skip(relative)
                    continue
                if stat.S_ISDIR(before.st_mode):
                    copy_directory(path, dest / entry.name)
                    continue
                if not stat.S_ISREG(before.st_mode):
                    skip(relative)
                    continue
                scanned += 1
                if scanned > settings.snapshot_max_files or before.st_size > limit - total:
                    raise SnapshotError("工作副本超过文件数或字节上限")
                output = dest / entry.name
                tail = b""
                copied = 0
                found_secret = False
                with path.open("rb") as reader, output.open("wb") as writer:
                    while chunk := reader.read(_CHUNK):
                        if cancelled.is_set():
                            raise SnapshotError("工作副本准备已取消")
                        copied += len(chunk)
                        if total + copied > limit:
                            raise SnapshotError("工作副本超过字节上限")
                        data = tail + chunk
                        if any(secret in data for secret in secret_bytes):
                            found_secret = True
                            break
                        tail = data[-overlap:] if overlap else b""
                        writer.write(chunk)
                if found_secret:
                    output.unlink()
                    skip(relative)
                    continue
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise SnapshotError("复制期间文件发生变化，请结束其他编辑后再运行命令")
                # 容器内普通用户需要读取输入，保留脚本可执行位，不复制宿主 ACL。
                output.chmod(0o755 if before.st_mode & 0o111 else 0o644)
                count += 1
                total += copied

    try:
        copy_directory(root, target)
    except (OSError, SnapshotError):
        if target.exists():
            shutil.rmtree(target)
        raise
    return SnapshotInfo(target, count, total, excluded, tuple(names))
