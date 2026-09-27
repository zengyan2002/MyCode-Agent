"""在创建容器之前记录名称，供取消与宿主重启后清理。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from mycode.models.operations import OperationRecord


@dataclass
class SandboxRunRecord:
    version: int
    sandbox_id: str
    operation_id: str
    attempt: int
    owner_token_hash: str
    owner_pid: int
    control_root: str
    workspace_root: str
    container_name: str
    container_id: str
    image_id: str
    endpoint: str
    stage: str
    created_at: float
    cleanup_error: str | None = None

    @classmethod
    def create(cls, root: Path, workspace: Path, operation: OperationRecord,
               image_id: str, endpoint: str) -> "SandboxRunRecord":
        identity = uuid.uuid4().hex
        return cls(1, identity, operation.operation_id, operation.attempt,
                   hashlib.sha256((operation.owner_token or "").encode()).hexdigest(),
                   os.getpid(), str(root.resolve()), str(workspace),
                   f"mycode-sbx-{identity}", "", image_id, endpoint, "preparing", time.time())

    @property
    def labels(self) -> dict[str, str]:
        return {
            "mycode.sandbox": "v1",
            "mycode.root": hashlib.sha256(os.path.normcase(self.control_root).encode()).hexdigest(),
            "mycode.sandbox-id": self.sandbox_id,
            "mycode.operation-id": self.operation_id,
        }


def record_directory(root: Path, record: SandboxRunRecord) -> Path:
    """在删除或写入前检查记录确实属于指定控制目录。"""
    if (record.version != 1 or not re.fullmatch(r"[0-9a-f]{32}", record.sandbox_id)
            or Path(record.control_root).resolve() != root.resolve()
            or record.container_name != f"mycode-sbx-{record.sandbox_id}"):
        raise ValueError("沙箱记录的归属或名称无效")
    base = root.resolve() / ".mycode" / "sandboxes"
    if not base.resolve().is_relative_to(root.resolve()):
        raise ValueError("沙箱控制目录越出项目")
    directory = base / record.sandbox_id
    if directory.resolve().parent != base.resolve():
        raise ValueError("沙箱记录目录越界")
    return directory


def save_record(root: Path, record: SandboxRunRecord) -> None:
    directory = record_directory(root, record)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "record.tmp"
    with temporary.open("w", encoding="utf-8") as writer:
        json.dump(asdict(record), writer, ensure_ascii=False)
        writer.flush()
        os.fsync(writer.fileno())
    temporary.replace(directory / "record.json")


def load_records(root: Path) -> tuple[SandboxRunRecord, ...]:
    base = root / ".mycode" / "sandboxes"
    if not base.exists():
        return ()
    result = []
    for path in sorted(base.glob("*/record.json")):
        record = SandboxRunRecord(**json.loads(path.read_text(encoding="utf-8")))
        if record_directory(root, record).resolve() != path.parent.resolve():
            raise ValueError("沙箱记录文件位置不匹配")
        result.append(record)
    return tuple(result)


def remove_record(root: Path, record: SandboxRunRecord) -> None:
    """仅在调用方确认容器不存在后删除该次输入副本和记录。"""
    directory = record_directory(root, record)
    if directory.exists():
        shutil.rmtree(directory)
