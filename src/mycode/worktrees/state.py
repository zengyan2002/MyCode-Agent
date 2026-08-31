"""读写 Worktree 状态文件，并进行不调用 Git 的目录快速预检。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from mycode.models.worktrees import (
    WorktreeHeadPrecheck,
    WorktreeKind,
    WorktreeLifecycle,
    WorktreeRecord,
    WorktreeStateLoadResult,
    WorktreeStateSnapshot,
    WorktreeTaskOwner,
    WorktreeTaskState,
)


class WorktreeStateError(RuntimeError):
    """说明状态文件无法读取、编码或原子保存。

    Attributes:
        path: 发生问题的正式状态文件路径。
        temporary_path: 原子保存失败时留下的临时文件；其他错误为 ``None``。
    """

    def __init__(
        self,
        message: str,
        *,
        path: Path,
        temporary_path: Path | None = None,
    ) -> None:
        """保存状态错误和可供人工诊断的文件位置。

        Args:
            message: 不包含状态文件正文的具体错误原因。
            path: 正式状态文件绝对路径。
            temporary_path: 保存失败后可能残留的临时文件绝对路径。

        Returns:
            新的 ``WorktreeStateError`` 异常。
        """

        super().__init__(message)
        self.path = path
        self.temporary_path = temporary_path


class WorktreeStateStore:
    """负责 Worktree JSON 状态的版本校验、原子保存和快速预检。

    Attributes:
        repo_root: 主仓库绝对路径，用于验证所有持久化 Worktree 路径。
        state_path: 固定的 ``.mycode/worktree_state.json`` 绝对路径。

    本类不运行 Git，也不决定是否删除目录。损坏状态会返回“不可信”结果，
    Manager 因而可以继续恢复 JSONL 对话，但必须停用依赖归属记录的破坏性操作。
    """

    def __init__(self, repo_root: Path) -> None:
        """为一个主仓库创建状态存储。

        Args:
            repo_root: 主仓库绝对路径。

        Returns:
            状态路径固定到该仓库下的新存储对象。

        Raises:
            ValueError: ``repo_root`` 不是绝对 ``Path``。
        """

        if not isinstance(repo_root, Path) or not repo_root.is_absolute():
            raise ValueError("WorktreeStateStore.repo_root 必须是绝对 Path")
        self.repo_root = repo_root.resolve()
        self.state_path = self.repo_root / ".mycode" / "worktree_state.json"

    def load(self) -> WorktreeStateLoadResult:
        """读取状态文件，并把缺失与损坏明确区分。

        Returns:
            文件不存在时返回可信的 revision 0 空快照；格式完整时返回解码快照；
            JSON、版本或字段损坏时返回 ``trusted=False`` 和错误原因。
        """

        if not self.state_path.exists():
            return WorktreeStateLoadResult(
                snapshot=WorktreeStateSnapshot(),
                trusted=True,
            )
        try:
            text = self.state_path.read_text(encoding="utf-8")
            raw = json.loads(text)
            snapshot = self._decode_snapshot(raw)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
            return WorktreeStateLoadResult(
                snapshot=None,
                trusted=False,
                error=f"无法读取 {self.state_path}：{exc}",
            )
        return WorktreeStateLoadResult(snapshot=snapshot, trusted=True)

    def save(self, snapshot: WorktreeStateSnapshot) -> None:
        """在同目录写完、同步并原子替换 Worktree 状态文件。

        Args:
            snapshot: Manager 已经递增 revision 的完整可信快照。

        Returns:
            临时文件成功替换正式文件时不返回数据。

        Raises:
            ValueError: ``snapshot`` 类型错误或 revision 比磁盘可信快照旧。
            WorktreeStateError: 创建目录、编码、写入、同步或替换失败。若临时
                文件已经创建，异常的 ``temporary_path`` 会指出其位置。
        """

        if not isinstance(snapshot, WorktreeStateSnapshot):
            raise ValueError("save snapshot 类型无效")
        loaded = self.load()
        if (
            loaded.trusted
            and loaded.snapshot is not None
            and snapshot.revision < loaded.snapshot.revision
        ):
            raise ValueError("Worktree 状态 revision 不能倒退")
        if not loaded.trusted:
            raise WorktreeStateError(
                "现有 Worktree 状态不可信，拒绝覆盖",
                path=self.state_path,
            )

        temporary_path = self.state_path.with_name(
            f"{self.state_path.name}.{uuid4().hex}.tmp"
        )
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                self._encode_snapshot(snapshot),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.state_path)
        except (OSError, TypeError, ValueError) as exc:
            raise WorktreeStateError(
                f"无法原子保存 Worktree 状态：{exc}",
                path=self.state_path,
                temporary_path=(temporary_path if temporary_path.exists() else None),
            ) from exc

    def precheck(self, record: WorktreeRecord) -> WorktreeHeadPrecheck:
        """纯文件读取 Worktree ``.git`` 指针和 admin ``HEAD``。

        Args:
            record: 等待快速恢复的受管 Worktree 记录。

        Returns:
            明显匹配、明显冲突或无法判断的 ``WorktreeHeadPrecheck``。返回匹配
            也不代表最终可信，调用方必须继续调用 Git 后端验证登记和 commit。
        """

        dot_git = record.path / ".git"
        try:
            pointer = dot_git.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            return WorktreeHeadPrecheck(
                matched=None,
                admin_dir=None,
                head_ref=None,
                reason=f"无法读取 .git 指针：{exc}",
            )
        prefix = "gitdir:"
        if not pointer.lower().startswith(prefix):
            return WorktreeHeadPrecheck(
                matched=False,
                admin_dir=None,
                head_ref=None,
                reason=".git 文件不是 gitdir 指针格式",
            )
        raw_path = pointer[len(prefix) :].strip()
        if not raw_path:
            return WorktreeHeadPrecheck(
                matched=False,
                admin_dir=None,
                head_ref=None,
                reason=".git 指针缺少 admin 目录",
            )
        candidate = Path(raw_path)
        admin_dir = (
            candidate.resolve()
            if candidate.is_absolute()
            else (record.path / candidate).resolve()
        )
        common_dir = self._common_git_dir()
        if common_dir is None:
            return WorktreeHeadPrecheck(
                matched=None,
                admin_dir=admin_dir,
                head_ref=None,
                reason="无法确定主仓库 Git admin 目录",
            )
        managed_admin_root = (common_dir / "worktrees").resolve()
        try:
            admin_dir.relative_to(managed_admin_root)
        except ValueError:
            return WorktreeHeadPrecheck(
                matched=False,
                admin_dir=admin_dir,
                head_ref=None,
                reason=".git 指针越过当前仓库的 Worktree admin 目录",
            )
        try:
            head_text = (admin_dir / "HEAD").read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            return WorktreeHeadPrecheck(
                matched=None,
                admin_dir=admin_dir,
                head_ref=None,
                reason=f"无法读取 Worktree admin HEAD：{exc}",
            )
        ref_prefix = "ref: "
        if not head_text.startswith(ref_prefix):
            return WorktreeHeadPrecheck(
                matched=False,
                admin_dir=admin_dir,
                head_ref=None,
                reason="Worktree admin HEAD 处于 detached 状态",
            )
        head_ref = head_text[len(ref_prefix) :].strip()
        expected_ref = f"refs/heads/{record.branch}"
        matched = head_ref == expected_ref
        return WorktreeHeadPrecheck(
            matched=matched,
            admin_dir=admin_dir,
            head_ref=head_ref,
            reason=("文件系统预检匹配" if matched else "admin HEAD 分支与状态记录不一致"),
        )

    def _common_git_dir(self) -> Path | None:
        """从主仓库 ``.git`` 目录或指针推导公共 Git admin 目录。

        Returns:
            普通主仓库返回 ``repo/.git``；主目录本身是 linked worktree 时返回
            admin 目录的共同父目录；格式无法识别时返回 ``None``。
        """

        dot_git = self.repo_root / ".git"
        if dot_git.is_dir():
            return dot_git.resolve()
        try:
            pointer = dot_git.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return None
        if not pointer.lower().startswith("gitdir:"):
            return None
        raw_path = pointer[len("gitdir:") :].strip()
        candidate = Path(raw_path)
        admin_dir = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.repo_root / candidate).resolve()
        )
        if admin_dir.parent.name != "worktrees":
            return None
        return admin_dir.parent.parent

    @staticmethod
    def _encode_snapshot(snapshot: WorktreeStateSnapshot) -> dict[str, object]:
        """把不可变状态模型转换成 JSON 可编码对象。

        Args:
            snapshot: 要写入磁盘的完整快照。

        Returns:
            只包含字典、列表、字符串、整数、布尔值和 ``None`` 的对象。
        """

        records: list[dict[str, object]] = []
        for record in snapshot.records:
            owner = None
            if record.owner is not None:
                owner = {
                    "session_id": record.owner.session_id,
                    "task_id": record.owner.task_id,
                    "origin": record.owner.origin,
                    "team_id": record.owner.team_id,
                    "agent_id": record.owner.agent_id,
                }
            records.append(
                {
                    "name": record.name,
                    "path": os.fspath(record.path),
                    "branch": record.branch,
                    "base_ref": record.base_ref,
                    "base_commit": record.base_commit,
                    "kind": record.kind.value,
                    "lifecycle": record.lifecycle.value,
                    "owner": owner,
                    "owner_pid": record.owner_pid,
                    "task_state": (
                        record.task_state.value if record.task_state is not None else None
                    ),
                    "created_at": record.created_at.isoformat(),
                    "last_used_at": record.last_used_at.isoformat(),
                    "initialization_complete": record.initialization_complete,
                    "warnings": list(record.warnings),
                }
            )
        return {
            "version": snapshot.version,
            "revision": snapshot.revision,
            "records": records,
            "session_bindings": dict(snapshot.session_bindings),
        }

    def _decode_snapshot(self, raw: object) -> WorktreeStateSnapshot:
        """把 JSON 对象严格转换成 Worktree 状态模型。

        Args:
            raw: ``json.loads`` 返回的根对象。

        Returns:
            字段完整、版本受支持且路径仍在受管目录中的状态快照。

        Raises:
            ValueError: 根结构、版本、记录、枚举、时间、路径或会话绑定无效。
        """

        if not isinstance(raw, Mapping):
            raise ValueError("状态根节点必须是对象")
        expected = {"version", "revision", "records", "session_bindings"}
        if set(raw) != expected:
            raise ValueError("状态根字段不完整或包含未知字段")
        if raw["version"] != 1:
            raise ValueError(f"不支持的 Worktree 状态版本：{raw['version']}")
        records_raw = raw["records"]
        if not isinstance(records_raw, list):
            raise ValueError("状态 records 必须是列表")
        records = tuple(self._decode_record(item) for item in records_raw)
        bindings_raw = raw["session_bindings"]
        if not isinstance(bindings_raw, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in bindings_raw.items()
        ):
            raise ValueError("状态 session_bindings 必须是字符串映射")
        revision = raw["revision"]
        return WorktreeStateSnapshot(
            version=1,
            revision=revision,  # type: ignore[arg-type]
            records=records,
            session_bindings=tuple(bindings_raw.items()),
        )

    def _decode_record(self, raw: object) -> WorktreeRecord:
        """解码并校验状态文件中的一条 Worktree 记录。

        Args:
            raw: ``records`` 列表中的一个 JSON 值。

        Returns:
            路径位于 ``.mycode/worktrees`` 内的 ``WorktreeRecord``。

        Raises:
            ValueError: 字段集合、归属、路径、枚举、时间或其他值无效。
        """

        if not isinstance(raw, Mapping):
            raise ValueError("Worktree 记录必须是对象")
        expected = {
            "name",
            "path",
            "branch",
            "base_ref",
            "base_commit",
            "kind",
            "lifecycle",
            "owner",
            "owner_pid",
            "task_state",
            "created_at",
            "last_used_at",
            "initialization_complete",
            "warnings",
        }
        if set(raw) != expected:
            raise ValueError("Worktree 记录字段不完整或包含未知字段")
        path_value = raw["path"]
        if not isinstance(path_value, str):
            raise ValueError("Worktree 记录 path 必须是字符串")
        path = Path(path_value)
        if not path.is_absolute():
            raise ValueError("Worktree 记录 path 必须是绝对路径")
        managed_root = (self.repo_root / ".mycode" / "worktrees").resolve()
        path = path.resolve()
        try:
            path.relative_to(managed_root)
        except ValueError as exc:
            raise ValueError("Worktree 记录 path 越过受管目录") from exc
        owner_raw = raw["owner"]
        owner = None
        if owner_raw is not None:
            if not isinstance(owner_raw, Mapping) or not set(owner_raw).issubset({
                "session_id", "task_id", "origin", "team_id", "agent_id"
            }) or not {"session_id", "task_id", "origin"}.issubset(owner_raw):
                raise ValueError("Worktree 记录 owner 无效")
            owner = WorktreeTaskOwner(
                session_id=owner_raw["session_id"],  # type: ignore[arg-type]
                task_id=owner_raw["task_id"],  # type: ignore[arg-type]
                origin=owner_raw["origin"],  # type: ignore[arg-type]
                team_id=owner_raw.get("team_id"),  # type: ignore[arg-type]
                agent_id=owner_raw.get("agent_id"),  # type: ignore[arg-type]
            )
        task_state_raw = raw["task_state"]
        task_state = (
            None
            if task_state_raw is None
            else WorktreeTaskState(task_state_raw)  # type: ignore[arg-type]
        )
        warnings = raw["warnings"]
        if not isinstance(warnings, list):
            raise ValueError("Worktree 记录 warnings 必须是列表")
        return WorktreeRecord(
            name=raw["name"],  # type: ignore[arg-type]
            path=path,
            branch=raw["branch"],  # type: ignore[arg-type]
            base_ref=raw["base_ref"],  # type: ignore[arg-type]
            base_commit=raw["base_commit"],  # type: ignore[arg-type]
            kind=WorktreeKind(raw["kind"]),  # type: ignore[arg-type]
            lifecycle=WorktreeLifecycle(raw["lifecycle"]),  # type: ignore[arg-type]
            owner=owner,
            owner_pid=raw["owner_pid"],  # type: ignore[arg-type]
            task_state=task_state,
            created_at=datetime.fromisoformat(raw["created_at"]),  # type: ignore[arg-type]
            last_used_at=datetime.fromisoformat(raw["last_used_at"]),  # type: ignore[arg-type]
            initialization_complete=raw["initialization_complete"],  # type: ignore[arg-type]
            warnings=tuple(warnings),
        )
