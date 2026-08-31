"""把过长的工具结果和上下文摘要前的用户原话保存到工作区文件中"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from mycode.errors import ArtifactError, redact_secrets
from mycode.models.config import SecretValue
from mycode.models.messages import UserMessage

# 会话 ID 只能包含字母、数字、下划线和连字符，长度为 1～128 个字符
_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


@dataclass(frozen=True)
class ArtifactRecord:
    """记录一份 artifact 文件的路径和原始正文大小
    """

    # artifact 文件相对于工作区根目录的路径；模型可用该路径调用 read_file
    relative_path: str
    # 敏感信息被替换前，待保存正文的字符数
    original_chars: int
    # 敏感信息被替换前，待保存正文编码成 UTF-8 后的字节数
    original_bytes: int



@dataclass(frozen=True)
class StagedArtifact:
    """记录一份等待摘要响应通过检查的用户原话临时文件

    创建该对象时，全部用户原话已经写入 temporary_path。摘要响应通过
    格式检查后，程序把它改名为 final_path；请求失败或取消时将其删除。
    """

    # 当前实际存在的临时文件
    temporary_path: Path
    # 摘要检查通过后，临时文件要改成的正式文件名
    final_path: Path
    # 正式文件相对于工作区的路径，以及原始正文大小
    record: ArtifactRecord


class ArtifactStore:
    """负责保存和清理当前会话产生的 artifact 文件

    CLI 启动会话时会创建一个 ArtifactStore。工具结果过长时，用它保存完整正文；
    生成上下文摘要时，用它临时保存用户原话。执行 /clear 或正常退出时，它会
    删除当前会话的 artifact 目录，不影响其他会话的文件
    """

    def __init__(
        self,
        workspace_root: Path,
        run_id: str,
        *,
        secrets: Iterable[SecretValue] = (),
    ) -> None:
        if _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("本次程序运行的 artifact 目录 ID 格式无效")
        try:
            # 将工作区目录解析到绝对路径
            root = workspace_root.resolve(strict=True)
        except OSError as exc:
            raise ArtifactError("无法解析 artifact 工作区") from exc
        if not root.is_dir():
            raise ArtifactError("artifact 工作区不是目录")

        # 这些路径只由工作区和已校验的本地会话 ID 构造，不接收模型路径。
        # 工作区路径
        self._workspace_root = root
        # 当前工作区内所有 artifact 的总目录
        self._artifacts_root = root / ".mycode" / "artifacts"
        # 本次程序运行（会话）使用的 artifact 目录
        self._run_root = self._artifacts_root / run_id
        # 保存配置中的 API Key 等敏感值；写入 artifact 前，将正文中出现的这些值替换为 ***
        self._secrets = tuple(secrets)
        # 序号让同一会话中的文件名可读且不依赖工具完成的并发顺序之外状态。
        self._sequence = 0

    @property
    def run_root(self) -> Path:
        """返回当前会话固定目录；目录可能尚未实际创建"""
        return self._run_root

    def safe_text(self, content: str) -> str:
        """替换应用已加载的敏感值，返回允许写入 artifact 的文本"""
        return redact_secrets(content, self._secrets)

    def _ensure_run(self) -> None:
        """按需创建应用目录和当前会话目录。"""
        try:
            self._run_root.mkdir(parents=True, exist_ok=True)
            resolved = self._run_root.resolve(strict=True)
        except OSError as exc:
            raise ArtifactError("无法创建当前会话 artifact 目录") from exc
        try:
            # 借用relative_to检查resolved是否在工作区内
            resolved.relative_to(self._workspace_root)
        except ValueError as exc:
            raise ArtifactError("artifact 目录解析后超出工作区") from exc

    def _relative(self, path: Path) -> str:
        """把已构造的 artifact 路径转换成工作区相对 POSIX 路径。"""
        try:
            return path.resolve(strict=False).relative_to(
                self._workspace_root
            ).as_posix()
        except (OSError, ValueError) as exc:
            raise ArtifactError("artifact 文件路径超出工作区") from exc

    def _next_name(self, prefix: str, identity: str) -> str:
        """生成不包含外部原文的当前会话唯一文件名。

        Args:
              prefix: 写在文件名开头的文件类型，例如 ``tool`` 或 ``user-transcript``
              identity: 用于计算文件名短哈希的原始标识。工具结果使用工具名和调用 ID 的组合；用户原话文件使用 ``user-transcript`

        Returns:
              由文件类型、四位递增序号和20位短哈希组成的文本文件名
        """
        self._sequence += 1
        # 把工具调用身份转换成20位短哈希，避免在文件名中直接使用外部提供的字符串
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        return f"{prefix}-{self._sequence:04d}-{digest}.txt"

    def save_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        content: str,
    ) -> ArtifactRecord:
        """把一份工具结果保存到当前会话的 artifact 目录

        写入文件前会替换正文中的 API Key 等敏感值。文件已存在或写入失败时抛出 ArtifactError，不会覆盖已有文件。

        Args:
            tool_call_id: Provider 为本次工具调用分配的 ID，用于生成文件名
            tool_name: 本次实际执行的工具名，用于生成文件名
            content: 工具执行后返回的完整文本

        Returns:
            文件相对于工作区的路径，以及脱敏前正文的字符数和 UTF-8 字节数
        """
        self._ensure_run()
        # 统计工具执行完后的文本编译成utf-8后的字节数
        original_bytes = len(content.encode("utf-8"))
        # 拿到工具结果要保存的文件名
        name = self._next_name("tool", f"{tool_name}\0{tool_call_id}")
        # 正式文件只有在临时文件完整写入后才会出现。
        path = self._run_root / name
        relative_path = self._relative(path)
        temporary_path = self._run_root / f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary_path.open(
                "x",
                encoding="utf-8",
                newline="",
            ) as handle:
                # 将脱敏后的字符串写入缓冲区
                handle.write(self.safe_text(content))
                # 把 Python 文件对象缓冲区中的数据交给操作系统
                handle.flush()
                # 把还在操作系统缓存中的文件内容立即写入磁盘
                os.fsync(handle.fileno())
            if path.exists():
                temporary_path.unlink(missing_ok=True)
                raise ArtifactError("工具结果 artifact 已存在，拒绝覆盖")
            temporary_path.replace(path)
        except OSError as exc:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                raise ArtifactError(
                    "无法清理写入失败的工具结果临时文件"
                ) from cleanup_exc
            raise ArtifactError("无法写入工具结果 artifact") from exc
        return ArtifactRecord(
            relative_path=relative_path,
            original_chars=len(content),
            original_bytes=original_bytes,
        )

    def write_user_transcript_temp_file(
        self,
        messages: tuple[UserMessage, ...],
    ) -> StagedArtifact:
        """把用户原话写入临时文件，等待摘要验证成功后提交"""
        if not messages:
            raise ValueError("用户原文记录至少需要一条消息")
        # 创建当前会话目录
        self._ensure_run()
        # 获得文件名
        name = self._next_name("user-transcript", "user-transcript")
        # 摘要检查通过后，用户原话文件使用的正式路径；此时文件还不存在
        final_path = self._run_root / name
        # 摘要检查通过前，用户原话实际写入的临时文件路径
        temporary_path = self._run_root / f".{name}.{uuid.uuid4().hex}.tmp"
        # 存储用户原话的文本信息
        sections = []
        for index, message in enumerate(messages, start=1):
            sections.append(
                f"--- 用户消息 {index}（{len(message.content)} 字符）---\n"
                f"{message.content}\n"
            )
        content = "\n".join(sections)
        try:
            with temporary_path.open(
                "x",
                encoding="utf-8",
                newline="",
            ) as handle:
                handle.write(self.safe_text(content))
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise ArtifactError("无法暂存用户原文记录") from exc
        return StagedArtifact(
            temporary_path=temporary_path,
            final_path=final_path,
            record=ArtifactRecord(
                relative_path=self._relative(final_path),
                original_chars=len(content),
                original_bytes=len(content.encode("utf-8"))
            ),
        )

    def commit_staged(self, staged: StagedArtifact) -> ArtifactRecord:
        """把用户原话的临时文件改成正式文件，并返回正式文件的路径和大小，ContextManager 只在摘要响应通过格式检查后调用这个方法"""
        try:
            staged.temporary_path.resolve(strict=False).relative_to(
                self._run_root.resolve(strict=False)
            )
            staged.final_path.resolve(strict=False).relative_to(
                self._run_root.resolve(strict=False)
            )
            if staged.final_path.exists():
                raise ArtifactError("用户原文 artifact 已存在，拒绝覆盖")
            staged.temporary_path.replace(staged.final_path)
        except ArtifactError:
            raise
        except (OSError, ValueError) as exc:
            raise ArtifactError("无法提交用户原文 artifact") from exc
        return staged.record

    def discard_staged(self, staged: StagedArtifact) -> None:
        """摘要失败或取消时，删除之前保存用户原话的临时文件"""
        try:
            # 检查要删除的临时文件是否确实位于当前会话目录中
            staged.temporary_path.resolve(strict=False).relative_to(
                self._run_root.resolve(strict=False)
            )
            # 删除临时文件
            staged.temporary_path.unlink(missing_ok=True)
        except (OSError, ValueError) as exc:
            raise ArtifactError("无法清理用户原文临时文件") from exc

    def cleanup(self) -> None:
        """删除本次程序运行保存的所有 artifact 文件和会话目录"""
        if not self._run_root.exists():
            return
        try:
            # 取得当前会话目录真实存在的绝对路径
            resolved = self._run_root.resolve(strict=True)
            # 判断resolved是否是 artifact 总目录的直接子目录
            if resolved.parent != self._artifacts_root.resolve(strict=True):
                raise ArtifactError("当前 artifact 清理目录无效")
            # 是子目录的话，直接删除当前会话目录
            shutil.rmtree(resolved)
        except ArtifactError:
            raise
        except OSError as exc:
            raise ArtifactError("无法清理当前会话 artifact") from exc
