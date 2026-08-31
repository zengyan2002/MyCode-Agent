"""校验 Worktree slug，并构造受管目录和本地分支名。"""

from __future__ import annotations

import re
from pathlib import Path

from mycode.models.worktrees import WorktreeName


_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_MAX_SLUG_LENGTH = 64
_BRANCH_PREFIX = "mycode-worktree-"


def validate_worktree_slug(name: str, repo_root: Path) -> WorktreeName:
    """校验用户或 LLM 提供的 slug，并生成受管路径和分支名。

    名称可以包含斜杠来表达逻辑层级，例如 ``team-refactor/alice``。
    实际目录和分支会把斜杠换成加号，得到
    ``.mycode/worktrees/team-refactor+alice`` 和
    ``mycode-worktree-team-refactor+alice``。

    Args:
        name: 原始 slug，长度为 1 到 64，只允许 ASCII 字母、数字、点、
            下划线、连字符和正斜杠。
        repo_root: 主仓库绝对路径。受管目录固定放在它的
            ``.mycode/worktrees`` 子目录中。

    Returns:
        包含原名、平铺名、经过边界检查的绝对路径和候选分支名的
        ``WorktreeName``。最终分支仍应交给 Git 的 ``check-ref-format`` 验证。

    Raises:
        ValueError: 名称为空、过长、包含空段或非法字符，包含 ``.``/``..``
            路径段，主仓库不是绝对路径，或构造出的目录越过受管目录边界。
    """

    if not isinstance(name, str) or not name:
        raise ValueError("Worktree 名称不能为空")
    if len(name) > _MAX_SLUG_LENGTH:
        raise ValueError("Worktree 名称不能超过 64 个字符")
    if not isinstance(repo_root, Path) or not repo_root.is_absolute():
        raise ValueError("repo_root 必须是绝对 Path")

    segments = name.split("/")
    for segment in segments:
        if not segment:
            raise ValueError("Worktree 名称不能包含空路径段")
        if segment in {".", ".."}:
            raise ValueError("Worktree 名称不能包含 . 或 .. 路径段")
        if _SEGMENT_PATTERN.fullmatch(segment) is None:
            raise ValueError(
                "Worktree 名称只允许 ASCII 字母、数字、点、下划线、连字符和正斜杠"
            )

    flat = "+".join(segments)
    managed_root = (repo_root / ".mycode" / "worktrees").resolve()
    path = (managed_root / flat).resolve()
    try:
        path.relative_to(managed_root)
    except ValueError as exc:
        raise ValueError("Worktree 目录越过了 .mycode/worktrees 边界") from exc

    return WorktreeName(
        original=name,
        flat=flat,
        path=path,
        branch=f"{_BRANCH_PREFIX}{flat}",
    )
