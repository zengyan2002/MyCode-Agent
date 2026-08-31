"""管理当前会话里活动目录型 Skill 的额外只读文件范围。"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

from mycode.models.tools import ToolErrorCode
from mycode.models.skills import SkillDefinition
from mycode.tools.base import ToolFailure

_SKILL_RESOURCE_PREFIX = "~/.mycode/skills/"


class SkillResourceAccess:
    """把稳定的 Skill 虚拟路径映射到当前活动目录。

    主会话和每个 fork 各自持有一个实例。SkillRuntime 在激活和停用时
    更新映射，ReadFile 只通过 resolve_readable_file 查询，不会得到写入
    能力。
    """

    def __init__(self) -> None:
        """创建没有任何活动资源根的映射。"""

        # 键是规范化 Skill 名，值是已经 resolve 的真实目录。
        self._roots: dict[str, Path] = {}

    @property
    def active_names(self) -> frozenset[str]:
        """返回当前获得资源读取范围的目录型 Skill 名。

        Returns:
            不会被调用方修改的名字集合。
        """

        return frozenset(self._roots)

    def activate(self, skill: SkillDefinition) -> None:
        """为一个活动目录型 Skill 注册只读根。

        Args:
            skill: Catalog 中当前有效的 SkillDefinition。单文件 Skill 的
                root_path 为 None，因此不会扩大 ReadFile 范围。

        Returns:
            None。目录型 Skill 激活后可使用固定虚拟前缀读取自身文件。

        Raises:
            ToolFailure: 目录型 Skill 的真实根目录不存在或不是目录。
        """

        if skill.root_path is None:
            return
        try:
            root = skill.root_path.resolve(strict=True)
        except OSError as exc:
            raise ToolFailure(
                ToolErrorCode.IO_ERROR,
                f"Skill {skill.name} 的资源目录无法解析",
            ) from exc
        if not root.is_dir():
            raise ToolFailure(
                ToolErrorCode.NOT_FOUND,
                f"Skill {skill.name} 的资源目录不存在",
            )
        self._roots[skill.name.casefold()] = root

    def deactivate(self, name: str) -> bool:
        """撤销一个 Skill 的额外资源读取范围。

        Args:
            name: Catalog 中的 Skill 名，比较时忽略大小写。

        Returns:
            原来存在映射时返回 True；没有活动映射时返回 False。
        """

        return self._roots.pop(name.casefold(), None) is not None

    def clear(self) -> None:
        """撤销当前会话中的全部 Skill 资源读取范围。

        Returns:
            None。
        """

        self._roots.clear()

    def resolve_readable_file(self, raw_path: str) -> tuple[Path, str]:
        """把 Skill 虚拟路径解析成根目录内的真实普通文件。

        Args:
            raw_path: 形如 ~/.mycode/skills/{name}/references/file.md 的
                模型输入路径。

        Returns:
            真实绝对路径和规范化后的虚拟显示路径。

        Raises:
            ToolFailure: 前缀、Skill 名、相对路径、真实边界或文件类型无效。
        """

        normalized = raw_path.replace("\\", "/")
        if not normalized.startswith(_SKILL_RESOURCE_PREFIX):
            raise ToolFailure(
                ToolErrorCode.PATH_OUTSIDE_WORKSPACE,
                "Skill 资源路径必须使用 ~/.mycode/skills/<名称>/ 前缀",
            )
        remainder = normalized[len(_SKILL_RESOURCE_PREFIX) :]
        name, separator, relative = remainder.partition("/")
        if not separator or not name or not relative:
            raise ToolFailure(
                ToolErrorCode.PATH_OUTSIDE_WORKSPACE,
                "Skill 资源路径必须同时包含 Skill 名和文件相对路径",
            )
        root = self._roots.get(name.casefold())
        if root is None:
            raise ToolFailure(
                ToolErrorCode.PATH_OUTSIDE_WORKSPACE,
                f"Skill {name} 当前没有活动的资源读取范围",
            )
        windows = PureWindowsPath(relative)
        posix = PurePosixPath(relative)
        if (
            windows.is_absolute()
            or windows.drive
            or posix.is_absolute()
            or ".." in windows.parts
            or ".." in posix.parts
        ):
            raise ToolFailure(
                ToolErrorCode.PATH_OUTSIDE_WORKSPACE,
                "Skill 资源路径不能使用绝对路径或父目录跳转",
            )
        try:
            candidate = root.joinpath(*posix.parts).resolve(strict=False)
        except OSError as exc:
            raise ToolFailure(
                ToolErrorCode.IO_ERROR,
                "Skill 资源路径无法解析",
            ) from exc
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ToolFailure(
                ToolErrorCode.PATH_OUTSIDE_WORKSPACE,
                "Skill 资源真实路径位于 Skill 目录之外",
            ) from exc
        if not candidate.exists():
            raise ToolFailure(
                ToolErrorCode.NOT_FOUND,
                "Skill 资源文件不存在",
            )
        if not candidate.is_file():
            raise ToolFailure(
                ToolErrorCode.NOT_A_FILE,
                "Skill 资源路径指向的不是普通文件",
            )
        display = (
            f"{_SKILL_RESOURCE_PREFIX}{name.casefold()}/"
            f"{PurePosixPath(relative).as_posix()}"
        )
        return candidate, display
