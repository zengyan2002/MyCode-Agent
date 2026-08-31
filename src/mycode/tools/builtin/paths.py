"""检查文件和搜索工具接收的路径；通常只允许访问工作区，read_file 还可以读取用户记忆目录中的单个文件"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

from mycode.models.tools import ToolErrorCode
from mycode.tools.base import ToolContext, ToolFailure

if TYPE_CHECKING:
    from mycode.skills.resources import SkillResourceAccess

#判断当前的路径是否在根目录下
def _inside(root: Path, candidate: Path) -> bool:
    """使用 pathlib ，不依赖字符串前缀判断。"""
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_relative(raw_path: str, *, pattern: bool = False) -> str:
    """
    检查相对路径的写法，并统一路径分隔符

    本函数不访问文件系统，只检查空路径、空字符、绝对路径和父目录跳转。 pattern 为 True 时，还会检查 Glob 字符类方括号是否成对

    Args:
        raw_path: 模型传入的文件路径或 Glob 模式
        pattern: 是否把 raw_path 当作 Glob 模式检查

    Returns:
        将反斜杠替换成正斜杠后的相对路径
    """
    # 这是语法层的第一道防线；后续 WorkspacePaths 仍会解析真实路径。两层
    # 缺一不可：语法检查给出清楚错误，真实路径检查负责阻止符号链接逃逸。
    if not raw_path or "\x00" in raw_path:
        raise ToolFailure(
            ToolErrorCode.INVALID_ARGUMENTS,
            "路径或匹配模式必须是非空字符串",
        )

    # 同时检查 Windows 与 POSIX 语法，确保应用运行在任一平台时，
    # 都能拒绝另一平台格式的绝对路径。
    windows = PureWindowsPath(raw_path)
    posix = PurePosixPath(raw_path.replace("\\", "/"))
    if windows.is_absolute() or windows.drive or posix.is_absolute():
        raise ToolFailure(
            ToolErrorCode.PATH_OUTSIDE_WORKSPACE,
            "不允许使用绝对路径，请改用工作区相对路径",
        )
    if ".." in windows.parts or ".." in posix.parts:
        raise ToolFailure(
            ToolErrorCode.PATH_OUTSIDE_WORKSPACE,
            "不允许通过父目录跳转访问工作区之外",
        )

    normalized = raw_path.replace("\\", "/")
    if pattern:
        # Python 的 fnmatch 会把未闭合字符类当作普通文本；这里主动判错，
        # 给模型一个可操作的重试信号。
        depth = 0
        escaped = False
        for character in normalized:
            if character == "\\" and not escaped:
                escaped = True
                continue
            if not escaped and character == "[":
                depth += 1
            elif not escaped and character == "]":
                depth -= 1
                if depth < 0:
                    break
            escaped = False
        if depth != 0:
            raise ToolFailure(
                ToolErrorCode.INVALID_PATTERN,
                "Glob 模式包含未配对的字符类方括号",
            )
    return normalized


class WorkspacePaths:
    """确认真实目标仍在工作区根目录内之后，才返回解析后的路径。"""
    def __init__(self, root: Path) -> None:
        try:
            #将路径解析为绝对路径并展开符号链接，强制校验路径在文件系统中‌必须实际存在‌
            resolved = root.resolve(strict=True)
        except OSError as exc:
            raise ToolFailure(
                ToolErrorCode.IO_ERROR,
                "工作区根目录无法解析",
            ) from exc
        if not resolved.is_dir():
            raise ToolFailure(
                ToolErrorCode.IO_ERROR,
                "工作区根路径不是目录",
            )
        self.root = resolved

    #把模型传入的工作区相对路径，安全地解析成真实绝对路径；只有目标确实存在、是普通文件，而且解析符号链接后仍位于工作区内，才返回这个路径。
    def existing_file(self, raw_path: str) -> Path:
        # strict=False 允许先解析路径再自行区分“不存在”和“不是普通文件”，
        # 同时仍会展开路径中已经存在的符号链接组件。
        normalized = _validate_relative(raw_path)
        #拼接出候选路径
        candidate = self.root.joinpath(*PurePosixPath(normalized).parts)
        try:
            #resolve(strict=False) 会在不要求目标文件存在的情况下，把候选路径转换成规范化的绝对路径，
            #并尽可能解析其中已有的符号链接，方便后续判断真实目标是否仍在工作区内。
            resolved = candidate.resolve(strict=False)
        except OSError as exc:
            raise ToolFailure(
                ToolErrorCode.IO_ERROR,
                "文件路径无法解析",
            ) from exc
        if not _inside(self.root, resolved):
            raise ToolFailure(
                ToolErrorCode.PATH_OUTSIDE_WORKSPACE,
                "文件真实路径位于工作区之外",
            )
        if not resolved.exists():
            raise ToolFailure(ToolErrorCode.NOT_FOUND, "文件不存在")
        if not resolved.is_file():
            raise ToolFailure(
                ToolErrorCode.NOT_A_FILE,
                "路径指向的不是普通文件",
            )
        return resolved

    def readable_file(
        self,
        raw_path: str,
        user_memory_root: Path | None,
        skill_resources: SkillResourceAccess | None = None,
    ) -> tuple[Path, str]:
        """检查待读取的文件路径，并返回真实路径和显示路径。

        普通路径只能指向工作区内的文件；以 ``~/.mycode/memory/`` 开头的
        路径只能指向用户记忆目录下的单个文件，不允许包含子目录。

        Args:
            raw_path: 模型传入的文件路径。
            user_memory_root: 用户记忆目录的真实路径；为 None 时禁止读取用户记忆。
            skill_resources: 当前会话活动 Skill 的只读资源映射；为 None
                时禁止使用 Skill 虚拟路径。

        Returns:
            文件的真实绝对路径，以及返回给模型显示的路径。
        """

        normalized = raw_path.replace("\\", "/")
        if normalized.startswith("~/.mycode/skills/"):
            if skill_resources is None:
                raise ToolFailure(
                    ToolErrorCode.PATH_OUTSIDE_WORKSPACE,
                    "当前会话没有活动的 Skill 资源读取范围",
                )
            return skill_resources.resolve_readable_file(normalized)
        prefix = "~/.mycode/memory/"
        if not normalized.startswith(prefix):
            # 如果不是用户记忆路径，就把它当作普通工作区文件检查并返回
            path = self.existing_file(raw_path)
            return path, self.relative_path(path)
        # 是用户记忆路径 提取文件名
        filename = normalized[len(prefix) :]
        if (
            user_memory_root is None
            or not filename
            or "/" in filename
            or filename in {".", ".."}
        ):
            raise ToolFailure(
                ToolErrorCode.PATH_OUTSIDE_WORKSPACE,
                "用户记忆路径只能指向 ~/.mycode/memory/ 下的单个文件",
            )
        try:
            root = user_memory_root.resolve(strict=True)
            resolved = (root / filename).resolve(strict=False)
        except FileNotFoundError as exc:
            raise ToolFailure(ToolErrorCode.NOT_FOUND, "用户记忆目录不存在") from exc
        except OSError as exc:
            raise ToolFailure(ToolErrorCode.IO_ERROR, "用户记忆路径无法解析") from exc
        if not root.is_dir() or not _inside(root, resolved):
            raise ToolFailure(
                ToolErrorCode.PATH_OUTSIDE_WORKSPACE,
                "用户记忆文件真实路径位于允许目录之外",
            )
        if not resolved.exists():
            raise ToolFailure(ToolErrorCode.NOT_FOUND, "用户记忆文件不存在")
        if not resolved.is_file():
            raise ToolFailure(ToolErrorCode.NOT_A_FILE, "用户记忆路径不是普通文件")
        return resolved, f"{prefix}{filename}"

    def new_file(self, raw_path: str) -> Path:
        """
        检查新文件路径，并返回工作区内的目标绝对路径
        父目录必须已经存在且位于工作区内，目标文件必须尚不存在

        Args:
            raw_path:模型传入的工作区相对路径

        Returns:
            可以用于创建新文件的绝对路径
        """
        # 初步校验
        normalized = _validate_relative(raw_path)
        # 得到候选路径
        candidate = self.root.joinpath(*PurePosixPath(normalized).parts)
        try:
            parent = candidate.parent.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ToolFailure(
                ToolErrorCode.NOT_FOUND,
                "父目录不存在",
            ) from exc
        except OSError as exc:
            raise ToolFailure(
                ToolErrorCode.IO_ERROR,
                "父目录无法解析",
            ) from exc
        if not _inside(self.root, parent):
            raise ToolFailure(
                ToolErrorCode.PATH_OUTSIDE_WORKSPACE,
                "文件父目录的真实路径位于工作区之外",
            )
        if not parent.is_dir():
            raise ToolFailure(
                ToolErrorCode.NOT_FOUND,
                "父路径不是目录",
            )

        # 得到目标路径
        target = parent / candidate.name
        if target.exists() or target.is_symlink():
            # 目标路径存在且不是符号链接
            try:
                # 转换为绝对路径
                resolved_target = target.resolve(strict=False)
            except OSError as exc:
                raise ToolFailure(
                    ToolErrorCode.IO_ERROR,
                    "目标路径无法解析",
                ) from exc
            if not _inside(self.root, resolved_target):
                raise ToolFailure(
                    ToolErrorCode.PATH_OUTSIDE_WORKSPACE,
                    "目标真实路径位于工作区之外",
                )
            raise ToolFailure(
                ToolErrorCode.ALREADY_EXISTS,
                "目标文件已存在，请使用 edit_file 修改",
            )
        return target


    def matching_files(self, pattern: str) -> tuple[Path, ...]:
        """
        在整个工作区中查找符合 Glob 模式的普通文件，同时阻止通过目录或文件符号链接访问工作区外的内容，最后按稳定顺序返回真实文件路径

        Args:
            pattern: 模型传入的工作区相对 Glob 模式，例如 ``src/**/*.py``

        Returns:
            按工作区相对路径排序的文件绝对路径
        """
        # 不能直接使用 Path.rglob 后相信返回值：目录/文件符号链接都可能把
        # 遍历结果引向工作区外，因此每个候选都要重新 resolve 和检查。
        normalized = _validate_relative(pattern, pattern=True)

        #创建匹配结果列表
        matches: list[tuple[str, Path]] = []

        #递归遍历目录树
        #directory表示当前正在扫描的目录
        #dir_names当前扫描目录下的子目录名称列表
        #file_names当前目录下的文件名称列表
        for directory, dir_names, file_names in os.walk(
            self.root,
            followlinks=False,
        ):
            directory_path = Path(directory)
            #从遍历列表中删除目录符号链接
            #采用dir_names[:]=表示原地修改，要是dir_names=，则os.walk() 内部仍然可能使用原来的列表继续遍历。
            dir_names[:] = [
                name
                for name in dir_names
                if not (directory_path / name).is_symlink()
            ]
            #遍历当前目录中的文件
            for name in file_names:
                raw_candidate = directory_path / name
                try:
                    #将路径转换为规范的据对路径
                    resolved = raw_candidate.resolve(strict=False)
                except OSError:
                    continue
                if not _inside(self.root, resolved) or not resolved.is_file():
                    continue
                #绝对路径先转换为相对路径，转换成 POSIX 风格的字符串（把路径转换成统一使用 / 的字符串）
                #Windows系统上是\，在Mac和linux系统上天然的是/
                relative = raw_candidate.relative_to(self.root).as_posix()
                #创建纯路径匹配对象
                path = PurePosixPath(relative)
                #做匹配
                matched = path.match(normalized)
                if not matched and normalized.startswith("**/"):
                    matched = path.match(normalized[3:])
                if matched:
                    matches.append((relative, resolved))

        #排序，让大模型的输出更加稳定
        matches.sort(key=lambda item: item[0])
        # 大小写折叠键提供跨平台接近一致的顺序，原始路径作为第二关键字
        # 消除仅大小写不同名称的并列，使重复运行结果稳定。
        return tuple(path for _, path in matches)

    def relative_path(self, path: Path) -> str:
        """
        将路径转换为工作区相对路径，并统一使用正斜杠

        转换前会解析路径和已有的符号链接；解析后的路径必须位于工作区内

        Args:
            path: 需要转换的文件或目录路径

        Returns:
            相对于工作区根目录的 POSIX 风格路径字符串
        """
        try:
            #将路径解析成规范化的绝对路径，并展开已经存在的符号链接。
            resolved = path.resolve(strict=False)
        except OSError as exc:
            raise ToolFailure(
                ToolErrorCode.IO_ERROR,
                "路径无法解析",
            ) from exc
        if not _inside(self.root, resolved):
            raise ToolFailure(
                ToolErrorCode.PATH_OUTSIDE_WORKSPACE,
                "路径解析后位于工作区之外",
            )
        return resolved.relative_to(self.root).as_posix()


def preflight_tool_path(
    context: ToolContext,
    tool_name: str,
    raw_path: str,
) -> None:
    """在权限判断前，按照具体工具的规则检查路径。

    本函数只检查路径，不执行工具，也不会修改文件。不同工具会使用与实际
    执行时相同的路径解析方法，从而提前拒绝无效、越界或不符合要求的路径。

        Args:
        context: 当前工具使用的工作区、用户记忆目录和 Skill 资源映射。
        tool_name: 需要检查的工具名称。
        raw_path: 模型传入的原始路径或 Glob 模式。

    Returns:
        None。路径符合对应工具规则时正常返回，违规时抛出 ToolFailure。
    """

    paths = WorkspacePaths(context.workspace_root)
    if tool_name == "read_file":
        paths.readable_file(
            raw_path,
            context.user_memory_root,
            context.skill_resources,
        )
        return
    if tool_name == "edit_file":
        paths.existing_file(raw_path)
        return
    if tool_name == "write_file":
        paths.new_file(raw_path)
        return
    if tool_name in {"find_files", "search_code"}:
        paths.matching_files(raw_path)
        return
    raise ToolFailure(
        ToolErrorCode.INVALID_ARGUMENTS,
        f"工具 {tool_name} 没有可预检的路径操作",
    )
