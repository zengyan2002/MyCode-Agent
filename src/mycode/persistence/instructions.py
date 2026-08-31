"""读取三层 MYCODE.md，并在各自允许的目录内展开 @include。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from mycode.constants import INSTRUCTION_INCLUDE_MAX_DEPTH

# 用于识别 MYCODE.md 中单独占一行的 @include 指令，并提取后面的文件路径
# 捕获组group(1)用于捕获@include后面的路径
_INCLUDE_LINE = re.compile(r"^\s*@include\s+(.+?)\s*$")


@dataclass(frozen=True)
class InstructionWarning:
    """记录一条未能加载的@include指令引用及其具体原因。"""

    path: Path
    reason: str


@dataclass(frozen=True)
class LoadedInstructions:
    """保存已按优先级拼好的项目指令，以及启动时要显示的警告。"""

    content: str
    warnings: tuple[InstructionWarning, ...]


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _include_target(raw: str) -> str:
    """清理 @include 指令中捕获的路径文本。

    去除路径首尾的空白字符。如果路径外层使用成对的单引号、
    双引号或尖括号，则同时去除这些外层符号。本函数不检查路径
    是否存在，也不判断路径是否位于允许目录。

    Args:
        raw: 正则捕获组返回的原始路径文本，可能包含首尾空白或外层符号。

    Returns:
        清理首尾空白和成对外层符号后的路径文本。
    """
    target = raw.strip()
    if len(target) >= 2:
        if target[0] == target[-1] and target[0] in {"'", '"'}:
            target = target[1:-1].strip()
        elif target[0] == "<" and target[-1] == ">":
            target = target[1:-1].strip()
    return target


class ProjectInstructionLoader:
    """加载当前项目和当前用户为 MyCode 编写的手写指令。"""

    def __init__(self, workspace_root: Path) -> None:
        try:
            root = workspace_root.resolve(strict=True)
        except OSError as exc:
            raise ValueError("项目根目录无法解析") from exc
        if not root.is_dir():
            raise ValueError("项目根路径不是目录")
        self._workspace_root = root

    def _expand(
        self,
        path: Path,
        *,
        trust_root: Path,
        depth: int,
        visited: set[Path],
        warnings: list[InstructionWarning],
    ) -> str:
        """读取一个指令文件，并递归展开文件中的 @include 指令形成文本信息

        函数先确认目标文件位于允许目录内、尚未加载且可以作为 UTF-8
        文本读取，然后逐行处理文件内容。普通行保持不变；@include 行
        会替换为引用文件展开后的正文。无法加载的文件或引用会被跳过，
        具体路径和原因追加到 warnings。

        Args:
            path: 当前需要读取和展开的指令文件路径。
            trust_root: 当前指令来源允许访问的根目录。path 及其引用文件解析后的路径都必须位于该目录内。
            depth: 当前文件的 @include 嵌套层数。顶层文件传入 0，每进入一层引用增加 1。
            visited: 本次加载过程中已经成功读取的文件路径。函数会把当前文件的解析路径加入该集合，并用它阻止重复加载和循环引用。
            warnings: 用于收集加载警告的列表。路径无效、文件不存在、文件无法读取、重复引用和嵌套超限等问题会追加到该列表。

        Returns:
            展开全部有效 @include 后的指令文本。如果当前文件无法加载，则返回空字符串。
        """

        try:
            resolved = path.resolve(strict=False)
        except OSError:
            warnings.append(InstructionWarning(path, "路径无法解析"))
            return ""
        if not _inside(trust_root, resolved):
            warnings.append(InstructionWarning(path, "路径超出允许目录"))
            return ""
        if resolved in visited:
            warnings.append(InstructionWarning(resolved, "文件已经加载或形成循环引用"))
            return ""
        if not resolved.exists():
            warnings.append(InstructionWarning(resolved, "文件不存在"))
            return ""
        if not resolved.is_file():
            warnings.append(InstructionWarning(resolved, "路径不是普通文件"))
            return ""
        try:
            content = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            warnings.append(InstructionWarning(resolved, f"文件无法读取：{exc}"))
            return ""

        visited.add(resolved)
        output: list[str] = []
        for line in content.splitlines(keepends=True):
            match = _INCLUDE_LINE.match(line.rstrip("\r\n"))
            if match is None:
               # 没有@include行就原样加进结果
               output.append(line)
               continue
            # 清理 @include 指令中捕获的路径文本
            target = _include_target(match.group(1))
            # target为空记录警告并处理下一行
            if not target:
                warnings.append(InstructionWarning(resolved, "@include 路径不能为空"))
                continue
            # 不为空才计算引用文件路径
            include_path = resolved.parent / target
            if depth >= INSTRUCTION_INCLUDE_MAX_DEPTH:
                warnings.append(
                    InstructionWarning(include_path, "@include 超过 5 层嵌套限制")
                )
                continue
            included = self._expand(
                include_path,
                trust_root=trust_root,
                depth=depth + 1,
                visited=visited,
                warnings=warnings,
            )
            if included:
                output.append(included)
                if not included.endswith("\n"):
                    output.append("\n")
        return "".join(output)

    def load(self) -> LoadedInstructions:
        """读取并拼接当前项目生效的三层 MYCODE.md 指令。

        函数按照项目根、项目本地、用户级的顺序读取 MYCODE.md，
        这个顺序也是指令从高到低的优先级。每个入口文件中的
        @include 会在对应的允许目录内递归展开。

        不存在的入口文件会被跳过。引用文件无法读取、路径越界、
        重复加载或嵌套超限时，函数跳过对应引用，并把路径和原因记录
        到返回结果的 warnings 中。成功加载的各段指令会标明来源，
        再按照优先级拼接成一段文本。

        Returns:
            包含拼接后指令文本和加载警告的 LoadedInstructions。
            没有加载到有效指令时，content 为空字符串。
    """
        # 用户级配置目录
        user_root = (Path.home() / ".mycode").resolve(strict=False)
        # 找到三个指令来源   项目根指令>项目本地指令>用户级指令
        # (读取的项目指令文件路径，@include最远读取到哪个目录)
        sources = (
            (self._workspace_root / "MYCODE.md", self._workspace_root),
            (
                self._workspace_root / ".mycode" / "MYCODE.md",
                self._workspace_root,
            ),
            (user_root / "MYCODE.md", user_root),
        )
        # 保存未能正常加载的@include指令及失败原因
        warnings: list[InstructionWarning] = []
        # 保存本次加载过程中已经读取过的文件路径
        visited: set[Path] = set()
        # 保存最终成功加载并展开的每一层指令文本
        sections: list[str] = []
        for path, trust_root in sources:
            if not path.exists():
                continue
            expanded = self._expand(
                path,
                trust_root=trust_root,
                depth=0,
                visited=visited,
                warnings=warnings,
            )
            if expanded.strip():
                sections.append(
                    f"### 指令来源：{path}\n\n{expanded.strip()}"
                )

        if not sections:
            return LoadedInstructions("", tuple(warnings))
        precedence = (
            "以下项目指令按优先级从高到低排列；内容冲突时，"
            "遵循排在前面的指令。"
        )
        return LoadedInstructions(
            f"{precedence}\n\n" + "\n\n---\n\n".join(sections),
            tuple(warnings),
        )
