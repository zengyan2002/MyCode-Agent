"""扫描三层 Skill 目录，并为同名 Skill 选择可用版本。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from mycode.models.skills import (
    SkillCandidate,
    SkillCatalogSnapshot,
    SkillDiagnostic,
    SkillDiagnosticLevel,
    SkillRefreshResult,
    SkillSource,
)
from mycode.skills.parser import SkillParseError, SkillParser

_DISCOVERY_NAME = re.compile(r"^[a-z][a-z0-9-]{0,63}$")


@dataclass(frozen=True)
class SkillRoot:
    """把一个待扫描目录和它代表的优先级来源放在一起。"""

    # Loader 交给 Parser 的来源类别。
    source: SkillSource
    # 该来源实际扫描的目录。
    path: Path


class SkillLoader:
    """从项目级、用户级和内置级目录读取 Skill。

    CLI 通常使用 from_workspace 创建它。测试和其他明确调用方也可以直接
    传入 roots，以便扫描一个确定的目录集合。
    """

    def __init__(
        self,
        parser: SkillParser,
        roots: tuple[SkillRoot, ...],
    ) -> None:
        """创建 Loader，并固定来源扫描顺序。

        Args:
            parser: 负责解析每个 Markdown 候选的 SkillParser。
            roots: 从高到低排列的来源目录。
        """

        # Parser 只做单文件校验，目录发现和覆盖选择由 Loader 负责。
        self._parser = parser
        # 元组避免 reload 过程中调用方改变优先级。
        self._roots = roots

    @classmethod
    def from_workspace(
        cls,
        parser: SkillParser,
        workspace_root: Path,
        *,
        user_home: Path | None = None,
        builtin_root: Path | None = None,
    ) -> "SkillLoader":
        """根据当前工作区构造项目、用户和包资源三个扫描根。

        Args:
            parser: 负责解析候选文件的 SkillParser。
            workspace_root: CLI 已解析出的绝对工作区根目录。
            user_home: 当前用户主目录；未传时使用 Path.home()。
            builtin_root: 测试或打包检查显式提供的内置资源目录。未传时
                从 mycode.resources.skills 包资源取得。

        Returns:
            扫描顺序固定为 PROJECT、USER、BUILTIN 的 SkillLoader。

        Raises:
            ValueError: workspace_root 或 user_home 不是绝对路径。
            RuntimeError: 安装包中缺少内置 Skill 资源目录。
        """

        workspace = workspace_root.resolve()
        home = (user_home or Path.home()).resolve()
        if not workspace.is_absolute():
            raise ValueError("Skill 工作区根目录必须是绝对路径")
        if not home.is_absolute():
            raise ValueError("Skill 用户目录必须是绝对路径")
        if builtin_root is None:
            try:
                traversable = resources.files("mycode.resources.skills")
                builtin = Path(str(traversable)).resolve()
            except (ModuleNotFoundError, TypeError) as exc:
                raise RuntimeError("安装包缺少内置 Skill 资源目录") from exc
        else:
            builtin = builtin_root.resolve()
        return cls(
            parser,
            (
                SkillRoot(
                    SkillSource.PROJECT,
                    workspace / ".mycode" / "skills",
                ),
                SkillRoot(
                    SkillSource.USER,
                    home / ".mycode" / "skills",
                ),
                SkillRoot(SkillSource.BUILTIN, builtin),
            ),
        )

    @property
    def roots(self) -> tuple[SkillRoot, ...]:
        """返回当前固定的来源目录和优先级顺序。

        Returns:
            构造 Loader 时保存的 SkillRoot 元组。
        """

        return self._roots

    def scan(self) -> SkillCatalogSnapshot:
        """扫描全部来源，并为每个名字选择最高优先级的有效候选。

        Returns:
            包含最终定义、全部候选和逐文件诊断的新快照。单个文件失败
            不会抛出异常，也不会阻止其他名字加载。
        """

        candidates_by_name: dict[str, list[SkillCandidate]] = {}
        diagnostics: list[SkillDiagnostic] = []

        for root in self._roots:
            source_candidates: dict[str, list[SkillCandidate]] = {}
            for entry_path in self._discover(root.path):
                fallback_name = self._name_from_location(entry_path)
                try:
                    definition = self._parser.parse(entry_path, root.source)
                except SkillParseError as exc:
                    diagnostic = SkillDiagnostic(
                        path=exc.path,
                        skill_name=(
                            fallback_name
                            if _DISCOVERY_NAME.fullmatch(fallback_name)
                            else None
                        ),
                        level=SkillDiagnosticLevel.WARNING,
                        message=exc.reason,
                    )
                    candidate = SkillCandidate(
                        source=root.source,
                        entry_path=entry_path.resolve(),
                        definition=None,
                        diagnostic=diagnostic,
                    )
                    diagnostics.append(diagnostic)
                    source_candidates.setdefault(
                        fallback_name,
                        [],
                    ).append(candidate)
                    continue

                candidate = SkillCandidate(
                    source=root.source,
                    entry_path=entry_path.resolve(),
                    definition=definition,
                    diagnostic=None,
                )
                source_candidates.setdefault(
                    definition.name,
                    [],
                ).append(candidate)

            for name, same_source in source_candidates.items():
                normalized = self._reject_same_source_duplicates(
                    name,
                    same_source,
                    diagnostics,
                )
                candidates_by_name.setdefault(name, []).extend(normalized)

        selected: dict[str, object] = {}
        frozen_candidates: dict[str, tuple[SkillCandidate, ...]] = {}
        for name in sorted(candidates_by_name):
            ordered = tuple(candidates_by_name[name])
            frozen_candidates[name] = ordered
            for candidate in ordered:
                if candidate.definition is not None:
                    selected[name] = candidate.definition
                    break

        return SkillCatalogSnapshot(
            skills={
                name: definition
                for name, definition in selected.items()
            },
            candidates=frozen_candidates,
            diagnostics=tuple(diagnostics),
        )

    def reload(
        self,
        previous: SkillCatalogSnapshot,
    ) -> SkillCatalogSnapshot:
        """重新扫描全部来源，供 Service 与旧快照逐 Skill 比较。

        Args:
            previous: 当前正在使用的快照。Loader 不修改它；参数明确提醒
                调用方 reload 是一次新扫描，而不是就地修改。

        Returns:
            全新的扫描快照。旧版保留或局部提交由 SkillService 决定。
        """

        del previous
        return self.scan()

    def read_latest_body(
        self,
        skill: object,
    ) -> SkillRefreshResult:
        """执行前重新读取一个已经选中的 Skill。

        Args:
            skill: 当前 Catalog 中的 SkillDefinition。使用 object 标注会
                在运行时做明确检查，避免循环导入只为类型检查。

        Returns:
            新版本有效时返回 definition；文件删除时 missing 为 True；
            新版本无效时返回 diagnostic，调用方可以继续使用缓存。

        Raises:
            TypeError: 调用方传入的不是 SkillDefinition。
        """

        from mycode.models.skills import SkillDefinition

        if not isinstance(skill, SkillDefinition):
            raise TypeError("read_latest_body 需要 SkillDefinition")
        if not skill.entry_path.is_file():
            return SkillRefreshResult(
                definition=None,
                diagnostic=SkillDiagnostic(
                    path=skill.entry_path,
                    skill_name=skill.name,
                    level=SkillDiagnosticLevel.WARNING,
                    message="Skill 入口已经删除",
                ),
                missing=True,
            )
        try:
            refreshed = self._parser.parse(skill.entry_path, skill.source)
        except SkillParseError as exc:
            return SkillRefreshResult(
                definition=None,
                diagnostic=SkillDiagnostic(
                    path=exc.path,
                    skill_name=skill.name,
                    level=SkillDiagnosticLevel.WARNING,
                    message=(
                        f"新内容无效，继续使用上一次有效版本：{exc.reason}"
                    ),
                ),
            )
        return SkillRefreshResult(
            definition=refreshed,
            diagnostic=None,
        )

    def _discover(self, root: Path) -> tuple[Path, ...]:
        """发现根目录单文件和一级子目录 SKILL.md。

        Args:
            root: 当前项目级、用户级或内置级扫描根。

        Returns:
            按规范化绝对路径排序的入口文件。目录不存在时返回空元组。
        """

        if not root.is_dir():
            return ()
        found: list[Path] = [
            path
            for path in root.glob("*.md")
            if path.is_file()
        ]
        found.extend(
            child / "SKILL.md"
            for child in root.iterdir()
            if child.is_dir() and (child / "SKILL.md").is_file()
        )
        unique = {path.resolve() for path in found}
        return tuple(sorted(unique, key=lambda path: str(path).casefold()))

    def _name_from_location(self, entry_path: Path) -> str:
        """在文件内容损坏时从约定位置推测覆盖目标名字。

        Args:
            entry_path: 扫描到的单文件入口或 SKILL.md。

        Returns:
            单文件使用文件名，目录型使用父目录名，并统一转成小写。
        """

        raw_name = (
            entry_path.parent.name
            if entry_path.name == "SKILL.md"
            else entry_path.stem
        )
        return raw_name.casefold()

    def _reject_same_source_duplicates(
        self,
        name: str,
        candidates: list[SkillCandidate],
        diagnostics: list[SkillDiagnostic],
    ) -> tuple[SkillCandidate, ...]:
        """让同一来源里的多个同名有效候选一起失效。

        Args:
            name: 候选解析出的统一 Skill 名。
            candidates: 同一来源中归到该名字下的全部候选。
            diagnostics: 本次扫描的诊断列表；重复错误会追加到这里。

        Returns:
            保持原路径顺序的候选。没有冲突时原样返回；有冲突时把有效
            候选替换成失败候选，使选择逻辑可以继续尝试低优先级来源。
        """

        valid = [
            candidate
            for candidate in candidates
            if candidate.definition is not None
        ]
        if len(valid) <= 1:
            return tuple(candidates)

        conflicting_paths = "、".join(
            str(candidate.entry_path)
            for candidate in valid
        )
        normalized: list[SkillCandidate] = []
        for candidate in candidates:
            if candidate.definition is None:
                normalized.append(candidate)
                continue
            diagnostic = SkillDiagnostic(
                path=candidate.entry_path,
                skill_name=name,
                level=SkillDiagnosticLevel.WARNING,
                message=(
                    "同一来源存在多个同名 Skill，当前候选不参与选择："
                    f"{conflicting_paths}"
                ),
            )
            diagnostics.append(diagnostic)
            normalized.append(
                SkillCandidate(
                    source=candidate.source,
                    entry_path=candidate.entry_path,
                    definition=None,
                    diagnostic=diagnostic,
                )
            )
        return tuple(normalized)
