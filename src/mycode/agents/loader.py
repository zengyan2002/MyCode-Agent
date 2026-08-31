"""扫描项目、用户和内置三层角色目录，并按优先级选择定义。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from mycode.agents.parser import AgentParseError, AgentParser
from mycode.models.agents import (
    AgentCandidate,
    AgentCatalogSnapshot,
    AgentDiagnostic,
    AgentDiagnosticLevel,
    AgentSource,
)

_DISCOVERY_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9-]{0,63}$")


@dataclass(frozen=True)
class AgentRoot:
    """把一个待扫描目录和它代表的角色来源放在一起。

    Attributes:
        source: Loader 交给 Parser 的项目、用户或内置来源。
        path: 只扫描当前层 ``*.md`` 文件的目录。
    """

    source: AgentSource
    path: Path


class AgentLoader:
    """读取三层角色候选，并选择最高优先级的有效版本。

    CLI 使用 :meth:`from_workspace` 固定项目、用户、内置顺序。Loader 不
    修改当前 Catalog；每次 :meth:`scan` 都返回一份新的不可变快照。

    Attributes:
        _parser: 逐文件解析 Markdown frontmatter 和角色正文的解析器。
        _roots: 按项目、用户、内置优先级排列的扫描目录。
        _enable_verification: 是否让内置 Verification 进入有效角色目录。
    """

    def __init__(
        self,
        parser: AgentParser,
        roots: tuple[AgentRoot, ...],
        *,
        enable_verification: bool = False,
    ) -> None:
        """创建 Loader 并固定来源优先级与 Verification 开关。

        Args:
            parser: 负责校验每个 Markdown 文件的解析器。
            roots: 从高到低排列的扫描目录。
            enable_verification: 是否允许名字为 Verification 的最终定义
                进入有效目录；候选文件无论开关如何都会被解析和诊断。

        Returns:
            不返回数据。
        """

        if not isinstance(enable_verification, bool):
            raise ValueError("Verification 开关必须是布尔值")
        self._parser = parser
        self._roots = roots
        self._enable_verification = enable_verification

    @classmethod
    def from_workspace(
        cls,
        parser: AgentParser,
        workspace_root: Path,
        *,
        user_home: Path | None = None,
        builtin_root: Path | None = None,
        enable_verification: bool = False,
    ) -> "AgentLoader":
        """根据当前工作区创建项目、用户和内置三个扫描根。

        Args:
            parser: 负责解析每个候选文件的 AgentParser。
            workspace_root: 当前项目根目录。
            user_home: 用户主目录；未传时使用 ``Path.home()``。
            builtin_root: 测试或打包检查指定的内置资源目录；未传时从
                ``mycode.resources.agents`` 包中取得。
            enable_verification: 是否把 Verification 放入最终有效目录。

        Returns:
            扫描顺序固定为 PROJECT、USER、BUILTIN 的 Loader。

        Raises:
            RuntimeError: 安装包中缺少内置角色资源目录。
        """

        workspace = workspace_root.resolve()
        home = (user_home or Path.home()).resolve()
        if builtin_root is None:
            try:
                traversable = resources.files("mycode.resources.agents")
                builtin = Path(str(traversable)).resolve()
            except (ModuleNotFoundError, TypeError) as exc:
                raise RuntimeError("安装包缺少内置 Agent 资源目录") from exc
        else:
            builtin = builtin_root.resolve()
        return cls(
            parser,
            (
                AgentRoot(
                    AgentSource.PROJECT,
                    workspace / ".mycode" / "agents",
                ),
                AgentRoot(
                    AgentSource.USER,
                    home / ".mycode" / "agents",
                ),
                AgentRoot(AgentSource.BUILTIN, builtin),
            ),
            enable_verification=enable_verification,
        )

    @property
    def roots(self) -> tuple[AgentRoot, ...]:
        """返回 Loader 实际使用的来源目录。

        Returns:
            按覆盖优先级排列的 AgentRoot 元组。
        """

        return self._roots

    def scan(self) -> AgentCatalogSnapshot:
        """扫描全部来源并选择每个名字最高优先级的有效定义。

        Returns:
            包含有效定义、全部候选和诊断的新快照。单文件损坏不会抛出，
            同名角色会继续尝试更低优先级来源。
        """

        candidates_by_name: dict[str, list[AgentCandidate]] = {}
        diagnostics: list[AgentDiagnostic] = []

        for root in self._roots:
            same_source_by_name: dict[str, list[AgentCandidate]] = {}
            for entry_path in self._discover(root.path):
                fallback_key = entry_path.stem.casefold()
                try:
                    definition = self._parser.parse(entry_path, root.source)
                except AgentParseError as exc:
                    diagnostic = AgentDiagnostic(
                        path=exc.path,
                        agent_name=(
                            entry_path.stem
                            if _DISCOVERY_NAME.fullmatch(entry_path.stem)
                            else None
                        ),
                        level=AgentDiagnosticLevel.WARNING,
                        message=exc.reason,
                    )
                    candidate = AgentCandidate(
                        source=root.source,
                        entry_path=entry_path.resolve(),
                        definition=None,
                        diagnostic=diagnostic,
                    )
                    diagnostics.append(diagnostic)
                    same_source_by_name.setdefault(fallback_key, []).append(candidate)
                    continue

                same_source_by_name.setdefault(definition.key, []).append(
                    AgentCandidate(
                        source=root.source,
                        entry_path=entry_path.resolve(),
                        definition=definition,
                        diagnostic=None,
                    )
                )

            for key, same_source in same_source_by_name.items():
                candidates_by_name.setdefault(key, []).extend(
                    self._reject_same_source_duplicates(
                        key,
                        same_source,
                        diagnostics,
                    )
                )

        selected = {}
        frozen_candidates: dict[str, tuple[AgentCandidate, ...]] = {}
        for key in sorted(candidates_by_name):
            ordered = tuple(candidates_by_name[key])
            frozen_candidates[key] = ordered
            if key == "verification" and not self._enable_verification:
                continue
            definition = next(
                (
                    candidate.definition
                    for candidate in ordered
                    if candidate.definition is not None
                ),
                None,
            )
            if definition is not None:
                selected[key] = definition

        return AgentCatalogSnapshot(
            definitions=selected,
            candidates=frozen_candidates,
            diagnostics=tuple(diagnostics),
        )

    def reload(
        self,
        previous: AgentCatalogSnapshot,
    ) -> AgentCatalogSnapshot:
        """重新扫描角色目录，不修改调用方持有的旧快照。

        Args:
            previous: 当前正在使用的快照；参数表明 reload 是新扫描，保留
                旧定义的决策交给 AgentCatalog。

        Returns:
            当前磁盘内容对应的新扫描快照。
        """

        del previous
        return self.scan()

    def _discover(self, root: Path) -> tuple[Path, ...]:
        """列出一个来源目录当前层的 Markdown 文件。

        Args:
            root: 项目、用户或内置角色目录。

        Returns:
            按规范化绝对路径排序的文件元组。目录不存在时返回空元组。
        """

        if not root.is_dir():
            return ()
        paths = (
            path.resolve()
            for path in root.iterdir()
            if path.is_file() and path.suffix.casefold() == ".md"
        )
        return tuple(
            sorted(paths, key=lambda path: path.as_posix().casefold())
        )

    def _reject_same_source_duplicates(
        self,
        key: str,
        candidates: list[AgentCandidate],
        diagnostics: list[AgentDiagnostic],
    ) -> tuple[AgentCandidate, ...]:
        """让同一来源内的多个有效同名定义一起失效。

        Args:
            key: 已经 ``casefold`` 的角色名。
            candidates: 同一来源中归到该名字的候选。
            diagnostics: 本次扫描的诊断列表；冲突原因会追加到这里。

        Returns:
            保持原路径顺序的新候选元组。没有冲突时原样返回；冲突时把
            每个有效定义替换成指向自身文件的诊断候选。
        """

        valid = [item for item in candidates if item.definition is not None]
        if len(valid) <= 1:
            return tuple(candidates)
        conflict_paths = "、".join(str(item.entry_path) for item in valid)
        normalized: list[AgentCandidate] = []
        for candidate in candidates:
            if candidate.definition is None:
                normalized.append(candidate)
                continue
            diagnostic = AgentDiagnostic(
                path=candidate.entry_path,
                agent_name=candidate.definition.name,
                level=AgentDiagnosticLevel.WARNING,
                message=(
                    f"同一来源存在多个同名 Agent {key!r}：{conflict_paths}"
                ),
            )
            diagnostics.append(diagnostic)
            normalized.append(
                AgentCandidate(
                    source=candidate.source,
                    entry_path=candidate.entry_path,
                    definition=None,
                    diagnostic=diagnostic,
                )
            )
        return tuple(normalized)
