"""保存当前生效的角色定义，并按角色逐项安装 reload 结果。"""

from __future__ import annotations

from mycode.models.agents import (
    AgentCatalogSnapshot,
    AgentDefinition,
    AgentDiagnostic,
    AgentDiagnosticLevel,
    AgentReloadReport,
)


class AgentCatalog:
    """为主提示、Agent 工具和管理命令提供当前有效角色目录。

    Catalog 保存 Loader 最近一次成功安装的定义。reload 时，一个角色的
    新定义损坏或引用未知工具，只会保留该角色的旧版本，不影响其他角色。

    Attributes:
        _known_tool_names: 角色白黑名单允许引用的已注册工具名；尚未完成
            应用启动注册时为 ``None``。
        _snapshot: 当前 Agent 工具、提示目录和管理命令共同读取的有效定义。
    """

    def __init__(
        self,
        snapshot: AgentCatalogSnapshot,
        *,
        known_tool_names: frozenset[str] | None = None,
    ) -> None:
        """创建目录并记录可以被角色白黑名单引用的工具名。

        Args:
            snapshot: 启动时 Loader 生成的初始快照。
            known_tool_names: 当前 ToolRegistry 已注册的全部工具名；``None``
                表示暂不做注册表校验，适合 Parser/Loader 的独立调用方。

        Returns:
            不返回数据；初始快照校验通过后立即成为当前目录。
        """

        self._known_tool_names = known_tool_names
        self._snapshot = AgentCatalogSnapshot({}, {}, ())
        self.replace(snapshot)

    def get(self, name: str) -> AgentDefinition | None:
        """按大小写无关的名字查询当前有效角色。

        Args:
            name: 用户或模型提供的正式角色名。

        Returns:
            当前有效定义；名字为空或不存在时返回 ``None``。
        """

        if not isinstance(name, str) or not name.strip():
            return None
        return self._snapshot.definitions.get(name.strip().casefold())

    def list(self) -> tuple[AgentDefinition, ...]:
        """稳定列出当前所有有效角色。

        Returns:
            按正式角色名大小写无关排序的定义元组。
        """

        return tuple(
            sorted(
                self._snapshot.definitions.values(),
                key=lambda definition: definition.name.casefold(),
            )
        )

    @property
    def snapshot(self) -> AgentCatalogSnapshot:
        """返回当前不可变目录快照。

        Returns:
            Agent 工具和管理命令共同读取的当前快照。
        """

        return self._snapshot

    def set_known_tool_names(self, names: frozenset[str]) -> None:
        """设置角色白黑名单可以引用的已注册工具名。

        应用在 MCP 和 Skill 扫描完成后调用本方法。随后执行 reload，角色
        才会依据完整工具表逐项安装，避免把尚未注册的 MCP 工具误判为错误。

        Args:
            names: 当前共享 ToolRegistry 中已经注册的全部工具名。

        Returns:
            不返回数据；之后的 ``replace`` 和 ``install_reload`` 使用这份
            不可变名称集合校验角色定义。
        """

        self._known_tool_names = frozenset(names)

    def replace(self, snapshot: AgentCatalogSnapshot) -> None:
        """用一份完整且可安装的快照替换当前目录。

        Args:
            snapshot: Loader 生成的新快照。

        Returns:
            不返回数据；校验成功后当前目录立即指向新快照。

        Raises:
            ValueError: 任一有效定义引用 ToolRegistry 中不存在的工具。
        """

        diagnostics = self._tool_diagnostics(snapshot)
        if diagnostics:
            messages = "；".join(item.message for item in diagnostics)
            raise ValueError(messages)
        self._snapshot = snapshot

    def install_reload(
        self,
        scanned: AgentCatalogSnapshot,
    ) -> AgentReloadReport:
        """逐角色比较新扫描结果，并保留损坏角色的旧定义。

        Args:
            scanned: AgentLoader 刚生成的磁盘快照。

        Returns:
            新增、更新、删除、保留的角色名和全部诊断。当前目录也会在
            返回前切换到合并后的快照。
        """

        old = dict(self._snapshot.definitions)
        merged: dict[str, AgentDefinition] = {}
        diagnostics = list(scanned.diagnostics)
        added: list[str] = []
        updated: list[str] = []
        removed: list[str] = []
        retained: list[str] = []

        all_keys = sorted(set(old) | set(scanned.candidates) | set(scanned.definitions))
        for key in all_keys:
            previous = old.get(key)
            proposed = scanned.definitions.get(key)
            if proposed is not None:
                tool_issue = self._definition_tool_diagnostic(proposed)
                if tool_issue is not None:
                    diagnostics.append(tool_issue)
                    if previous is not None:
                        merged[key] = previous
                        retained.append(previous.name)
                    continue
                merged[key] = proposed
                if previous is None:
                    added.append(proposed.name)
                elif previous.revision != proposed.revision:
                    updated.append(proposed.name)
                continue

            has_candidates = key in scanned.candidates
            if previous is not None and has_candidates:
                merged[key] = previous
                retained.append(previous.name)
            elif previous is not None:
                removed.append(previous.name)

        self._snapshot = AgentCatalogSnapshot(
            definitions=merged,
            candidates=scanned.candidates,
            diagnostics=tuple(diagnostics),
        )
        return AgentReloadReport(
            added=tuple(sorted(added, key=str.casefold)),
            updated=tuple(sorted(updated, key=str.casefold)),
            removed=tuple(sorted(removed, key=str.casefold)),
            retained=tuple(sorted(retained, key=str.casefold)),
            diagnostics=tuple(diagnostics),
        )

    def _tool_diagnostics(
        self,
        snapshot: AgentCatalogSnapshot,
    ) -> tuple[AgentDiagnostic, ...]:
        """检查快照中每个有效角色引用的工具是否已经注册。

        Args:
            snapshot: 准备安装的角色快照。

        Returns:
            每个未知工具角色对应的诊断元组；无需校验或全部合法时为空。
        """

        diagnostics = []
        for definition in snapshot.definitions.values():
            diagnostic = self._definition_tool_diagnostic(definition)
            if diagnostic is not None:
                diagnostics.append(diagnostic)
        return tuple(diagnostics)

    def _definition_tool_diagnostic(
        self,
        definition: AgentDefinition,
    ) -> AgentDiagnostic | None:
        """检查单个角色白黑名单中的工具名。

        Args:
            definition: 准备进入目录的角色定义。

        Returns:
            发现未知工具时返回具体诊断；没有注册表约束或全部命中时
            返回 ``None``。
        """

        if self._known_tool_names is None:
            return None
        referenced = set(definition.disallowed_tools)
        if definition.tools is not None:
            referenced.update(definition.tools)
        unknown = sorted(referenced - self._known_tool_names)
        if not unknown:
            return None
        return AgentDiagnostic(
            path=definition.entry_path,
            agent_name=definition.name,
            level=AgentDiagnosticLevel.ERROR,
            message=f"Agent 引用了未注册工具：{', '.join(unknown)}",
        )
