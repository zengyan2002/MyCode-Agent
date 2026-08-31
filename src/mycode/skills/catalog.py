"""保存当前已经通过选择和校验的 Skill 快照。"""

from __future__ import annotations

from mycode.models.skills import (
    SkillCatalogSnapshot,
    SkillDefinition,
    SkillDiagnostic,
)


class SkillCatalog:
    """为命令、LoadSkill 和运行时提供当前有效 Skill。

    SkillService 在启动和 reload 成功后替换快照。其他模块只能查询副本，
    不能直接改写 Catalog 内部字典。
    """

    def __init__(
        self,
        snapshot: SkillCatalogSnapshot | None = None,
    ) -> None:
        """创建空 Catalog，或复制一份已有扫描快照。

        Args:
            snapshot: Loader 生成的初始快照；未传时从空集合开始。
        """

        initial = snapshot or SkillCatalogSnapshot()
        # 复制字典，避免调用方在构造后修改初始快照影响查询结果。
        self._skills = dict(initial.skills)
        # 候选信息供 /skill info 解释覆盖和回退。
        self._candidates = dict(initial.candidates)
        # 诊断按 Loader 的确定顺序保存。
        self._diagnostics = tuple(initial.diagnostics)

    def get(self, name: str) -> SkillDefinition | None:
        """按名字查找当前有效 Skill。

        Args:
            name: 用户命令或 LoadSkill 传入的名字。比较时忽略大小写和
                首尾空白。

        Returns:
            当前选中的 SkillDefinition；名称未知时返回 None。
        """

        return self._skills.get(name.strip().casefold())

    def list(self) -> tuple[SkillDefinition, ...]:
        """按名字返回全部当前有效 Skill。

        Returns:
            名称排序后的不可变 SkillDefinition 元组。
        """

        return tuple(
            self._skills[name]
            for name in sorted(self._skills)
        )

    @property
    def diagnostics(self) -> tuple[SkillDiagnostic, ...]:
        """返回当前快照的扫描诊断。

        Returns:
            Loader 产生的不可变诊断元组。
        """

        return self._diagnostics

    def snapshot(self) -> SkillCatalogSnapshot:
        """复制当前 Catalog，供 reload 比较和管理命令展示。

        Returns:
            字典与 Catalog 内部存储分离的新 SkillCatalogSnapshot。
        """

        return SkillCatalogSnapshot(
            skills=dict(self._skills),
            candidates=dict(self._candidates),
            diagnostics=self._diagnostics,
        )

    def replace(self, snapshot: SkillCatalogSnapshot) -> None:
        """一次性替换当前有效 Skill、候选和诊断。

        Args:
            snapshot: 已经完成解析、覆盖选择和外部依赖校验的新快照。

        Returns:
            None。调用结束后新的查询立即生效。
        """

        skills = dict(snapshot.skills)
        candidates = dict(snapshot.candidates)
        diagnostics = tuple(snapshot.diagnostics)
        self._skills = skills
        self._candidates = candidates
        self._diagnostics = diagnostics
