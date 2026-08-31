"""会话旁路文件保存的组合运行状态。"""

from __future__ import annotations

from dataclasses import dataclass, field

from mycode.models.skills import SkillSessionState
from mycode.models.teams import TeamBinding


@dataclass(frozen=True, slots=True)
class SessionRuntimeMetadata:
    """保存恢复主会话时需要的 Skill 状态和可选团队绑定。

    Attributes:
        skills: 按激活顺序保存的 inline Skill 名称。
        team: 当前会话管理的团队；普通会话为 None。
    """

    skills: SkillSessionState = field(default_factory=SkillSessionState)
    team: TeamBinding | None = None
