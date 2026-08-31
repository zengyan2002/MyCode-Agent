"""计算定义式、Fork 和后台子 Agent 最终可以看到的工具集合。"""

from __future__ import annotations

from mycode.models.agents import AgentDefinition, IndependentAgentOrigin
from mycode.models.tools import ToolView


CHILD_DENIED_TOOLS = frozenset(
    {
        "Agent",
        "TaskList",
        "TaskGet",
        "TaskStop",
        "AskUserQuestion",
        "SendMessage",
        "TeamCreate",
        "TeamGet",
        "TeamDelete",
        "TeamTakeover",
        "TeamMemberStop",
        "TeamTaskCreate",
        "TeamTaskList",
        "TeamTaskGet",
        "TeamTaskClaim",
        "TeamTaskUpdate",
    }
)

ASYNC_AGENT_ALLOWED_TOOLS = frozenset(
    {
        "read_file",
        "write_file",
        "edit_file",
        "execute_command",
        "find_files",
        "search_code",
        "load_skill",
        "LoadSkill",
    }
)


def build_child_tool_view(
    *,
    origin: IndependentAgentOrigin,
    parent_visible_names: frozenset[str] | None,
    background: bool,
    role: AgentDefinition | None,
    additional_allowlist: frozenset[str] | None = None,
) -> ToolView:
    """按固定顺序生成子 Agent 的全来源最终工具过滤条件。

    Args:
        origin: 定义式、普通 Fork 或 Skill Fork。
        parent_visible_names: 父 Provider 请求实际收到的工具名；普通 Fork
            必须提供，定义式和 Skill Fork 可以为 ``None``。
        background: 是否应用后台固定白名单。
        role: 定义式角色的白名单和黑名单；无角色时为 ``None``。
        additional_allowlist: fork Skill 的 allowedTools 等额外业务白名单；
            ``None`` 表示不增加这一层限制。

    Returns:
        写有 ``final_allowlist`` 和 ``denied_tool_names`` 的 ToolView。Registry
        会把这些限制应用到 BUILTIN、SYSTEM、MCP、SKILL 全部来源。

    Raises:
        ValueError: 普通 Fork 缺少父工具快照。
    """

    allowlist: frozenset[str] | None = None
    if origin is IndependentAgentOrigin.FORK:
        if parent_visible_names is None:
            raise ValueError("Fork 子 Agent 必须包含父 Agent 可见工具快照")
        allowlist = parent_visible_names
    if background:
        allowlist = _intersect_allowlist(
            allowlist,
            ASYNC_AGENT_ALLOWED_TOOLS,
        )
    if role is not None and role.tools is not None:
        allowlist = _intersect_allowlist(allowlist, role.tools)
    if additional_allowlist is not None:
        allowlist = _intersect_allowlist(allowlist, additional_allowlist)
    denied = set(CHILD_DENIED_TOOLS)
    if role is not None:
        denied.update(role.disallowed_tools)
    return ToolView(
        final_allowlist=allowlist,
        denied_tool_names=frozenset(denied),
    )


def _intersect_allowlist(
    current: frozenset[str] | None,
    restriction: frozenset[str],
) -> frozenset[str]:
    """把一个新白名单限制叠加到已有白名单。

    Args:
        current: 已有白名单；``None`` 表示此前没有收窄。
        restriction: 当前层允许保留的工具名。

    Returns:
        首层限制的副本，或两个白名单的交集。
    """

    return restriction if current is None else current & restriction
