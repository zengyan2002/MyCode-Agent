"""提供模型可见且参数列表稳定的统一 Agent SYSTEM 工具。"""

from __future__ import annotations

from collections.abc import Mapping

from mycode.agents.service import AgentService
from mycode.models.agents import AgentToolRequest
from mycode.models.json_types import JsonValue
from mycode.models.tools import ToolAccess, ToolDefinition, ToolErrorCode
from mycode.tools.base import ToolContext, ToolOutput


_AGENT_TOOL = ToolDefinition(
    name="Agent",
    description=(
        "把边界明确的子任务委派给独立子 Agent。指定 subagent_type 使用预定义角色；"
        "留空则 Fork 当前对话并在后台运行。创建团队成员时必须同时填写 team_name、"
        "name 和 subagent_type，且不要填写 run_in_background。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "minLength": 1, "pattern": r"\S"},
            "description": {"type": "string", "minLength": 1, "pattern": r"\S"},
            "name": {"type": "string", "minLength": 1, "pattern": r"\S"},
            "subagent_type": {
                "type": "string",
                "minLength": 1,
                "pattern": r"\S",
                "description": "预定义 Agent 角色名；创建团队成员时必填。",
            },
            "model": {"type": "string", "minLength": 1, "pattern": r"\S"},
            "run_in_background": {
                "type": "boolean",
                "description": "仅用于一次性 Agent；团队成员不能填写。",
            },
            "team_name": {
                "type": "string",
                "minLength": 1,
                "pattern": r"\S",
                "description": "创建长期成员时填写当前团队名称。",
            },
            "backend": {
                "type": "string",
                "enum": ["auto", "tmux", "iterm2", "in-process"],
            },
            "plan_mode_required": {"type": "boolean"},
        },
        "required": ["prompt", "description"],
        "additionalProperties": False,
        "allOf": [
            {
                "if": {"required": ["team_name"]},
                "then": {
                    "required": ["name", "subagent_type"],
                    "not": {"required": ["run_in_background"]},
                },
            },
            {
                "if": {"not": {"required": ["team_name"]}},
                "then": {
                    "not": {
                        "anyOf": [
                            {"required": ["backend"]},
                            {"required": ["plan_mode_required"]},
                        ]
                    }
                },
            },
        ],
    },
    access=ToolAccess.WRITE,
)


class AgentTool:
    """把模型参数转换成 AgentToolRequest，并交给 AgentService 执行。

    Attributes:
        _service: 应用中唯一的委派协调服务，负责定义式/Fork 分流和
            前后台运行。
    """

    def __init__(self, service: AgentService) -> None:
        """保存应用中唯一的 AgentService。

        Args:
            service: 负责角色/Fork 分流、独立运行和后台移交的协调对象。

        Returns:
            不返回数据。
        """

        self._service = service

    @property
    def definition(self) -> ToolDefinition:
        """返回统一 Agent 工具的名称、用途和固定 JSON Schema。

        Returns:
            注册为 SYSTEM 来源的只读 ToolDefinition。
        """

        return _AGENT_TOOL

    async def execute(
        self,
        arguments: Mapping[str, JsonValue],
        context: ToolContext,
    ) -> ToolOutput:
        """执行一次模型委派请求。

        Args:
            arguments: 已通过 Schema 校验的委派字段。
            context: 当前主 Agent 的工具上下文；服务已经持有会话和运行路由，
                因此这里不读取或修改上下文字段。

        Returns:
            前台子 Agent 结果、后台任务 ID，或结构化失败信息。
        """

        try:
            request = AgentToolRequest(
                prompt=str(arguments["prompt"]),
                description=str(arguments["description"]),
                name=_optional_string(arguments.get("name")),
                subagent_type=_optional_string(arguments.get("subagent_type")),
                model=_optional_string(arguments.get("model")),
                run_in_background=(
                    bool(arguments["run_in_background"])
                    if "run_in_background" in arguments
                    else None
                ),
                team_name=_optional_string(arguments.get("team_name")),
                backend=_optional_string(arguments.get("backend")),
                plan_mode_required=(
                    bool(arguments["plan_mode_required"])
                    if "plan_mode_required" in arguments
                    else None
                ),
            )
        except ValueError as exc:
            return ToolOutput.fail(ToolErrorCode.INVALID_ARGUMENTS, str(exc))
        if request.team_name is None:
            return await self._service.delegate(request)
        return await self._service.delegate(request, team_actor=context.team_actor)


def _optional_string(value: JsonValue | None) -> str | None:
    """把 Schema 已校验的可选 JSON 字符串转成 Python 可选字符串。

    Args:
        value: 工具参数中的可选值。

    Returns:
        未填写时返回 ``None``，填写时返回原字符串。
    """

    return value if isinstance(value, str) else None
