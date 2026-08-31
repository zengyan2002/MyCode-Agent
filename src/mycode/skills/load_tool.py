"""提供 Agent 按轻量目录加载完整 Skill 的系统工具。"""

from __future__ import annotations

from collections.abc import Mapping

from mycode.agent.cancellation import CancellationToken
from mycode.errors import MyCodeError
from mycode.models.json_types import JsonValue
from mycode.models.tools import ToolAccess, ToolDefinition, ToolErrorCode
from mycode.skills.service import SkillService
from mycode.tools.base import ToolContext, ToolOutput

_LOAD_SKILL = ToolDefinition(
    name="LoadSkill",
    description=(
        "按名字加载一个可用 Skill 的完整工作流程。仅在用户需求与 Skill "
        "目录说明匹配时调用；可选 arguments 会填入该 Skill 的参数占位符。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "arguments": {"type": "string"},
        },
        "required": ["name"],
        "additionalProperties": False,
    },
    access=ToolAccess.READ,
)


class SkillLoadRouter:
    """把主 Agent 的 LoadSkill 工具调用送到 SkillService。

    独立 Agent 不使用本路由，而是在自己的 ToolContext 中安装只修改当前
    SkillRuntime 的路由，因此 nested Skill 不会创建下一层 Agent。
    """

    def __init__(self) -> None:
        """创建尚未绑定运行目标的路由。

        Returns:
            None。
        """

        # main scope 最终绑定应用唯一的 SkillService。
        self._main_service: SkillService | None = None

    def bind_main(self, service: SkillService) -> None:
        """绑定主 Agent 的 SkillService。

        Args:
            service: 负责热读、inline 激活和 fork 启动的应用服务。

        Returns:
            None。
        """

        self._main_service = service

    async def load(
        self,
        scope: str,
        name: str,
        arguments: str,
    ) -> str:
        """按 ToolContext 范围加载 Skill，并返回工具结果文字。

        Args:
            scope: 主会话固定传入 ``main``；其他值说明路由装配错误。
            name: Agent 从轻量目录选择的 Skill 名。
            arguments: 替换 $ARGUMENTS 的用户补充。

        Returns:
            inline 激活确认或 fork Skill 的最终报告，不包含 SOP 正文。

        Raises:
            MyCodeError: 目标尚未绑定或 Skill 调用失败。
        """

        if scope != "main":
            raise MyCodeError("主 Skill 路由只接受 main 范围")
        if self._main_service is None:
            raise MyCodeError("主 Skill 加载器尚未初始化")
        result = await self._main_service.load_for_agent(
            name,
            arguments,
            CancellationToken(),
        )
        if result.final_text is not None:
            content = result.final_text
        else:
            content = f"Skill {result.skill.name} 已激活"
        if result.warning:
            content = f"{result.warning}\n{content}"
        return content


class LoadSkillTool:
    """让当前 Agent 激活 inline Skill 或运行 fork Skill。

    主 Agent 和独立 Agent 共用同一个无状态工具实例。execute 从 ToolContext
    读取路由和范围，再选择主 SkillService 或当前 SkillForkRunner。两条路径
    都只返回简短确认或最终报告，不把完整 SKILL.md 复制进工具历史。
    """

    @property
    def definition(self) -> ToolDefinition:
        """返回模型可见的 LoadSkill 名称、说明和输入格式。

        Returns:
            来源会由 CLI 注册成 SYSTEM 的只读 ToolDefinition。
        """

        return _LOAD_SKILL

    async def execute(
        self,
        arguments: Mapping[str, JsonValue],
        context: ToolContext,
    ) -> ToolOutput:
        """加载目标 Skill，并返回不含 SOP 正文的结果。

        Args:
            arguments: 已通过 Schema 校验的 name 和可选 arguments 字符串。
            context: 当前工具工作区；加载流程不直接使用它读写文件。

        Returns:
            inline 返回简短激活确认，fork 返回最终 assistant 报告；未知或
            无效 Skill 返回 NOT_FOUND 或 IO_ERROR。
        """

        name = str(arguments["name"]).strip()
        user_arguments = str(arguments.get("arguments", "")).strip()
        try:
            router = context.skill_load_router
            if router is None:
                raise MyCodeError("Skill 加载路由尚未初始化")
            content = await router.load(
                context.skill_load_scope,
                name,
                user_arguments,
            )
        except MyCodeError as exc:
            code = (
                ToolErrorCode.NOT_FOUND
                if "未知 Skill" in str(exc) or "已经删除" in str(exc)
                else ToolErrorCode.IO_ERROR
            )
            return ToolOutput.fail(code, str(exc))

        return ToolOutput.ok(content)
