"""稳定全局指令模块及其确定性拼装器。"""

from __future__ import annotations

from collections.abc import Iterable

from mycode.models.prompts import PromptSection


DEFAULT_PROMPT_SECTIONS = (
    PromptSection(
        "身份",
        100,
        "你是 MyCode，一个在用户工作区内协作的终端 AI 助手。",
    ),
    PromptSection(
        "行为",
        200,
        "持续推进当前任务，必要时反复调用工具，直到得到可验证的结果；收到工具结果后判断是否需要继续。",
    ),
    PromptSection(
        "工具使用",
        300,
        "优先使用合适的专用工具，只有没有合适专用工具时才使用通用命令。"
        "编辑已有文件前必须先读取目标内容。所有工具路径必须相对于工作区根目录，"
        "不得越过工作区边界。同一次回复中的并行工具调用必须彼此独立。",
    ),
    PromptSection(
        "代码质量",
        400,
        "保持改动聚焦、接口清晰，并用与风险相称的编译和测试证据验证结果。",
    ),
    PromptSection(
        "安全边界",
        500,
        "不得泄露密钥或敏感配置，不得绕过工具权限、工作区限制或用户授权边界。",
    ),
    PromptSection(
        "输出风格",
        600,
        "清晰、直接地回答；任务完成后说明结果和验证证据，不虚构已经执行的操作。",
    ),
)

PLAN_FULL_INSTRUCTION = (
    "当前处于仅规划模式。你可以读取和分析内容，但不得声称已经修改文件或执行命令。"
    "具有写入能力的工具会在执行前被拦截。请返回一份计划，其中应包含"
    "任务目标、当前发现、实施步骤、预期影响、验证方式，以及需要用户批准的事项。"
    "在实际执行前，请告知用户关闭仅规划模式并重新发送请求。"
)
PLAN_COMPACT_REMINDER = (
    "仍处于仅规划模式：只读分析，不执行写操作；继续按已给出的完整 Plan 规则工作。"
)


class PromptAssembler:
    """
       拼接提示词
    """
    def __init__(
        self,
        sections: Iterable[PromptSection] = DEFAULT_PROMPT_SECTIONS,
        *,
        project_instructions: str = "",
    ) -> None:
        """按 `(priority, name)` 排序并生成可缓存的稳定前缀。"""
        configured = tuple(sections)
        if project_instructions.strip():
            configured = (
                *configured,
                PromptSection(
                    "项目指令",
                    700,
                    "以下内容按出现顺序从高优先级到低优先级排列；"
                    "发生冲突时遵循靠前的明确规则。\n\n"
                    f"{project_instructions.strip()}",
                ),
            )
        ordered = tuple(
            sorted(configured, key=lambda item: (item.priority, item.name))
        )
        names = [section.name for section in ordered]
        if len(names) != len(set(names)):
            raise ValueError("提示模块名称不能重复")
        if not ordered:
            raise ValueError("至少需要一个提示模块")
        self._sections = ordered

    @property
    def sections(self) -> tuple[PromptSection, ...]:
        return self._sections

    def build(self) -> str:
        return "".join(
            f"## {section.name}\n\n{section.content.strip()}\n\n"
            for section in self._sections
        )
