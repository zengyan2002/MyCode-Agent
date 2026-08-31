"""稳定提示词与可信运行时补充指令的领域模型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from html import escape


@dataclass(frozen=True)
class PromptSection:
    """一段可按优先级确定性拼装的稳定全局指令。"""

    name: str
    priority: int
    content: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("提示模块名称不能为空")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("提示模块优先级必须是整数")
        if self.priority < 0:
            raise ValueError("提示模块优先级不能为负数")
        if not self.content.strip():
            raise ValueError("提示模块内容不能为空")


class RuntimeInstructionKind(str, Enum):
    """
    定义只能由应用内部生成、不能直接来自用户输入的运行时指令类型。
    """

    #完整的运行环境信息，例如工作区、操作系统
    ENVIRONMENT_CONTEXT = "environment_context"
    #执行过程中发生的环境变化
    ENVIRONMENT_UPDATE = "environment_update"
    #当前模式的完整要求，例如Plan模式规则
    MODE_INSTRUCTION = "mode_instruction"
    #对当前模式的简短提醒
    MODE_REMINDER = "mode_reminder"
    #其它运行时的通知
    RUNTIME_NOTICE = "runtime_notice"
    # 历史压缩后保存的结构化摘要，不伪装成用户或助手消息。
    COMPACTION_CHECKPOINT = "compaction_checkpoint"
    # 提醒模型摘要不含文件细节，需要时必须重新读取 artifact 或源码。
    COMPACTION_BOUNDARY = "compaction_boundary"
    # 用户级或项目级长期笔记的索引；需要正文时再用 read_file 打开链接。
    LONG_TERM_MEMORY = "long_term_memory"
    # 启动时可用 Skill 的名字和一句说明，不包含 SOP 正文。
    SKILL_CATALOG = "skill_catalog"
    # 主 Agent 可委派的预定义角色名和一句用途，不包含角色系统提示正文。
    AGENT_CATALOG = "agent_catalog"
    # 当前会话已激活 Skill 的完整 SOP，后激活者排在更靠后的位置。
    ACTIVE_SKILL = "active_skill"
    # Hook 在生命周期事件中动态生成、只供下一次模型请求读取的提醒。
    HOOK_NOTIFICATION = "hook_notification"
    # 当前运行只剩少量模型调用次数时生成的收敛提醒，不写入 Conversation。
    MODEL_CALL_BUDGET = "model_call_budget"
    # 最后一次无工具请求使用的正式报告要求，不写入 Conversation。
    FINALIZATION = "finalization"


@dataclass(frozen=True)
class RuntimeInstruction:
    """一条运行时补充指令，会被渲染成带标签的文本，拼在提示词内"""
    kind: RuntimeInstructionKind
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RuntimeInstructionKind):
            raise ValueError("运行时指令必须包含有效种类")
        if not self.content.strip():
            raise ValueError("运行时指令内容不能为空")

    def render(self) -> str:
        """使用固定标签渲染，并阻止动态内容闭合外层标签。"""
        '''
        例如生成：
        <environment_context>
        工作区：D:/project
        操作系统：Windows
        </environment_context>
        '''
        tag = self.kind.value
        return f"<{tag}>\n{escape(self.content, quote=False)}\n</{tag}>"


@dataclass(frozen=True)
class PromptContext:
    """给Provider请求中的提示词，包含长期不变的基础提示词和本次请求临时追加的运行时指令"""

    stable: str
    runtime: tuple[RuntimeInstruction, ...] = ()

    def __post_init__(self) -> None:
        if not self.stable.strip():
            raise ValueError("提示上下文必须包含稳定文本")
        if not isinstance(self.runtime, tuple) or not all(
            isinstance(item, RuntimeInstruction) for item in self.runtime
        ):
            raise ValueError("运行时指令必须是 RuntimeInstruction 元组")
