"""Skill 加载、运行和持久化共同使用的数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from mycode.models.json_types import JsonObject
from mycode.models.tools import ToolAccess


class SkillSource(str, Enum):
    """说明一个 Skill 实际从哪一层目录加载。"""

    # 当前工作区中的 .mycode/skills，优先级最高。
    PROJECT = "project"
    # 当前用户主目录中的 .mycode/skills，可供多个项目复用。
    USER = "user"
    # 随 mycode Python 包一起安装的资源，作为最后的默认版本。
    BUILTIN = "builtin"


class SkillMode(str, Enum):
    """说明 Skill 是沿用主对话，还是在临时对话中执行。"""

    # SOP 加入当前主对话，后续轮次仍然生效。
    INLINE = "inline"
    # SOP 只在一次临时对话中生效，结束后只返回最终回复。
    FORK = "fork"


class SkillContextMode(str, Enum):
    """说明 fork Skill 可以从主对话带走多少历史。"""

    # 不带任何主对话历史。
    NONE = "none"
    # 带最近五个完整用户轮次。
    RECENT = "recent"
    # 带当前仍可用的全部主对话历史。
    FULL = "full"


class SkillDiagnosticLevel(str, Enum):
    """区分可以继续运行的提示和导致候选失效的错误。"""

    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class SkillToolSpec:
    """保存 tool.json 中一个已经通过静态校验的专属工具。

    SkillParser 创建该对象，后续工具注册表用它生成模型可见定义，
    SkillSubprocessTool 用它启动实际脚本。
    """

    # 注册给模型的全局工具名。
    name: str
    # 告诉模型这个工具具体能完成什么。
    description: str
    # function calling 用来校验模型参数的 JSON Schema。
    input_schema: JsonObject
    # 工具是只读还是可能改变外部状态；Plan 和权限检查依赖此值。
    access: ToolAccess
    # 原始命令数组；第一项是运行时，第二项是 Skill 内脚本。
    command: tuple[str, ...]
    # command 第二项解析出的真实脚本路径。
    entry_path: Path
    # 提供 tool.json、scripts 和 references 的目录型 Skill 根目录。
    skill_root: Path
    # 单次子进程最多允许运行多少秒。
    timeout_seconds: float
    # 标准输出最多允许返回多少 UTF-8 字节。
    max_output_bytes: int


@dataclass(frozen=True)
class SkillDefinition:
    """代表一个已经解析成功、可以交给 Agent 使用的 Skill。

    Loader 从磁盘生成它；Catalog 保存当前选中的版本；Runtime 在激活时
    读取其中的 SOP、模式、工具白名单和资源目录。
    """

    # Skill 的唯一名字，同时也是动态斜杠命令名。
    name: str
    # 启动时提供给 Agent 和 /help 的一句功能说明。
    description: str
    # None 表示不限制；空集合表示不提供业务工具；非空集合是精确白名单。
    allowed_tools: frozenset[str] | None
    # 当前 Skill 使用主对话还是临时 fork 对话。
    mode: SkillMode
    # fork 时从主对话复制历史的范围。
    context: SkillContextMode
    # 本次 Skill 可选的模型 ID；None 表示沿用当前 Provider 配置。
    model: str | None
    # 实际选中的项目级、用户级或内置级来源。
    source: SkillSource
    # 单文件入口或目录型 Skill 的 SKILL.md 真实路径。
    entry_path: Path
    # 目录型 Skill 的根目录；单文件 Skill 为 None，不扩大 ReadFile 范围。
    root_path: Path | None
    # 去掉 YAML frontmatter 后、准备注入 Agent 的 SOP 正文。
    prompt_body: str
    # 目录型 Skill 通过 tool.json 声明的专属工具。
    tools: tuple[SkillToolSpec, ...]
    # 入口文件和 tool.json 内容计算出的摘要，用来识别热更新。
    revision: str

    @property
    def is_directory(self) -> bool:
        """判断当前定义是否来自带 SKILL.md 的目录型 Skill。

        Returns:
            目录型 Skill 返回 True；单文件 Skill 返回 False。
        """

        return self.root_path is not None


@dataclass(frozen=True)
class SkillDiagnostic:
    """记录一个 Skill 文件为什么被跳过或继续使用缓存。

    Loader 和 Service 产生该对象，/skill reload 与启动错误展示会直接
    使用其中的路径和说明。
    """

    # 发生问题的 Markdown 或 tool.json 路径。
    path: Path
    # 能识别出名字时保存 Skill 名；无法识别时为 None。
    skill_name: str | None
    # warning 表示已回退或跳过，error 表示需要拒绝当前操作。
    level: SkillDiagnosticLevel
    # 给用户看的具体原因，不包含 traceback。
    message: str


@dataclass(frozen=True)
class SkillCandidate:
    """保存扫描到的一个候选及其解析结果。

    candidates 主要供覆盖回退、/skill info 和 reload 诊断使用。
    definition 与 diagnostic 只会有一个有值。
    """

    # 该候选所在的优先级来源。
    source: SkillSource
    # 扫描到的入口文件真实路径。
    entry_path: Path
    # 解析成功后的定义；解析失败时为 None。
    definition: SkillDefinition | None
    # 解析失败原因；解析成功时为 None。
    diagnostic: SkillDiagnostic | None

    def __post_init__(self) -> None:
        """检查候选不会同时处于成功和失败状态。

        Raises:
            ValueError: definition 与 diagnostic 同时有值或同时为空。
        """

        if (self.definition is None) == (self.diagnostic is None):
            raise ValueError("Skill 候选必须且只能包含定义或诊断")


@dataclass(frozen=True)
class SkillCatalogSnapshot:
    """保存一次完整扫描选出的 Skill 和所有候选信息。

    Loader 创建快照，Catalog 会复制字典后再保存，避免调用方修改当前
    对外可见的 Skill 集合。
    """

    # 按规范化名字索引的最终有效 Skill。
    skills: dict[str, SkillDefinition] = field(default_factory=dict)
    # 按名字保存从高到低的候选，方便解释覆盖和回退。
    candidates: dict[str, tuple[SkillCandidate, ...]] = field(
        default_factory=dict
    )
    # 本次扫描产生的全部警告和错误。
    diagnostics: tuple[SkillDiagnostic, ...] = ()


@dataclass(frozen=True)
class SkillRefreshResult:
    """说明执行前重读 Skill 得到了新版本、缓存回退还是文件缺失。"""

    # 新版本解析成功时保存定义；否则为 None。
    definition: SkillDefinition | None
    # 新版本无效时保存警告；读取成功时为 None。
    diagnostic: SkillDiagnostic | None
    # 入口文件已经删除时为 True。
    missing: bool = False


@dataclass(frozen=True)
class ActiveSkill:
    """保存主会话中一个已激活 inline Skill 的运行状态。"""

    # Catalog 中的 Skill 名。
    name: str
    # 本会话内从 1 开始的激活顺序。
    activated_order: int
    # 当前活动 SOP 来自哪个磁盘版本。
    revision: str
    # 本次激活时替换 $ARGUMENTS 使用的原始参数。
    arguments: str


@dataclass(frozen=True)
class SkillInvocation:
    """保存用户显式调用 Skill 时需要传给运行层的信息。"""

    # 目标 Skill 名，不含斜杠。
    name: str
    # Skill 名之后的全部原始文本，去掉首尾空白。
    arguments: str
    # UI 和主会话中保留的简短命令文本。
    display_text: str


@dataclass(frozen=True)
class SkillInvocationResult:
    """告诉 Application 一次 Skill 调用需要走主循环还是返回 fork 报告。"""

    # 实际执行时使用的最新有效 Skill 定义。
    skill: SkillDefinition
    # 主界面和主会话中保存的简短用户调用文字。
    display_text: str
    # fork 成功后的最终 assistant 文字；inline 激活时为 None。
    final_text: str | None = None
    # 执行前热读失败并回退缓存时，给用户展示的明确提醒。
    warning: str | None = None

    @property
    def is_fork(self) -> bool:
        """判断结果是否来自已经完成的独立执行。

        Returns:
            有最终 fork 回复时返回 True；需要进入主 Agent Loop 时返回 False。
        """

        return self.final_text is not None


@dataclass(frozen=True)
class SkillReloadIssue:
    """保存 reload 中一个未应用项目的名字和具体原因。"""

    # 受影响的 Skill 名。
    name: str
    # 新版本没有应用的原因。
    reason: str


@dataclass(frozen=True)
class SkillReloadReport:
    """汇总一次 /skill reload 对每个 Skill 实际做了什么。"""

    # 本次成功加入 Catalog 的名字。
    added: tuple[str, ...] = ()
    # 本次成功替换为新版本的名字。
    updated: tuple[str, ...] = ()
    # 因文件删除而移除的名字。
    removed: tuple[str, ...] = ()
    # 新版本无效、继续使用旧缓存的项目。
    retained: tuple[SkillReloadIssue, ...] = ()
    # 没有旧版本且当前候选无效的项目。
    skipped: tuple[SkillReloadIssue, ...] = ()
    # 因删除或改成 fork 而从主会话取消激活的名字。
    deactivated: tuple[str, ...] = ()
    # reload 期间产生的其他诊断。
    diagnostics: tuple[SkillDiagnostic, ...] = ()


@dataclass(frozen=True)
class SkillSessionState:
    """保存会话恢复所需的最少 Skill 状态。

    SessionManager 只持久化名字和顺序。恢复时 Runtime 会从当前 Catalog
    重新读取 SOP、白名单和工具，不沿用旧文件内容。
    """

    # 按原激活顺序排列的 inline Skill 名。
    active_skills: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillRestoreReport:
    """说明恢复会话时哪些 inline Skill 成功重新激活。"""

    # 按保存顺序成功激活的 Skill 名。
    restored: tuple[str, ...] = ()
    # 缺失、无效或已经改成 fork 的项目对应的用户可见说明。
    warnings: tuple[str, ...] = ()
