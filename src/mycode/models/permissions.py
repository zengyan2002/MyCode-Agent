"""权限配置、决策和人工确认共用的数据模型。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol


class PermissionMode(str, Enum):
    """
    当前会话的整体权限模式
    """
    # 明确允许的才能执行，未匹配规则的一律拒绝。  /permissions strict
    STRICT = "strict"
    # 默认模式。未匹配规则时，进入人工确认流程   /permissions default
    DEFAULT = "default"
    # 放行模式。工具调用没有匹配规则时自动允许，不询问用户  /permissions allow
    ALLOW = "allow"


class PermissionEffect(str, Enum):
    """
    表示一条具体规则的结果
    """
    # 该条规则放行
    ALLOW = "allow"
    # 该条规则禁止
    DENY = "deny"


class PermissionScope(str, Enum):
    """
    表示权限规则保存在哪一层，影响范围多大
    覆盖范围：USER > PROJECT > LOCAL
    """
    # 当前用户的所有项目生效，配置位置~/.mycode/permissions.yaml
    USER = "user"
    # 当前项目生效，配置位置 <项目>/.mycode/permissions.yaml
    PROJECT = "project"
    # 当前机器上的当前项目生效 <项目>/.mycode/permissions.local.yaml
    LOCAL = "local"
    # 仅保存在内存中，当前会话生效
    SESSION = "session"


class PermissionTool(str, Enum):
    """
    表示权限系统能够识别的工具权限类别
    """
    #shell命令工具  调用的工具execute_command
    SHELL = "Shell"
    #读文件工具  调用的工具read_file
    READ_FILE = "ReadFile"
    #写或者编辑文件工具  调用的工具write_file  edit_file
    WRITE_FILE = "WriteFile"
    #查找文件  调用的工具find_files
    FIND_FILES = "FindFiles"
    #查找代码  调用的工具search_code
    SEARCH_CODE = "SearchCode"
    #外部 MCP Server 提供的命名空间工具
    MCP = "MCP"
    # 目录型 Skill 通过 tool.json 注册的专属工具
    SKILL = "Skill"


class ApprovalChoice(str, Enum):
    """
    需要人工确认时，用户做出的选择
    """
    # 用户拒绝
    DENY = "deny"
    # 只允许眼前这一次调用
    ALLOW_ONCE = "allow_once"
    # 允许当前这个具体调用在本次会话中重复执行
    ALLOW_SESSION = "allow_session"
    # 允许当前调用，并将精确规则写入当前项目：<项目>/.mycode/permissions.local.yaml
    ALLOW_PERMANENT = "allow_permanent"


class PermissionOutcome(str, Enum):
    """
    权限策略自动计算出的结果。
    """
    #直接执行
    ALLOW = "allow"
    #直接拒绝
    DENY = "deny"
    #弹出权限确认呢，需要用户确认
    ASK = "ask"


@dataclass(frozen=True)
class PermissionOperation:
    """
    一项等待权限检查的工具操作。

    记录工具类型以及用于规则匹配、界面展示和路径沙箱检查的信息。
    """
    # 模型调用属于哪一种权限工具
    tool: PermissionTool
    # 权限系统真正拿来与规则比较的值
    match_value: str
    # 权限确认界面中展示给用户看的操作文本
    display_value: str
    # 文件工具交给路径沙箱检查的路径；非路径工具（例如Shell）为 None
    path_value: str | None = None

    def __post_init__(self) -> None:
        if not self.match_value:
            raise ValueError("待授权操作的匹配值不能为空")
        if not self.display_value:
            raise ValueError("待授权操作的显示值不能为空")
        if self.path_value is not None and not self.path_value:
            raise ValueError("待授权操作的路径值不能为空")


@dataclass(frozen=True)
class PermissionRule:
    """
    PermissionRule 是一条经过解析和编译的权限规则，记录“对什么工具、匹配什么内容、允许还是拒绝、来自哪一层”，并负责判断某个 PermissionOperation 是否符合该规则。
    """
    # 当前规则针对哪一种工具
    tool: PermissionTool
    # 规则中的匹配模式
    pattern: str
    # 匹配成功后的结果
    effect: PermissionEffect
    # 规则来自哪个权限层
    scope: PermissionScope
    # 规则里是否包含通配符
    is_glob: bool
    # 通配规则的“具体程度”，用于多个规则同时匹配时选出更精确的一个  (
    #     普通字符数量,
    #     通配符数量的负数,
    #     模式总长度,
    # )
    specificity: tuple[int, int, int]
    # 规则来自哪里，主要用于错误信息、日志和权限说明。
    source: str
    # 由 pattern 预编译得到的正则表达式
    matcher: re.Pattern[str] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.pattern:
            raise ValueError("权限规则模式不能为空")
        if not self.source:
            raise ValueError("权限规则来源不能为空")
        if len(self.specificity) != 3:
            raise ValueError("权限规则具体程度必须包含三个维度")

    def matches(self, operation: PermissionOperation) -> bool:
        # 匹配当前规则和模型的工具请求转换来的 PermissionOperation 对象
        return (
            self.tool is operation.tool
            and self.matcher.fullmatch(operation.match_value) is not None
        )


@dataclass(frozen=True)
class PermissionLayer:
    """
    某一级别的权限配置。

    把生效范围相同的权限模式和规则放在一起，
    规则决议器按照会话、本地、项目、用户的顺序逐层判断。
    """
    # 表示是什么权限层
    scope: PermissionScope
    # 当前会话的整体权限模式
    mode: PermissionMode | None
    # 这一层的所有权限规则
    rules: tuple[PermissionRule, ...]
    # 这一层的配置文件
    source_path: Path | None

    def __post_init__(self) -> None:
        if self.scope is PermissionScope.SESSION and self.source_path is not None:
            raise ValueError("会话级权限层不能关联配置文件")
        if any(rule.scope is not self.scope for rule in self.rules):
            raise ValueError("权限层包含了其他 scope 的规则")


@dataclass(frozen=True)
class LoadedPermissionSettings:
    """
    权限配置加载完成后的不可变快照。

    汇总用户级、项目级和本地级权限层，并保存按照
    LOCAL → PROJECT → USER 优先级解析出的会话初始模式。
    local_path 是“永久允许”规则的写入位置。

    会话级规则在程序运行期间动态产生，由
    PermissionController 单独维护，因此不保存在该对象中。
    """
    # 表示会话启动时使用的整体权限模式
    initial_mode: PermissionMode
    # 保存每一层的权限的配置  没有session是因为会话规则是在程序运行期间动态产生的：
    user: PermissionLayer
    project: PermissionLayer
    local: PermissionLayer
    # 表示用户永久授权后需要写入的文件  用户选择“永久允许”时，规则会写到这个路径
    local_path: Path

    def __post_init__(self) -> None:
        if self.user.scope is not PermissionScope.USER:
            raise ValueError("user 权限层 scope 错误")
        if self.project.scope is not PermissionScope.PROJECT:
            raise ValueError("project 权限层 scope 错误")
        if self.local.scope is not PermissionScope.LOCAL:
            raise ValueError("local 权限层 scope 错误")
        if not self.local_path.is_absolute():
            raise ValueError("本地权限配置路径必须是绝对路径")


@dataclass(frozen=True)
class PermissionDecision:
    """
    表示权限策略对某一次 PermissionOperation 的判断结果。
    """
    # 权限策略计算出的成果
    outcome: PermissionOutcome
    # 适合程序判断、日志统计的稳定原因代码
    # 本项目代码会产生rule_allow、rule_deny、strict_mode、allow_mode、default_mode
    reason: str
    # 面向用户或日志的说明文本
    message: str
    # 如果本次结果来自某条具体规则，这里保存命中的规则
    matched_rule: PermissionRule | None = None


    def __post_init__(self) -> None:
        if not self.reason or not self.message:
            raise ValueError("权限决策必须包含原因和消息")
        if (
            self.matched_rule is not None
            and self.outcome is PermissionOutcome.ASK
        ):
            raise ValueError("需要人工确认的决策不能包含命中规则")


@dataclass(frozen=True)
class PermissionRequest:
    """
    权限系统发送给UI的人工请求
    只会在权限策略无法自动决定，需要询问用户时创建
    """
    # 询问的操作
    operation: PermissionOperation
    # 表示系统根据当前操作生成的建议权限规则
    suggested_rule: str

    def __post_init__(self) -> None:
        if not self.suggested_rule:
            raise ValueError("人工确认请求必须包含建议规则")


class PermissionApprover(Protocol):
    """由当前终端界面向用户展示权限请求并返回选择。"""
    async def request_permission(
        self,
        request: PermissionRequest,
    ) -> ApprovalChoice: ...


class PermissionStore(Protocol):
    """
    永久权限规则存储接口  LocalPermissionStore是其实现类
    """
    def allow_permanently(
        self,
        operation: PermissionOperation,
    ) -> PermissionLayer: ...
