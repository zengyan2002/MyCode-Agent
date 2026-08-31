"""工具的集中注册与 JSON Schema 参数校验。"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from mycode.models.json_types import JsonValue
from mycode.models.tools import (
    ToolAccess,
    ToolDefinition,
    ToolExecutionPolicy,
    ToolSource,
    ToolView,
)
from mycode.tools.base import Tool

# 工具名必须以英文字母开头，只能包含字母、数字、下划线和短横线，总长度为 1～64 个字符
_TOOL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class SkillToolRegistration:
    """把一个专属工具实现和所属 Skill 的运行信息交给注册表。"""

    # 已经实现 Tool 协议的专属工具对象。
    tool: Tool
    # 提供该工具的 Skill 名。
    skill_name: str
    # project、user 或 builtin，用于首次信任判断。
    skill_origin: str
    # Skill 入口路径，用于首次信任提示。
    source_path: Path
    # 子进程实际使用的命令数组。
    command: tuple[str, ...]
    # ToolExecutor 使用的独立超时。
    timeout_seconds: float
    # SkillSubprocessTool 使用的 stdout 上限。
    max_output_bytes: int


class ToolRegistry:
    """保存全部工具，并按当前对话状态生成模型可见工具列表。"""

    def __init__(self) -> None:
        # Python dict 保留插入顺序；definitions 会按该顺序暴露给 Provider，
        # 因而注册顺序也是对模型稳定可见的协议组成部分。
        self._tools: dict[str, Tool] = {}
        # Schema 在注册时编译成 validator，运行时不重复检查 Schema 本身。
        self._validators: dict[str, Draft202012Validator] = {}
        # 保存每个已注册工具的可信来源，供权限适配层选择规则类别。
        self._sources: dict[str, ToolSource] = {}
        # 保存执行器、白名单和首次信任需要的运行信息。
        self._policies: dict[str, ToolExecutionPolicy] = {}

    def register(
        self,
        tool: Tool,
        *,
        source: ToolSource = ToolSource.BUILTIN,
        timeout_seconds: float | None = None,
    ) -> None:
        """校验并注册工具，同时保存其可信来源和可选独立超时。

        Args:
            tool: 等待注册的工具实现。
            source: 工具来源；未显式指定时视为 MyCode 内置工具。
            timeout_seconds: 该工具一次执行允许使用的最长秒数；``None``
                表示继续使用 ToolExecutor 的全局超时。

        Returns:
            不返回数据；注册成功后工具定义和执行策略可以按名称查询。

        Raises:
            ValueError: 工具定义、名称或独立超时无效，或者名称已经注册。
        """
        definition, validator = self._validate_tool(tool)
        if definition.name in self._tools:
            raise ValueError(f"工具名称已注册：{definition.name}")
        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError(
                f"工具 {definition.name!r} 的独立超时必须是正数"
            )

        self._tools[definition.name] = tool
        self._validators[definition.name] = validator
        self._sources[definition.name] = source
        self._policies[definition.name] = ToolExecutionPolicy(
            source=source,
            timeout_seconds=(
                None
                if timeout_seconds is None
                else float(timeout_seconds)
            ),
        )

    def _validate_tool(
        self,
        tool: Tool,
    ) -> tuple[ToolDefinition, Draft202012Validator]:
        """校验一个工具自身的名称、说明、分类和 JSON Schema。

        Args:
            tool: 等待静态登记或 Skill 批量替换的工具实现。

        Returns:
            工具定义和已经编译好的参数校验器。

        Raises:
            TypeError: 对象没有提供 Tool 协议要求的定义。
            ValueError: 名称、描述、读写类别或 Schema 不合法。
        """

        try:
            definition = tool.definition
        except AttributeError as exc:
            raise TypeError("只能注册实现 Tool 协议的对象") from exc
        if not _TOOL_NAME.fullmatch(definition.name):
            raise ValueError(f"工具名称无效：{definition.name!r}")
        if not definition.description.strip():
            raise ValueError(f"工具 {definition.name!r} 必须提供描述")
        if not isinstance(definition.access, ToolAccess):
            raise ValueError(f"工具 {definition.name!r} 必须声明读写分类")
        try:
            Draft202012Validator.check_schema(definition.input_schema)
        except SchemaError as exc:
            raise ValueError(
                f"工具 {definition.name!r} 的输入 Schema 无效"
            ) from exc

        return definition, Draft202012Validator(definition.input_schema)

    def replace_skill_tools(
        self,
        registrations: Sequence[SkillToolRegistration],
    ) -> None:
        """整批替换 Skill 专属工具，失败时保留当前专属工具层。

        Args:
            registrations: 当前有效目录型 Skill 的全部专属工具及运行信息。

        Returns:
            None。全部工具校验且无全局重名后，新工具层立即生效。

        Raises:
            ValueError: 工具名与非 Skill 工具或本批其他工具冲突，或运行
                配置不完整。
        """

        next_tools: dict[str, Tool] = {}
        next_validators: dict[str, Draft202012Validator] = {}
        next_policies: dict[str, ToolExecutionPolicy] = {}
        non_skill_names = {
            name
            for name, source in self._sources.items()
            if source is not ToolSource.SKILL
        }
        for registration in registrations:
            definition, validator = self._validate_tool(registration.tool)
            name = definition.name
            if name in non_skill_names:
                raise ValueError(f"Skill 工具名称与现有工具冲突：{name}")
            if name in next_tools:
                raise ValueError(f"多个 Skill 工具使用了同一名称：{name}")
            if not registration.skill_name.strip():
                raise ValueError(f"Skill 工具 {name!r} 缺少所属 Skill")
            if registration.timeout_seconds <= 0:
                raise ValueError(f"Skill 工具 {name!r} 超时必须为正数")
            if registration.max_output_bytes <= 0:
                raise ValueError(f"Skill 工具 {name!r} 输出上限必须为正数")
            next_tools[name] = registration.tool
            next_validators[name] = validator
            next_policies[name] = ToolExecutionPolicy(
                source=ToolSource.SKILL,
                skill_name=registration.skill_name.casefold(),
                skill_origin=registration.skill_origin,
                source_path=registration.source_path,
                command=registration.command,
                timeout_seconds=registration.timeout_seconds,
                max_output_bytes=registration.max_output_bytes,
            )

        retained_names = [
            name
            for name, source in self._sources.items()
            if source is not ToolSource.SKILL
        ]
        self._tools = {
            **{name: self._tools[name] for name in retained_names},
            **next_tools,
        }
        self._validators = {
            **{name: self._validators[name] for name in retained_names},
            **next_validators,
        }
        self._sources = {
            **{name: self._sources[name] for name in retained_names},
            **{name: ToolSource.SKILL for name in next_tools},
        }
        self._policies = {
            **{name: self._policies[name] for name in retained_names},
            **next_policies,
        }

    def get(self, name: str) -> Tool | None:
        """根据已注册工具的名字得到该工具对象

        Args:
            name: 需要查找的工具注册名称

        Returns:
            对应的工具对象；名称未注册时返回 None
        """
        return self._tools.get(name)

    @property
    def registered_names(self) -> frozenset[str]:
        """返回基础、系统、MCP 和 Skill 层当前占用的全部工具名。

        Returns:
            启动校验和 SkillService 可用于检查白名单引用的只读名字集合。
        """

        return frozenset(self._tools)

    def source_for(self, name: str) -> ToolSource | None:
        """查询已注册工具的可信来源。

        Args:
            name: 工具注册名称。

        Returns:
            已保存的工具来源；名称未知时返回 ``None``。
        """
        return self._sources.get(name)

    def execution_policy(
        self,
        name: str,
    ) -> ToolExecutionPolicy | None:
        """查询工具执行时需要的来源和限制。

        Args:
            name: 已注册工具名。

        Returns:
            对应执行策略；名称未知时返回 None。
        """

        return self._policies.get(name)

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        """返回当前对话中可以发送给模型的工具定义。"""
        definitions, _ = self.definitions_for(ToolView())
        return definitions

    def definitions_for(
        self,
        view: ToolView,
    ) -> tuple[tuple[ToolDefinition, ...], ToolView]:
        """根据活动 Skill、白名单和 MCP 状态解析本轮工具。

        Args:
            view: SkillRuntime 提供的活动 Skill 名和业务白名单。

        Returns:
            Provider 可见定义，以及写入同一批工具名快照的新 ToolView。
        """

        visible: list[ToolDefinition] = []
        for name, tool in self._tools.items():
            source = self._sources[name]
            if source is ToolSource.MCP and name not in view.active_mcp_names:
                continue
            if source is ToolSource.SKILL:
                policy = self._policies[name]
                if (
                    policy.skill_name is None
                    or policy.skill_name not in view.active_skill_names
                ):
                    continue
            if (
                view.business_allowlist is not None
                and source is not ToolSource.SYSTEM
                and name not in view.business_allowlist
            ):
                continue
            if (
                view.final_allowlist is not None
                and name not in view.final_allowlist
            ):
                continue
            if name in view.denied_tool_names:
                continue
            visible.append(tool.definition)
        names = frozenset(definition.name for definition in visible)
        return (
            tuple(visible),
            view.resolved(names),
        )

    def deferred_mcp_names_for(
        self,
        active_mcp_names: frozenset[str] | set[str],
    ) -> tuple[str, ...]:
        """列出某个 Agent 尚未激活的 MCP 工具名。

        Args:
            active_mcp_names: 该 Agent 自己的 ToolActivationState 当前集合。

        Returns:
            按工具注册顺序排列的未激活 MCP 名称。
        """

        return tuple(
            name
            for name in self._tools
            if self._sources[name] is ToolSource.MCP
            and name not in active_mcp_names
        )

    def search_mcp(
        self,
        query: str,
        active_mcp_names: frozenset[str] | set[str],
    ) -> tuple[ToolDefinition, ...]:
        """搜索某个 Agent 尚未激活的 MCP 工具。

        搜索时忽略大小写，匹配顺序依次为工具名完全相同、工具名包含关键
        词、工具说明包含关键词；同等级按注册顺序排列。该方法不修改注册
        表或调用方集合，ToolSearchTool 负责把返回名字写入当前 Agent 状态。

        Args:
            query: 用来搜索工具名和工具说明的关键词。
            active_mcp_names: 当前 Agent 已经激活、需要排除的 MCP 工具名。

        Returns:
            最多五个匹配的工具定义。关键词为空或没有匹配时返回空元组。
        """

        normalized = query.strip().casefold()
        if not normalized:
            return ()

        # 保存匹配到的工具，每项依次为：匹配等级、注册顺序、工具名和工具定义
        matches: list[tuple[int, int, str, ToolDefinition]] = []
        for index, (name, tool) in enumerate(self._tools.items()):
            if (
                self._sources[name] is not ToolSource.MCP
                or name in active_mcp_names
            ):
                # 只搜索尚未提供给模型的 MCP 工具，内置工具、系统工具和已经启用的 MCP 工具都不再参与搜索。
                continue
            definition = tool.definition
            normalized_name = name.casefold()
            normalized_description = definition.description.casefold()
            # 工具名完全相同，优先级最高
            if normalized_name == normalized:
                rank = 0
            # 工具名包含关键词，优先级第二
            elif normalized in normalized_name:
                rank = 1
            # 工具说明包含关键词，优先级第三
            elif normalized in normalized_description:
                rank = 2
            # 剩余情况就是完全不匹配
            else:
                continue
            matches.append((rank, index, name, definition))

        # 按照匹配程度和注册顺序对匹配上的工具进行排序
        selected = sorted(matches, key=lambda match: (match[0], match[1]))[:5]
        return tuple(item[3] for item in selected)

    def validate_arguments(
        self,
        name: str,
        arguments: Mapping[str, JsonValue],
    ) -> str | None:
        # 发给模型的 Schema 只是提示，模型返回的 JSON 仍是不可信输入；
        # 必须在任何文件或命令副作用发生前重新验证。
        validator = self._validators.get(name)
        if validator is None:
            return f"未知工具：{name}"

        # jsonschema 可能为同一个错误值报告多个连带问题；排序后只返回
        # 稳定的首个错误，方便模型根据一致反馈修正参数。
        errors = sorted(
            validator.iter_errors(dict(arguments)),
            key=lambda error: (
                tuple(str(part) for part in error.absolute_path),
                error.message,
            ),
        )
        if not errors:
            return None
        error = errors[0]
        path = ".".join(str(part) for part in error.absolute_path)
        location = f"（位置：{path}）" if path else ""
        return f"工具参数无效{location}：{error.message}"
