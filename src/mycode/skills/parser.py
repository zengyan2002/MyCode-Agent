"""解析 Skill Markdown、YAML frontmatter 和目录型专属工具。"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence, Set
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from mycode.models.json_types import JsonObject
from mycode.models.tools import ToolAccess
from mycode.models.skills import (
    SkillContextMode,
    SkillDefinition,
    SkillMode,
    SkillSource,
    SkillToolSpec,
)

_SKILL_NAME = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_TOOL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_DEFAULT_TIMEOUT_SECONDS = 60.0
_MAX_TIMEOUT_SECONDS = 300.0
_DEFAULT_OUTPUT_BYTES = 1_048_576
_MAX_OUTPUT_BYTES = 10_485_760


class SkillParseError(ValueError):
    """说明某个 Skill 文件的内容不符合已批准的格式。

    Attributes:
        path: 实际出错的 Markdown 或 tool.json 路径。
        reason: 可以直接展示给用户的错误原因。
    """

    def __init__(self, path: Path, reason: str) -> None:
        """保存出错文件和原因。

        Args:
            path: 解析失败的实际文件路径。
            reason: 不包含 traceback 的具体错误说明。
        """

        super().__init__(f"{path}: {reason}")
        # /skill reload 用这个路径告诉用户应该修改哪个文件。
        self.path = path
        # Loader 用这个说明生成可继续展示的诊断。
        self.reason = reason


def replace_skill_arguments(prompt_body: str, arguments: str) -> str:
    """把 SOP 中每个 $ARGUMENTS 替换成用户传入的原始参数。

    Args:
        prompt_body: 已去掉 YAML frontmatter 的 Skill 正文。
        arguments: Skill 名之后的原始文本；没有参数时传空字符串。

    Returns:
        完成普通字符串替换后的 SOP。函数不会解释引号、变量或 Shell 语法。
    """

    return prompt_body.replace("$ARGUMENTS", arguments)


class SkillParser:
    """把一个 Skill 入口文件转换成可以执行的 SkillDefinition。

    CLI 使用静态命令名创建 Parser，Loader 随后把扫描到的每个候选交给
    它。解析器只读文件和校验格式，不注册命令、不启动脚本。
    """

    def __init__(
        self,
        *,
        reserved_names: Set[str] = frozenset(),
    ) -> None:
        """创建解析器并记录不能被 Skill 占用的命令名。

        Args:
            reserved_names: 现有静态命令名和别名。比较时忽略大小写。
        """

        # 规范化后保存，解析每个文件时不重复处理同一批命令名。
        self._reserved_names = frozenset(
            name.strip().casefold()
            for name in reserved_names
            if name.strip()
        )

    def parse(
        self,
        entry_path: Path,
        source: SkillSource,
    ) -> SkillDefinition:
        """读取并校验一个 Skill 入口及同目录可选的 tool.json。

        Args:
            entry_path: 单文件 Skill，或目录型 Skill 的 SKILL.md 路径。
            source: 该文件来自项目级、用户级还是内置级目录。

        Returns:
            包含 SOP、模式、白名单和专属工具的可执行定义。

        Raises:
            SkillParseError: 文件不可读、YAML/JSON 无效、字段缺失或路径越界。
        """

        path = entry_path.resolve()
        if not path.is_file():
            raise SkillParseError(path, "Skill 入口不存在或不是文件")
        try:
            raw_text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise SkillParseError(path, f"无法按 UTF-8 读取：{exc}") from exc

        metadata, prompt_body = self._split_markdown(path, raw_text)
        name = self._required_text(metadata, "name", path)
        if not _SKILL_NAME.fullmatch(name):
            raise SkillParseError(
                path,
                "name 必须以小写字母开头，且只能包含小写字母、数字和连字符",
            )
        if name.casefold() in self._reserved_names:
            raise SkillParseError(path, f"name 与内置命令冲突：{name}")

        description = self._required_text(metadata, "description", path)
        allowed_tools = self._allowed_tools(metadata, path)
        mode = self._enum_value(
            metadata,
            "mode",
            SkillMode,
            SkillMode.INLINE,
            path,
        )
        context = self._enum_value(
            metadata,
            "context",
            SkillContextMode,
            SkillContextMode.RECENT,
            path,
        )
        model = self._optional_text(metadata, "model", path)

        directory_skill = path.name == "SKILL.md"
        root_path = path.parent.resolve() if directory_skill else None
        tool_path = path.parent / "tool.json"
        tools: tuple[SkillToolSpec, ...] = ()
        tool_bytes = b""
        if directory_skill and tool_path.exists():
            tools, tool_bytes = self._parse_tool_file(
                tool_path.resolve(),
                root_path,
            )

        digest = hashlib.sha256()
        digest.update(raw_text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tool_bytes)
        return SkillDefinition(
            name=name,
            description=description,
            allowed_tools=allowed_tools,
            mode=mode,
            context=context,
            model=model,
            source=source,
            entry_path=path,
            root_path=root_path,
            prompt_body=prompt_body,
            tools=tools,
            revision=digest.hexdigest(),
        )

    def _split_markdown(
        self,
        path: Path,
        raw_text: str,
    ) -> tuple[Mapping[str, Any], str]:
        """分开 YAML frontmatter 和 Markdown SOP。

        Args:
            path: 当前入口路径，用在错误信息中。
            raw_text: 从入口文件读取的完整 UTF-8 文本。

        Returns:
            YAML 字段映射和去掉首尾空白的 SOP 正文。

        Raises:
            SkillParseError: frontmatter 边界、YAML 类型或正文不合法。
        """

        lines = raw_text.splitlines()
        if not lines or lines[0].strip() != "---":
            raise SkillParseError(path, "文件必须以 YAML frontmatter 开头")
        closing_index = next(
            (
                index
                for index, line in enumerate(lines[1:], start=1)
                if line.strip() == "---"
            ),
            None,
        )
        if closing_index is None:
            raise SkillParseError(path, "YAML frontmatter 缺少结束分隔线")
        yaml_text = "\n".join(lines[1:closing_index])
        try:
            loaded = yaml.safe_load(yaml_text)
        except yaml.YAMLError as exc:
            raise SkillParseError(path, f"YAML 无法解析：{exc}") from exc
        if not isinstance(loaded, Mapping):
            raise SkillParseError(path, "YAML frontmatter 必须是键值对象")
        body = "\n".join(lines[closing_index + 1 :]).strip()
        if not body:
            raise SkillParseError(path, "Markdown SOP 正文不能为空")
        return loaded, body

    def _required_text(
        self,
        metadata: Mapping[str, Any],
        field_name: str,
        path: Path,
    ) -> str:
        """读取一个必填的非空字符串字段。

        Args:
            metadata: frontmatter 解析出的键值。
            field_name: 当前要读取的字段名。
            path: 入口路径，用在错误信息中。

        Returns:
            去掉首尾空白后的字段文本。

        Raises:
            SkillParseError: 字段缺失、不是字符串或只有空白。
        """

        value = metadata.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise SkillParseError(path, f"{field_name} 必须是非空字符串")
        return value.strip()

    def _optional_text(
        self,
        metadata: Mapping[str, Any],
        field_name: str,
        path: Path,
    ) -> str | None:
        """读取一个可选字符串字段。

        Args:
            metadata: frontmatter 解析出的键值。
            field_name: 当前要读取的字段名。
            path: 入口路径，用在错误信息中。

        Returns:
            字段未填写时返回 None；填写时返回去掉首尾空白的文本。

        Raises:
            SkillParseError: 字段存在但不是非空字符串。
        """

        value = metadata.get(field_name)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise SkillParseError(path, f"{field_name} 必须是非空字符串")
        return value.strip()

    def _allowed_tools(
        self,
        metadata: Mapping[str, Any],
        path: Path,
    ) -> frozenset[str] | None:
        """保留 allowedTools 的未填写、空列表和精确列表三种状态。

        Args:
            metadata: frontmatter 解析出的键值。
            path: 入口路径，用在错误信息中。

        Returns:
            未填写时为 None；填写时为去重后的不可变集合。

        Raises:
            SkillParseError: 字段不是数组，或数组含非法工具名。
        """

        if "allowedTools" not in metadata:
            return None
        value = metadata["allowedTools"]
        if not isinstance(value, Sequence) or isinstance(
            value,
            (str, bytes, bytearray),
        ):
            raise SkillParseError(path, "allowedTools 必须是字符串数组")
        names: set[str] = set()
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise SkillParseError(
                    path,
                    "allowedTools 中的每个工具名都必须是非空字符串",
                )
            normalized = item.strip()
            if not _TOOL_NAME.fullmatch(normalized):
                raise SkillParseError(
                    path,
                    f"allowedTools 包含无效工具名：{normalized!r}",
                )
            names.add(normalized)
        return frozenset(names)

    def _enum_value(
        self,
        metadata: Mapping[str, Any],
        field_name: str,
        enum_type: type[SkillMode] | type[SkillContextMode],
        default: SkillMode | SkillContextMode,
        path: Path,
    ) -> SkillMode | SkillContextMode:
        """读取 mode 或 context，并转换成对应枚举。

        Args:
            metadata: frontmatter 解析出的键值。
            field_name: mode 或 context。
            enum_type: 负责校验该字段的枚举类型。
            default: 字段未填写时使用的值。
            path: 入口路径，用在错误信息中。

        Returns:
            校验后的枚举值。

        Raises:
            SkillParseError: 字段不是字符串或取值不在枚举中。
        """

        raw_value = metadata.get(field_name)
        if raw_value is None:
            return default
        if not isinstance(raw_value, str):
            raise SkillParseError(path, f"{field_name} 必须是字符串")
        try:
            return enum_type(raw_value.strip().casefold())
        except ValueError as exc:
            choices = "、".join(item.value for item in enum_type)
            raise SkillParseError(
                path,
                f"{field_name} 只能是：{choices}",
            ) from exc

    def _parse_tool_file(
        self,
        tool_path: Path,
        skill_root: Path,
    ) -> tuple[tuple[SkillToolSpec, ...], bytes]:
        """解析目录型 Skill 的 tool.json。

        Args:
            tool_path: tool.json 的真实路径。
            skill_root: 当前目录型 Skill 的真实根目录。

        Returns:
            校验后的专属工具列表，以及计算 Skill revision 使用的原始字节。

        Raises:
            SkillParseError: JSON、字段、Schema、限制或脚本路径不合法。
        """

        try:
            raw_bytes = tool_path.read_bytes()
            loaded = json.loads(raw_bytes.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SkillParseError(
                tool_path,
                f"tool.json 无法解析：{exc}",
            ) from exc
        if not isinstance(loaded, Mapping):
            raise SkillParseError(tool_path, "tool.json 顶层必须是对象")
        raw_tools = loaded.get("tools")
        if (
            not isinstance(raw_tools, list)
            or not raw_tools
        ):
            raise SkillParseError(
                tool_path,
                "tool.json 的 tools 必须是非空数组",
            )

        specs: list[SkillToolSpec] = []
        seen_names: set[str] = set()
        for index, raw_tool in enumerate(raw_tools):
            if not isinstance(raw_tool, Mapping):
                raise SkillParseError(
                    tool_path,
                    f"tools[{index}] 必须是对象",
                )
            spec = self._parse_tool(tool_path, skill_root, index, raw_tool)
            if spec.name in seen_names:
                raise SkillParseError(
                    tool_path,
                    f"tool.json 内工具名称重复：{spec.name}",
                )
            seen_names.add(spec.name)
            specs.append(spec)
        return tuple(specs), raw_bytes

    def _parse_tool(
        self,
        tool_path: Path,
        skill_root: Path,
        index: int,
        raw_tool: Mapping[str, Any],
    ) -> SkillToolSpec:
        """校验 tool.json 中的一项工具定义。

        Args:
            tool_path: tool.json 路径，用在错误信息中。
            skill_root: 脚本入口必须留在其中的真实目录。
            index: 当前工具在 tools 数组中的位置。
            raw_tool: JSON 解析出的单项对象。

        Returns:
            可以注册和执行的 SkillToolSpec。

        Raises:
            SkillParseError: 必填字段、Schema、命令或限制不合法。
        """

        label = f"tools[{index}]"
        name = self._json_text(raw_tool, "name", tool_path, label)
        if not _TOOL_NAME.fullmatch(name):
            raise SkillParseError(
                tool_path,
                f"{label}.name 不是合法工具名：{name!r}",
            )
        description = self._json_text(
            raw_tool,
            "description",
            tool_path,
            label,
        )
        raw_access = self._json_text(
            raw_tool,
            "access",
            tool_path,
            label,
        )
        try:
            access = ToolAccess(raw_access)
        except ValueError as exc:
            raise SkillParseError(
                tool_path,
                f"{label}.access 只能是 read 或 write",
            ) from exc

        raw_schema = raw_tool.get("inputSchema")
        if not isinstance(raw_schema, dict):
            raise SkillParseError(
                tool_path,
                f"{label}.inputSchema 必须是 JSON 对象",
            )
        try:
            Draft202012Validator.check_schema(raw_schema)
        except SchemaError as exc:
            raise SkillParseError(
                tool_path,
                f"{label}.inputSchema 不是合法 JSON Schema",
            ) from exc

        raw_command = raw_tool.get("command")
        if (
            not isinstance(raw_command, list)
            or len(raw_command) < 2
            or any(
                not isinstance(part, str) or not part
                for part in raw_command
            )
        ):
            raise SkillParseError(
                tool_path,
                f"{label}.command 必须是至少含运行时和脚本入口的字符串数组",
            )
        entry_part = Path(raw_command[1])
        if entry_part.is_absolute() or ".." in entry_part.parts:
            raise SkillParseError(
                tool_path,
                f"{label}.command 的脚本入口必须位于 Skill 目录内",
            )
        entry_path = (skill_root / entry_part).resolve()
        if not self._is_within(entry_path, skill_root) or not entry_path.is_file():
            raise SkillParseError(
                tool_path,
                f"{label}.command 的脚本入口不存在或越过 Skill 目录",
            )

        timeout_seconds = self._number_limit(
            raw_tool,
            "timeoutSeconds",
            _DEFAULT_TIMEOUT_SECONDS,
            _MAX_TIMEOUT_SECONDS,
            tool_path,
            label,
        )
        max_output = self._integer_limit(
            raw_tool,
            "maxOutputBytes",
            _DEFAULT_OUTPUT_BYTES,
            _MAX_OUTPUT_BYTES,
            tool_path,
            label,
        )
        return SkillToolSpec(
            name=name,
            description=description,
            input_schema=raw_schema,
            access=access,
            command=tuple(raw_command),
            entry_path=entry_path,
            skill_root=skill_root,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output,
        )

    def _json_text(
        self,
        raw_tool: Mapping[str, Any],
        field_name: str,
        path: Path,
        label: str,
    ) -> str:
        """读取 tool.json 中一项必填的非空字符串。

        Args:
            raw_tool: 当前工具的 JSON 对象。
            field_name: 要读取的字段名。
            path: tool.json 路径。
            label: 当前工具在数组中的位置说明。

        Returns:
            去掉首尾空白后的字段值。

        Raises:
            SkillParseError: 字段缺失、不是字符串或只有空白。
        """

        value = raw_tool.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise SkillParseError(
                path,
                f"{label}.{field_name} 必须是非空字符串",
            )
        return value.strip()

    def _number_limit(
        self,
        raw_tool: Mapping[str, Any],
        field_name: str,
        default: float,
        maximum: float,
        path: Path,
        label: str,
    ) -> float:
        """读取一个有正数上限的工具配置。

        Args:
            raw_tool: 当前工具的 JSON 对象。
            field_name: 要读取的数字字段。
            default: 字段未填写时使用的值。
            maximum: 允许的最大值。
            path: tool.json 路径。
            label: 当前工具在数组中的位置说明。

        Returns:
            校验后的浮点数。

        Raises:
            SkillParseError: 值不是正数或超过上限。
        """

        value = raw_tool.get(field_name, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
            or value > maximum
        ):
            raise SkillParseError(
                path,
                f"{label}.{field_name} 必须大于 0 且不超过 {maximum:g}",
            )
        return float(value)

    def _integer_limit(
        self,
        raw_tool: Mapping[str, Any],
        field_name: str,
        default: int,
        maximum: int,
        path: Path,
        label: str,
    ) -> int:
        """读取一个有正整数上限的工具配置。

        Args:
            raw_tool: 当前工具的 JSON 对象。
            field_name: 要读取的整数字段。
            default: 字段未填写时使用的值。
            maximum: 允许的最大值。
            path: tool.json 路径。
            label: 当前工具在数组中的位置说明。

        Returns:
            校验后的整数。

        Raises:
            SkillParseError: 值不是正整数或超过上限。
        """

        value = raw_tool.get(field_name, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value > maximum
        ):
            raise SkillParseError(
                path,
                f"{label}.{field_name} 必须是大于 0 且不超过 {maximum} 的整数",
            )
        return value

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        """判断解析后的真实路径是否仍在指定根目录内。

        Args:
            path: 已 resolve 的目标路径。
            root: 已 resolve 的 Skill 根目录。

        Returns:
            目标等于根目录或位于根目录下时返回 True。
        """

        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True
