"""解析独立子 Agent 的 Markdown 文件和 YAML frontmatter。"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from mycode.models.agents import (
    AgentDefinition,
    AgentPermissionMode,
    AgentSource,
)
from mycode.models.worktrees import WorkspaceIsolationMode

_AGENT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9-]{0,63}$")
_TOOL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_FIELDS = frozenset(
    {
        "name",
        "description",
        "tools",
        "disallowedTools",
        "model",
        "maxModelCalls",
        "permissionMode",
        "background",
        "isolation",
    }
)


class AgentParseError(ValueError):
    """说明一份角色 Markdown 的路径和具体格式错误。

    Attributes:
        path: 实际解析失败的文件绝对路径。
        reason: 可以直接放入 `/agent reload` 输出的错误原因。
    """

    def __init__(self, path: Path, reason: str) -> None:
        """保存解析失败的文件和原因。

        Args:
            path: 出错的角色 Markdown 绝对路径。
            reason: 不含 traceback 的具体错误说明。

        Returns:
            不返回数据；异常文本会同时包含文件路径和具体原因。
        """

        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


class AgentParser:
    """把一份角色 Markdown 转换成可进入 AgentCatalog 的定义。

    Loader 调用该类解析每个候选文件。解析器只读取单个文件、校验字段并
    计算 revision，不负责来源覆盖、工具是否已注册或 Verification 开关。
    """

    def parse(
        self,
        entry_path: Path,
        source: AgentSource,
    ) -> AgentDefinition:
        """读取并校验一份角色 Markdown。

        Args:
            entry_path: 待解析的 Markdown 文件路径。
            source: 文件来自项目、用户还是内置资源。

        Returns:
            包含角色指令、工具限制和运行默认值的不可变定义。

        Raises:
            AgentParseError: 文件不可读、YAML 无效、出现未知字段，或任一
                已批准字段的类型和值不合法。
        """

        path = entry_path.resolve()
        if not path.is_file():
            raise AgentParseError(path, "Agent 入口不存在或不是文件")
        if not isinstance(source, AgentSource):
            raise AgentParseError(path, "Agent 来源无效")
        try:
            raw_text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise AgentParseError(path, f"无法按 UTF-8 读取：{exc}") from exc

        metadata, prompt_body = self._split_markdown(path, raw_text)
        if "maxTurns" in metadata:
            raise AgentParseError(
                path,
                "maxTurns 已废弃，请改用 maxModelCalls",
            )
        unknown = sorted(set(metadata) - _FIELDS)
        if unknown:
            raise AgentParseError(
                path,
                f"YAML 包含未知字段：{', '.join(unknown)}",
            )

        name = self._required_text(metadata, "name", path)
        if not _AGENT_NAME.fullmatch(name):
            raise AgentParseError(
                path,
                "name 必须以字母开头，且只能包含字母、数字和连字符",
            )
        description = self._required_text(metadata, "description", path)
        tools = self._tool_names(metadata, "tools", path, optional=True)
        disallowed_tools = self._tool_names(
            metadata,
            "disallowedTools",
            path,
            optional=False,
        )
        model = self._model(metadata, path)
        max_model_calls = self._max_model_calls(metadata, path)
        permission_mode = self._permission_mode(metadata, path)
        background = self._background(metadata, path)
        isolation = self._isolation(metadata, path)

        return AgentDefinition(
            name=name,
            description=description,
            tools=tools,
            disallowed_tools=disallowed_tools or frozenset(),
            model=model,
            max_model_calls=max_model_calls,
            permission_mode=permission_mode,
            default_background=background,
            source=source,
            entry_path=path,
            prompt_body=prompt_body,
            revision=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            isolation=isolation,
        )

    def _split_markdown(
        self,
        path: Path,
        raw_text: str,
    ) -> tuple[Mapping[str, Any], str]:
        """分开 YAML frontmatter 和角色系统指令正文。

        Args:
            path: 当前文件路径，用在错误信息中。
            raw_text: 从文件读取的完整 UTF-8 文本。

        Returns:
            YAML 字段映射，以及去掉首尾空白的 Markdown 正文。

        Raises:
            AgentParseError: frontmatter 边界、YAML 类型或正文不合法。
        """

        lines = raw_text.splitlines()
        if not lines or lines[0].strip() != "---":
            raise AgentParseError(path, "文件必须以 YAML frontmatter 开头")
        closing_index = next(
            (
                index
                for index, line in enumerate(lines[1:], start=1)
                if line.strip() == "---"
            ),
            None,
        )
        if closing_index is None:
            raise AgentParseError(path, "YAML frontmatter 缺少结束分隔线")
        try:
            loaded = yaml.safe_load("\n".join(lines[1:closing_index]))
        except yaml.YAMLError as exc:
            raise AgentParseError(path, f"YAML 无法解析：{exc}") from exc
        if not isinstance(loaded, Mapping):
            raise AgentParseError(path, "YAML frontmatter 必须是键值对象")
        if any(not isinstance(key, str) for key in loaded):
            raise AgentParseError(path, "YAML frontmatter 的字段名必须是字符串")
        body = "\n".join(lines[closing_index + 1 :]).strip()
        if not body:
            raise AgentParseError(path, "Markdown 系统提示正文不能为空")
        return loaded, body

    def _required_text(
        self,
        metadata: Mapping[str, Any],
        field_name: str,
        path: Path,
    ) -> str:
        """读取一个必填的非空字符串字段。

        Args:
            metadata: frontmatter 解析出的键值映射。
            field_name: 当前要读取的字段名。
            path: 当前文件路径，用在错误信息中。

        Returns:
            去掉首尾空白后的字段文本。

        Raises:
            AgentParseError: 字段缺失、不是字符串或只有空白。
        """

        value = metadata.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise AgentParseError(path, f"{field_name} 必须是非空字符串")
        return value.strip()

    def _tool_names(
        self,
        metadata: Mapping[str, Any],
        field_name: str,
        path: Path,
        *,
        optional: bool,
    ) -> frozenset[str] | None:
        """读取工具白名单或黑名单并保留“未填写”的含义。

        Args:
            metadata: frontmatter 解析出的键值映射。
            field_name: ``tools`` 或 ``disallowedTools``。
            path: 当前文件路径，用在错误信息中。
            optional: 字段缺失时是否返回 ``None``；黑名单传 ``False``，
                缺失时返回空集合。

        Returns:
            去重后的工具名集合；可选字段未填写时返回 ``None``。

        Raises:
            AgentParseError: 字段不是字符串数组，或数组含非法工具名。
        """

        if field_name not in metadata:
            return None if optional else frozenset()
        value = metadata[field_name]
        if not isinstance(value, Sequence) or isinstance(
            value,
            (str, bytes, bytearray),
        ):
            raise AgentParseError(path, f"{field_name} 必须是字符串数组")
        names: set[str] = set()
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise AgentParseError(
                    path,
                    f"{field_name} 中的每个工具名都必须是非空字符串",
                )
            name = item.strip()
            if not _TOOL_NAME.fullmatch(name):
                raise AgentParseError(
                    path,
                    f"{field_name} 包含无效工具名：{name!r}",
                )
            names.add(name)
        return frozenset(names)

    def _model(
        self,
        metadata: Mapping[str, Any],
        path: Path,
    ) -> str | None:
        """读取模型覆盖，并把未填写或 ``inherit`` 转为 ``None``。

        Args:
            metadata: frontmatter 解析出的键值映射。
            path: 当前文件路径，用在错误信息中。

        Returns:
            Provider 模型名；继承父模型时返回 ``None``。

        Raises:
            AgentParseError: model 存在但不是非空字符串。
        """

        value = metadata.get("model")
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise AgentParseError(path, "model 必须是非空字符串")
        normalized = value.strip()
        return None if normalized.casefold() == "inherit" else normalized

    def _max_model_calls(
        self,
        metadata: Mapping[str, Any],
        path: Path,
    ) -> int | None:
        """读取角色最大模型调用次数。

        Args:
            metadata: frontmatter 解析出的键值映射。
            path: 当前文件路径，用在错误信息中。

        Returns:
            正整数调用次数；字段未填写时返回 ``None``。

        Raises:
            AgentParseError: 字段不是正整数。
        """

        value = metadata.get("maxModelCalls")
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AgentParseError(path, "maxModelCalls 必须是正整数")
        return value

    def _permission_mode(
        self,
        metadata: Mapping[str, Any],
        path: Path,
    ) -> AgentPermissionMode:
        """读取角色权限模式，未填写时使用 ``inherit``。

        Args:
            metadata: frontmatter 解析出的键值映射。
            path: 当前文件路径，用在错误信息中。

        Returns:
            校验后的角色权限模式枚举。

        Raises:
            AgentParseError: 字段不是字符串或不属于四个允许值。
        """

        value = metadata.get("permissionMode", AgentPermissionMode.INHERIT.value)
        if not isinstance(value, str):
            raise AgentParseError(path, "permissionMode 必须是字符串")
        try:
            return AgentPermissionMode(value.strip().casefold())
        except ValueError as exc:
            choices = "、".join(mode.value for mode in AgentPermissionMode)
            raise AgentParseError(
                path,
                f"permissionMode 只能是：{choices}",
            ) from exc

    def _background(
        self,
        metadata: Mapping[str, Any],
        path: Path,
    ) -> bool:
        """读取角色默认后台开关。

        Args:
            metadata: frontmatter 解析出的键值映射。
            path: 当前文件路径，用在错误信息中。

        Returns:
            字段填写的布尔值；未填写时返回 ``False``。

        Raises:
            AgentParseError: background 存在但不是布尔值。
        """

        value = metadata.get("background", False)
        if not isinstance(value, bool):
            raise AgentParseError(path, "background 必须是布尔值")
        return value

    def _isolation(
        self,
        metadata: Mapping[str, Any],
        path: Path,
    ) -> WorkspaceIsolationMode:
        """读取定义式子 Agent 的工作区隔离模式。

        Args:
            metadata: frontmatter 解析出的键值映射。
            path: 当前角色 Markdown 的绝对路径，用在错误信息中。

        Returns:
            ``worktree`` 对应独立 Worktree，``shared`` 对应共享调用方目录；
            未填写时默认返回独立 Worktree。

        Raises:
            AgentParseError: ``isolation`` 不是字符串，或不是 ``worktree``/
                ``shared`` 中的一个。
        """

        value = metadata.get("isolation", WorkspaceIsolationMode.WORKTREE.value)
        if not isinstance(value, str):
            raise AgentParseError(path, "isolation 必须是 worktree 或 shared")
        try:
            return WorkspaceIsolationMode(value.strip())
        except ValueError as exc:
            raise AgentParseError(
                path,
                "isolation 必须是 worktree 或 shared",
            ) from exc
