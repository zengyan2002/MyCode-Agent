"""在启动阶段校验 Hook 模板，并在事件发生时替换变量。"""

from __future__ import annotations

import re

from mycode.errors import ConfigError
from mycode.models.hooks import HookContext, HookTemplate


_VARIABLE = re.compile(
    r"\$(?:TOOL_ARGS\.[A-Za-z_][A-Za-z0-9_]*|"
    r"EVENT|TOOL_NAME|FILE_PATH|MESSAGE|ERROR)"
)
_POSSIBLE_VARIABLE = re.compile(r"\$[A-Za-z_][A-Za-z0-9_.]*")


def parse_template(value: object, *, source: str) -> HookTemplate:
    """校验一段动作文本并记录其中的上下文变量。

    Args:
        value: YAML 动作字段中读取到的值，必须是字符串。
        source: 当前字段的配置位置，出错时原样写入提示。

    Returns:
        已记录全部合法变量的模板，动作执行时可直接展开。

    Raises:
        ConfigError: 值不是字符串，或文本中包含未知变量。
    """

    if not isinstance(value, str):
        raise ConfigError(f"{source} 必须是字符串")
    variables = tuple(match.group(0) for match in _VARIABLE.finditer(value))
    legal_spans = {match.span() for match in _VARIABLE.finditer(value)}
    for candidate in _POSSIBLE_VARIABLE.finditer(value):
        if candidate.span() not in legal_spans:
            raise ConfigError(f"{source} 包含未知模板变量：{candidate.group(0)}")
    return HookTemplate(text=value, variables=variables)


def expand_template(template: HookTemplate, context: HookContext) -> str:
    """把一个已校验模板展开成动作真正使用的文本。

    Args:
        template: 启动阶段由 `parse_template` 生成的模板。
        context: 当前事件实际提供的工具、消息和错误数据。

    Returns:
        所有合法变量均已替换的字符串。当前事件没有相应值时替换为空串。
    """

    def replacement(match: re.Match[str]) -> str:
        """把当前匹配到的一个变量替换成事件中的对应值。

        Args:
            match: ``_VARIABLE`` 在模板原文中找到的一个合法变量。

        Returns:
            当前事件中的字符串值；字段不存在时返回空字符串。
        """

        name = match.group(0)
        if name == "$EVENT":
            return context.event.value
        if name == "$TOOL_NAME":
            return context.tool_name or ""
        if name == "$FILE_PATH":
            return context.file_path or ""
        if name == "$MESSAGE":
            return context.message or ""
        if name == "$ERROR":
            return context.error or ""
        key = name.removeprefix("$TOOL_ARGS.")
        if context.tool_args is None:
            return ""
        value = context.tool_args.get(key)
        return "" if value is None else str(value)

    return _VARIABLE.sub(replacement, template.text)
