"""解析 Hook 的简化条件语法，并用真实事件字段计算结果。"""

from __future__ import annotations

import fnmatch
import re

from mycode.errors import ConfigError
from mycode.models.hooks import (
    HookCondition,
    HookConditionGroup,
    HookConditionMode,
    HookContext,
    HookOperator,
)


_ATOM = re.compile(
    r"^\s*(event|tool|file_path|message|error|args\.[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*(==|!=|=~|~=)\s*(.+?)\s*$",
    re.DOTALL,
)


def _parse_expected(raw: str, *, source: str) -> str:
    """把条件右侧的引号字符串还原成普通文本。

    Args:
        raw: 操作符右侧包含引号的原始文本。
        source: 当前 Hook 的配置位置，用于错误提示。

    Returns:
        去掉首尾引号并处理引号、反斜杠转义后的目标字符串。
    """

    if not raw or raw[0] not in {"'", '"'}:
        raise ConfigError(f"{source} 的条件值必须使用单引号或双引号")
    quote = raw[0]
    if len(raw) < 2 or raw[-1] != quote:
        raise ConfigError(f"{source} 的条件字符串没有正确闭合")
    body = raw[1:-1]
    # 条件值只处理引号和反斜杠本身的转义。像正则中的 \s 必须原样
    # 保留，不能交给 Python 字符串解析器改写或产生转义警告。
    return body.replace("\\" + quote, quote).replace("\\\\", "\\")


def _parse_atom(expression: str, *, source: str) -> HookCondition:
    """把一段不含逻辑连接符的表达式转换为原子条件。

    Args:
        expression: 一段 ``字段 操作符 '目标值'`` 文本。
        source: 当前 Hook 的配置位置，用于错误提示。

    Returns:
        字段和操作符已经校验、正则已经预编译的原子条件。
    """

    match = _ATOM.fullmatch(expression)
    if match is None:
        raise ConfigError(f"{source} 的条件格式必须是 field operator 'value'")
    field, operator_text, raw_expected = match.groups()
    expected = _parse_expected(raw_expected, source=source)
    operator = HookOperator(operator_text)
    compiled = None
    if operator is HookOperator.REGEX:
        try:
            compiled = re.compile(expected)
        except re.error as exc:
            raise ConfigError(f"{source} 的正则表达式无效：{exc.msg}") from exc
    return HookCondition(field, operator, expected, compiled)


def _split_expression(
    value: str,
    *,
    source: str,
) -> tuple[list[str], HookConditionMode]:
    """只在引号外识别逻辑连接符和不支持的分组括号。

    Args:
        value: YAML ``if`` 字段中的完整表达式原文。
        source: 当前 Hook 的配置位置，用于错误提示。

    Returns:
        保留原文内容的原子条件列表，以及 ALL 或 ANY 组合方式。
    """

    pieces: list[str] = []
    operators: list[str] = []
    quote: str | None = None
    escaped = False
    start = 0
    index = 0
    while index < len(value):
        character = value[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
            index += 1
            continue
        if character in {"(", ")"}:
            raise ConfigError(f"{source} 的条件不支持括号嵌套")
        operator = (
            value[index : index + 2]
            if value[index : index + 2] in {"&&", "||"}
            else None
        )
        if operator is not None:
            pieces.append(value[start:index])
            operators.append(operator)
            index += 2
            start = index
            continue
        index += 1
    pieces.append(value[start:])
    if len(set(operators)) > 1:
        raise ConfigError(f"{source} 的条件不能混用 && 和 ||")
    if any(not piece.strip() for piece in pieces):
        raise ConfigError(f"{source} 的条件包含空子条件")
    mode = (
        HookConditionMode.ANY
        if operators and operators[0] == "||"
        else HookConditionMode.ALL
    )
    return pieces, mode


def parse_condition(value: object, *, source: str) -> HookConditionGroup:
    """解析一条 Hook 的完整条件表达式。

    Args:
        value: YAML `if` 字段读取到的字符串。
        source: 当前 Hook 的配置位置，用于生成可定位错误。

    Returns:
        已拆成原子条件的 ALL 或 ANY 条件组。

    Raises:
        ConfigError: 表达式混用逻辑符、使用括号或原子条件不合法。
    """

    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{source} 的 if 必须是非空字符串")
    pieces, mode = _split_expression(value, source=source)
    return HookConditionGroup(
        mode=mode,
        conditions=tuple(
            _parse_atom(piece, source=source) for piece in pieces
        ),
    )


def _field_value(condition: HookCondition, context: HookContext) -> object:
    """从事件上下文读取一个条件字段的原始值。

    Args:
        condition: 指明需要读取 ``event``、``tool`` 或 ``args.xxx`` 的条件。
        context: 当前生命周期接入点提供的真实事件数据。

    Returns:
        字段当前保存的原始值；当前事件没有该字段时返回 None。
    """

    if condition.field == "event":
        return context.event.value
    if condition.field == "tool":
        return context.tool_name
    if condition.field == "file_path":
        return context.file_path
    if condition.field == "message":
        return context.message
    if condition.field == "error":
        return context.error
    if context.tool_args is None:
        return None
    return context.tool_args.get(condition.field.removeprefix("args."))


def condition_matches(condition: HookCondition, context: HookContext) -> bool:
    """判断一个原子条件是否匹配当前事件。

    Args:
        condition: 启动阶段解析好的字段、操作符和目标值。
        context: 生命周期接入点提供的真实事件数据。

    Returns:
        字段存在且比较成功时返回 True；缺少字段或值不是标量时返回 False。
    """

    value = _field_value(condition, context)
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return False
    actual = str(value)
    if condition.operator is HookOperator.EQUALS:
        return actual == condition.expected
    if condition.operator is HookOperator.NOT_EQUALS:
        return actual != condition.expected
    if condition.operator is HookOperator.REGEX:
        assert condition.compiled_regex is not None
        return condition.compiled_regex.search(actual) is not None
    normalized_actual = actual.replace("\\", "/")
    normalized_pattern = condition.expected.replace("\\", "/")
    return fnmatch.fnmatchcase(normalized_actual, normalized_pattern)


def group_matches(group: HookConditionGroup, context: HookContext) -> bool:
    """按条件组声明的 ALL 或 ANY 方式计算全部子条件。

    Args:
        group: 启动阶段解析完成的条件组。
        context: 当前生命周期事件提供的实际数据。

    Returns:
        ALL 组全部匹配或 ANY 组至少一个匹配时返回 True。
    """

    results = (condition_matches(item, context) for item in group.conditions)
    return all(results) if group.mode is HookConditionMode.ALL else any(results)
