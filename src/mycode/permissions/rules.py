"""权限规则解析、glob 匹配与分层优先级决策。"""

from __future__ import annotations

import fnmatch
import os
import re

from mycode.errors import ConfigError
from mycode.models.permissions import (
    PermissionEffect,
    PermissionLayer,
    PermissionRule,
    PermissionScope,
    PermissionOperation,
    PermissionTool,
)

"""
预编译正则 检查并拆分权限规则 工具名(匹配模式)
例如：Shell(git *)
     ReadFile(src/**)
     WriteFile(README.md)
"""
_RULE_EXPRESSION = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*)\((.*)\)$", re.DOTALL)

"""
记录哪些权限工具处理的是路径或文件匹配模式
"""
_PATH_TOOLS = {
    PermissionTool.READ_FILE,
    PermissionTool.WRITE_FILE,
    PermissionTool.FIND_FILES,
    PermissionTool.SEARCH_CODE,
}


def _character_class_end(pattern: str, start: int) -> int:
    """
    从 [ 后面开始寻找真正结束字符类的 ]，同时避开取反符号和作为普通字符出现的 ]

    Parameters:
        pattern: 一条权限规则括号里面的“匹配模式字符串” 例如：ReadFile(src/[ab]*.py): allow 中的src/[ab]*.py
        start: 是当前字符类左方括号 [ 在 pattern 中的位置
    Returns:
        int: 真正用于结束字符类的]的位置
    """
    index = start + 1
    if index < len(pattern) and pattern[index] in {"!", "^"}:
        index += 1
    if index < len(pattern) and pattern[index] == "]":
        index += 1
    return pattern.find("]", index)


def _validate_glob(pattern: str, source: str) -> None:
    """
    校验正则表达式

     Parameters:
        pattern: 一条权限规则括号里面的“匹配模式字符串” 例如：ReadFile(src/[ab]*.py): allow 中的src/[ab]*.py
        source: 规则来源说明，用于在配置错误中指出所属权限层和文件
    """
    index = 0
    while index < len(pattern):
        if pattern[index] != "[":
            index += 1
            continue
        end = _character_class_end(pattern, index)
        if end < 0:
            raise ConfigError(f"{source} 的 glob 包含未闭合字符类")
        if end == index + 1:
            raise ConfigError(f"{source} 的 glob 包含空字符类")
        index = end + 1


def _glob_stats(pattern: str) -> tuple[int, int, bool]:
    """
    用来分析一条 glob 模式，统计其中有多少普通字符、多少通配结构，以及它是否属于通配规则（是否包含 Glob 通配语法）

    Parameters:
        pattern: 一条权限规则括号里面的“匹配模式字符串” 例如：ReadFile(src/[ab]*.py): allow 中的src/[ab]*.py
    Returns:
        tuple[int, int, bool]: 普通字符数，通配符数，是否属于统配规则（是否包含 Glob 通配语法）
    """
    # 普通字符数量
    literal_count = 0
    # 通配符数量
    wildcard_count = 0
    # 标志是否含有通配符
    has_glob = False
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            has_glob = True
            wildcard_count += 1
            index += 2 if index + 1 < len(pattern) and pattern[index + 1] == "*" else 1
            continue
        if char == "?":
            has_glob = True
            wildcard_count += 1
            index += 1
            continue
        if char == "[":
            end = _character_class_end(pattern, index)
            if end < 0:
                break
            has_glob = True
            wildcard_count += 1
            index = end + 1
            continue
        literal_count += 1
        index += 1
    return literal_count, wildcard_count, has_glob


def _path_glob_regex(pattern: str) -> str:
    r"""
    把路径权限规则使用的 glob 模式转换成正则表达式字符串
    例如：配置中ReadFile(src/*.py): allow
    函数会把src/*.py转换为\Asrc/[^/]*\.py\Z

    Parameters:
        pattern: 一条权限规则括号里面的“匹配模式字符串” 例如：ReadFile(src/[ab]*.py): allow 中的src/[ab]*.py
    Returns:
        str: 正则表达式的字符串
    """
    pieces = [r"\A"]
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                if index + 2 < len(pattern) and pattern[index + 2] == "/":
                    pieces.append("(?:.*/)?")
                    index += 3
                else:
                    pieces.append(".*")
                    index += 2
            else:
                pieces.append("[^/]*")
                index += 1
            continue
        if char == "?":
            pieces.append("[^/]")
            index += 1
            continue
        if char == "[":
            end = _character_class_end(pattern, index)
            if end < 0:
                raise AssertionError("glob 必须在编译前完成校验")
            content = pattern[index + 1 : end]
            if content.startswith("!"):
                content = "^" + content[1:]
            elif content.startswith("^"):
                content = "\\" + content
            content = content.replace("\\", r"\\")
            pieces.append("[" + content + "]")
            index = end + 1
            continue
        pieces.append(re.escape(char))
        index += 1
    pieces.append(r"\Z")
    return "".join(pieces)


def _compile_matcher(
    tool: PermissionTool,
    pattern: str,
    *,
    is_glob: bool,
) -> re.Pattern[str]:
    """
    根据工具类型和规则类型，选择正确的 Glob 翻译方式和大小写策略，最终生成统一、可复用的正则匹配器
    """
    flags = re.IGNORECASE if tool in _PATH_TOOLS and os.name == "nt" else 0
    if not is_glob:
        return re.compile(r"\A" + re.escape(pattern) + r"\Z", flags)
    translated = (
        fnmatch.translate(pattern)
        if tool in {
            PermissionTool.SHELL,
            PermissionTool.MCP,
            PermissionTool.SKILL,
        }
        else _path_glob_regex(pattern)
    )
    return re.compile(translated, flags)


def make_rule(
    # 当条规则针对的工具
    tool: PermissionTool,
    # 这条规则要匹配的具体内容，也就是 YAML 规则括号里的部分
    pattern: str,
    effect: PermissionEffect,
    *,
    scope: PermissionScope,
    source: str,
    #是否强制把 pattern 当成普通文本进行精确匹配，即使里面包含 *、?、[] 等 Glob 特殊字符
    force_exact: bool = False,
) -> PermissionRule:
    """
    根据工具、模式、效果和作用范围创建可执行权限规则。

    路径模式会统一分隔符；普通规则会校验并分析 glob，
    精确规则按字面量处理。函数同时预编译匹配器并计算
    规则具体程度。
    """
    #标准化路径，把Windows的\统一为/
    normalized = pattern.replace("\\", "/") if tool in _PATH_TOOLS else pattern
    if not normalized:
        raise ConfigError(f"{source} 的权限规则模式不能为空")
    if force_exact:
        literal_count, wildcard_count, has_glob = len(normalized), 0, False
    else:
        _validate_glob(normalized, source)
        literal_count, wildcard_count, has_glob = _glob_stats(normalized)
    is_glob = has_glob
    matcher = _compile_matcher(tool, normalized, is_glob=is_glob)
    specificity = (
        literal_count if is_glob else len(normalized),
        -wildcard_count if is_glob else 0,
        len(normalized),
    )
    return PermissionRule(
        tool=tool,
        pattern=normalized,
        effect=effect,
        scope=scope,
        is_glob=is_glob,
        specificity=specificity,
        source=source,
        matcher=matcher,
    )


def make_exact_allow_rule(
    operation: PermissionOperation,
    *,
    scope: PermissionScope,
    source: str,
) -> PermissionRule:
    """
    根据一次具体操作，生成一条只允许完全相同操作的精确规则，主要用于用户选择在本次会话允许
    """
    return make_rule(
        operation.tool,
        operation.match_value,
        PermissionEffect.ALLOW,
        scope=scope,
        source=source,
        force_exact=True,
    )


def parse_rule(
    expression: str,
    effect: str,
    *,
    scope: PermissionScope,
    source: str,
) -> PermissionRule:
    """
     解析一条配置中的权限规则，并生成可直接匹配权限操作的 PermissionRule。

    规则表达式必须采用“权限工具名(匹配模式)”格式，例如：
    `ReadFile(src/**/*.py)`、`WriteFile(src/**)` 或
    `Shell(git status)`。effect 只接受 `allow` 或 `deny`。

    Args:
        expression: 完整的权限规则表达式。
        effect: 规则命中后的效果，只能是 `allow` 或 `deny`。
        scope: 规则所属的权限层。
        source: 规则来源说明，用于生成可定位的配置错误。

    Returns:
        经过校验和编译的 PermissionRule。

    Raises:
        ConfigError: 规则格式、工具名称或规则效果不合法。
    """
    if not isinstance(expression, str):
        raise ConfigError(f"{source} 的权限规则名必须是字符串")
    match = _RULE_EXPRESSION.fullmatch(expression)
    if match is None:
        raise ConfigError(
            f"{source} 的权限规则必须写成 工具名(模式)：{expression!r}"
        )
    tool_name, pattern = match.groups()
    try:
        tool = PermissionTool(tool_name)
    except ValueError as exc:
        raise ConfigError(
            f"{source} 包含未知权限工具：{tool_name}"
        ) from exc
    try:
        parsed_effect = PermissionEffect(effect)
    except ValueError as exc:
        raise ConfigError(
            f"{source} 的规则结果只支持 allow 或 deny：{effect!r}"
        ) from exc
    return make_rule(
        tool,
        pattern,
        parsed_effect,
        scope=scope,
        source=source,
    )


def format_rule_expression(tool: PermissionTool, pattern: str) -> str:
    """
    格式化权限配置中的规则表达式字符串
    """
    return f"{tool.value}({pattern})"


def format_exact_rule_expression(operation: PermissionOperation) -> str:
    """把待授权操作编码为只匹配其字面量文本的 glob 表达式。"""

    escaped = "".join(
        {
            "*": "[*]",
            "?": "[?]",
            "[": "[[]",
            "]": "[]]",
        }.get(character, character)
        for character in operation.match_value
    )
    return format_rule_expression(operation.tool, escaped)


class PermissionRuleResolver:
    """
    分层权限规则选择器
    从 SESSION、LOCAL、PROJECT、USER 四层规则中，找出对当前操作最应该生效的那一条 PermissionRule。
    """
    def __init__(
        self,
        user: PermissionLayer,
        project: PermissionLayer,
    ) -> None:
        if user.scope is not PermissionScope.USER:
            raise ValueError("规则决议器 user 层 scope 错误")
        if project.scope is not PermissionScope.PROJECT:
            raise ValueError("规则决议器 project 层 scope 错误")
        self._user = user
        self._project = project

    def resolve(
        self,
        operation: PermissionOperation,
        *,
        session_rules: tuple[PermissionRule, ...],
        local: PermissionLayer,
    ) -> PermissionRule | None:
        """
            按权限层级和规则具体程度，解析当前操作最终命中的权限规则。

            规则首先按照 SESSION、LOCAL、PROJECT、USER 的优先级逐层查找。
            一旦某一层存在匹配规则，就只在该层内选择，不再检查低优先级层。

            同一层内的选择顺序为：
            1. 精确规则优先于 Glob 通配规则；
            2. 选择 specificity 最高的规则；
            3. 具体程度相同时，DENY 优先于 ALLOW；
            4. 仍有多条候选时，按稳定字段排序，避免结果受配置顺序影响。

            Args:
                operation: 当前需要进行权限匹配的具体工具操作。
                session_rules: 当前会话动态添加的权限规则。
                local: 当前项目最新的本地权限层快照。

            Returns:
                最终生效的 PermissionRule；所有权限层均无匹配规则时返回 None。
            """
        if local.scope is not PermissionScope.LOCAL:
            raise ValueError("规则决议器 local 层 scope 错误")
        # 权限层从高到低排列
        layers = (
            session_rules,
            local.rules,
            self._project.rules,
            self._user.rules,
        )
        for rules in layers:
            # 找出当前权限层中所有能完整匹配本次操作的规则
            matches = [rule for rule in rules if rule.matches(operation)]
            if not matches:
                continue

            # 当前层只要存在精确规则，就忽略同层所有 Glob 通配规则
            exact = [rule for rule in matches if not rule.is_glob]
            # 利用or的短路特性，要是exact有值，则表达式直接取值为exact
            candidates = exact or matches

            # 在精确性相同的候选规则中，选择具体程度最高的规则
            best_specificity = max(rule.specificity for rule in candidates)
            candidates = [
                rule
                for rule in candidates
                if rule.specificity == best_specificity
            ]

            # 具体程度相同时采用安全优先策略：DENY优先于ALLOW
            denied = [
                rule
                for rule in candidates
                if rule.effect is PermissionEffect.DENY
            ]
            selected = denied or candidates

            # 若还有多条规则候选，则排序，确保结果不受YAML书写顺序或规则加载顺序影响
            return sorted(
                selected,
                key=lambda rule: (
                    rule.tool.value,
                    rule.pattern,
                    rule.effect.value,
                    rule.source,
                ),
            )[0]
        return None
