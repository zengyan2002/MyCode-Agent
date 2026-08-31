"""YAML 配置、两层合并、环境变量展开与字段校验。"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import yaml
from dotenv import dotenv_values

from mycode.errors import ConfigError
from mycode.constants import (
    AUTO_COMPACTION_MARGIN_TOKENS,
    LOCAL_HOOK_CONFIG_RELATIVE_PATH,
)
from mycode.hooks.config import parse_hook_layers
from mycode.models.config import (
    AgentSettings,
    AppConfig,
    ExpandedConfigValue,
    HttpMcpServerConfig,
    McpServerConfig,
    Protocol,
    ProviderConfig,
    SecretValue,
    StdioMcpServerConfig,
    ThinkingMode,
    WorktreeIgnoredRule,
    WorktreePathRule,
    WorktreeSettings,
)
from mycode.models.hooks import HookDefinition, HookLayer

# 匹配配置字符串中的环境变量模板  例如可以匹配 ${HOST}
_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
# 校验 MCP Server 名能安全参与工具命名空间  例如可以匹配 web-search
_MCP_SERVER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
# stdio MCP Server 的 env 键必须是跨平台可识别的环境变量名，例如
# API_KEY 或 LOG_LEVEL。
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# 用于覆盖项目级配置文件路径的环境变量名。
MYCODE_CONFIG_ENV = "MYCODE_CONFIG"

# 通用配置允许出现的顶层字段。
_TOP_LEVEL_FIELDS = {
    "active",
    "providers",
    "mcp_servers",
    "hooks",
    "agents",
    "worktrees",
}
# Provider 配置允许出现的字段。
_PROVIDER_FIELDS = {
    "name",
    "protocol",
    "model",
    "base_url",
    "api_key",
    "thinking",
    "context_window_tokens",
    "compaction_output_tokens",
    "tool_result_spill_chars",
    "tool_batch_spill_chars",
}
# stdio MCP Server 配置允许出现的字段。
_STDIO_SERVER_FIELDS = {"type", "command", "args", "env"}
# HTTP MCP Server 配置允许出现的字段。
_HTTP_SERVER_FIELDS = {"type", "url", "headers"}
# 独立子 Agent 配置块只允许调整这四个已批准的运行参数。
_AGENT_FIELDS = {
    "auto_background_seconds",
    "agent_tool_timeout_seconds",
    "max_background_tasks",
    "enable_verification",
}
# Worktree 配置块只允许这些已经进入初始化和清理流程的字段。
_WORKTREE_FIELDS = {
    "stale_after_hours",
    "cleanup_interval_seconds",
    "copy_files",
    "symlink_directories",
    "copy_ignored",
    "hooks_path",
}
_WORKTREE_RULE_FIELDS = {"path", "required"}
_WORKTREE_IGNORED_RULE_FIELDS = {"pattern", "required"}


class _UniqueKeyLoader(yaml.SafeLoader):
    """安全加载 YAML，并拒绝同一映射中的重复键。"""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    """构造 YAML 映射并在键被覆盖前识别重复项。

    Args:
        loader: 当前 YAML 安全加载器。
        node: 等待转换的 YAML 映射节点。
        deep: 是否递归构造节点中的所有嵌套对象。

    Returns:
        不包含重复键的普通 Python 字典。
    """
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConfigError("YAML 映射键必须是可比较的标量") from exc
        if duplicate:
            raise ConfigError(f"YAML 包含重复键：{key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_project_environment(
    dotenv_path: Path,
    base_environment: Mapping[str, str],
) -> dict[str, str]:
    """合并当前项目的 ``.env`` 与进程环境。

    Args:
        dotenv_path: 当前工作目录下需要读取的 ``.env`` 路径。
        base_environment: 优先级更高的进程或测试环境。

    Returns:
        由 ``.env`` 打底、进程环境覆盖后的新字典。
    """
    merged: dict[str, str] = {}
    if dotenv_path.exists():
        try:
            raw_text = dotenv_path.read_text(encoding="utf-8")
        except UnicodeError as exc:
            raise ConfigError(
                f"无法读取 .env 文件 {dotenv_path}：内容不是有效的 UTF-8 文本"
            ) from exc
        except OSError as exc:
            raise ConfigError(
                f"无法读取 .env 文件 {dotenv_path}：{type(exc).__name__}"
            ) from exc

        dotenv_environment = dotenv_values(
            stream=StringIO(raw_text),
            interpolate=False,
        )
        merged.update(
            {
                key: value
                for key, value in dotenv_environment.items()
                if value is not None
            }
        )

    merged.update(base_environment)
    return merged


def default_config_path() -> Path:
    """返回默认用户级配置路径。

    Returns:
        当前用户主目录下的 ``.mycode/config.yaml``。
    """
    return Path.home() / ".mycode" / "config.yaml"


def default_project_config_path(working_directory: Path) -> Path:
    """返回工作区默认项目级配置路径。

    Args:
        working_directory: MyCode 启动时固定的工作区。

    Returns:
        工作区下的 ``.mycode/config.yaml``。
    """
    return working_directory / ".mycode" / "config.yaml"


def _resolve_project_config_path(
    working_directory: Path,
    environment: Mapping[str, str],
) -> Path:
    """解析两层加载中的项目级配置路径。

    Args:
        working_directory: 相对路径解析和默认项目路径使用的工作区。
        environment: 可能包含 ``MYCODE_CONFIG`` 的合并环境。

    Returns:
        显式项目配置或工作区默认项目配置的规范化路径。
    """
    raw_path = environment.get(MYCODE_CONFIG_ENV)
    if raw_path is None or not raw_path.strip():
        #
        return default_project_config_path(working_directory).resolve(
            strict=False
        )

    config_path = Path(raw_path.strip())
    if not config_path.is_absolute():
        config_path = working_directory / config_path
    return config_path.resolve(strict=False)


def _read_config_layer(
    path: Path,
    *,
    label: str,
    required: bool,
) -> dict[str, Any]:
    """读取并校验一层通用 YAML 配置的根结构。

    Args:
        path: 需要读取的配置文件。
        label: 错误信息中显示的配置层名称。
        required: 文件缺失时是否立即报错。

    Returns:
        通过根结构校验的配置字典；可选文件缺失或为空时返回空字典。
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        if required:
            raise ConfigError(f"配置文件不存在：{path}") from exc
        return {}
    except (OSError, UnicodeError) as exc:
        raise ConfigError(
            f"无法读取{label}配置 {path}：{type(exc).__name__}"
        ) from exc

    try:
        raw = yaml.load(raw_text, Loader=_UniqueKeyLoader)
    except ConfigError as exc:
        raise ConfigError(f"{label}配置 {path}：{exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{label}配置 {path} 的 YAML 格式错误：{exc}") from exc

    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label}配置 {path} 顶层必须是映射")
    if any(not isinstance(key, str) for key in raw):
        raise ConfigError(f"{label}配置 {path} 顶层键必须是字符串")
    unknown = sorted(set(raw) - _TOP_LEVEL_FIELDS)
    if unknown:
        raise ConfigError(
            f"{label}配置 {path} 包含未知字段：{', '.join(unknown)}"
        )
    return dict(raw)


def _required_string(
    value: Mapping[str, Any],
    field: str,
    path: str,
) -> str:
    """读取一个必填的非空字符串字段。

    Args:
        value: 字段所属配置映射。
        field: 需要读取的字段名。
        path: 用于诊断的配置位置。

    Returns:
        去除首尾空白后的字段值。
    """
    raw = value.get(field)
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"配置项 {path}.{field} 必须是非空字符串")
    return raw.strip()


def _parse_thinking(
    raw: Any,
    path: str,
    protocol: Protocol,
) -> ThinkingMode:
    """解析并校验 Provider 思考模式。

    Args:
        raw: YAML 中的原始字段值。
        path: Provider 的诊断路径。
        protocol: 当前 Provider 使用的协议。

    Returns:
        校验完成的思考模式枚举。
    """
    if raw is None or raw is False:
        mode = ThinkingMode.DISABLED
    elif isinstance(raw, str) and raw in {
        ThinkingMode.ENABLED,
        ThinkingMode.ADAPTIVE,
    }:
        mode = ThinkingMode(raw)
    else:
        raise ConfigError(
            f"配置项 {path}.thinking 必须是 false、enabled 或 adaptive"
        )
    if protocol is Protocol.OPENAI and mode is not ThinkingMode.DISABLED:
        raise ConfigError(f"配置项 {path}.thinking：OpenAI 协议只允许 false")
    return mode


def _positive_integer(raw: Any, *, path: str, default: int) -> int:
    """读取一个可选正整数配置。

    Args:
        raw: YAML 字段原值；字段缺失时为 ``None``。
        path: 错误信息中使用的完整配置位置。
        default: 字段缺失时返回的产品默认值。

    Returns:
        用户提供的正整数，或字段缺失时的默认值。
    """
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ConfigError(f"配置项 {path} 必须是正整数")
    return raw


def _parse_url(raw: str, path: str) -> str:
    """校验 HTTP/HTTPS URL 并拒绝内嵌凭据。

    Args:
        raw: 等待校验的完整 URL。
        path: URL 字段的诊断路径。

    Returns:
        未改变的合法 URL。
    """
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError(f"配置项 {path} 必须是完整的 HTTP/HTTPS 地址")
    if parsed.username is not None or parsed.password is not None:
        raise ConfigError(f"配置项 {path} 不能包含用户名或密码")
    return raw


def _parse_api_key(
    raw: Any,
    path: str,
    environment: Mapping[str, str],
) -> SecretValue:
    """解析 Provider API Key，并支持完整变量引用。

    Args:
        raw: YAML 中的 API Key 原始值。
        path: Provider 的诊断路径。
        environment: 用于变量解析的合并环境。

    Returns:
        不会在普通打印中泄露的敏感值。
    """
    if not isinstance(raw, str) or not raw:
        raise ConfigError(f"配置项 {path}.api_key 必须是非空字符串")
    match = _ENV_REFERENCE.fullmatch(raw)
    if match:
        variable = match.group(1)
        value = environment.get(variable)
        if not value:
            raise ConfigError(
                f"配置项 {path}.api_key 引用的环境变量 {variable} "
                "不存在或为空"
            )
        return SecretValue(value)
    if "${" in raw or "}" in raw:
        raise ConfigError(
            f"配置项 {path}.api_key 的环境变量引用必须完整写成 "
            "${ENV_VAR}"
        )
    return SecretValue(raw)


def _parse_provider(
    raw: Any,
    index: int,
    environment: Mapping[str, str],
    *,
    layer_label: str = "",
) -> ProviderConfig:
    """解析一条完整 Provider 配置。

    Args:
        raw: YAML 中的 Provider 原始对象。
        index: Provider 在当前列表中的索引。
        environment: 用于 API Key 解析的环境。
        layer_label: 可选的用户级或项目级诊断前缀。

    Returns:
        校验完成的 Provider 配置。
    """
    prefix = f"{layer_label}." if layer_label else ""
    path = f"{prefix}providers[{index}]"
    if not isinstance(raw, Mapping):
        raise ConfigError(f"配置项 {path} 必须是映射")
    unknown = sorted(set(raw) - _PROVIDER_FIELDS)
    if unknown:
        raise ConfigError(
            f"配置项 {path} 包含未知字段：{', '.join(unknown)}"
        )

    name = _required_string(raw, "name", path)
    protocol_raw = _required_string(raw, "protocol", path)
    try:
        protocol = Protocol(protocol_raw)
    except ValueError as exc:
        raise ConfigError(
            f"配置项 {path}.protocol 只支持 anthropic 或 openai"
        ) from exc
    model = _required_string(raw, "model", path)
    base_url = _parse_url(
        _required_string(raw, "base_url", path),
        f"{path}.base_url",
    )
    api_key = _parse_api_key(raw.get("api_key"), path, environment)
    thinking = _parse_thinking(raw.get("thinking"), path, protocol)
    defaults = ProviderConfig(
        name=name,
        protocol=protocol,
        model=model,
        base_url=base_url,
        api_key=api_key,
        thinking=thinking,
    )
    context_window_tokens = _positive_integer(
        raw.get("context_window_tokens"),
        path=f"{path}.context_window_tokens",
        default=defaults.context_window_tokens,
    )
    compaction_output_tokens = _positive_integer(
        raw.get("compaction_output_tokens"),
        path=f"{path}.compaction_output_tokens",
        default=defaults.compaction_output_tokens,
    )
    tool_result_spill_chars = _positive_integer(
        raw.get("tool_result_spill_chars"),
        path=f"{path}.tool_result_spill_chars",
        default=defaults.tool_result_spill_chars,
    )
    tool_batch_spill_chars = _positive_integer(
        raw.get("tool_batch_spill_chars"),
        path=f"{path}.tool_batch_spill_chars",
        default=defaults.tool_batch_spill_chars,
    )
    if (
        context_window_tokens
        <= compaction_output_tokens + AUTO_COMPACTION_MARGIN_TOKENS
    ):
        raise ConfigError(
            f"配置项 {path}.context_window_tokens 必须大于 "
            "compaction_output_tokens 与 13000 的和"
        )
    return ProviderConfig(
        name=name,
        protocol=protocol,
        model=model,
        base_url=base_url,
        api_key=api_key,
        thinking=thinking,
        context_window_tokens=context_window_tokens,
        compaction_output_tokens=compaction_output_tokens,
        tool_result_spill_chars=tool_result_spill_chars,
        tool_batch_spill_chars=tool_batch_spill_chars,
    )


def _expand_config_value(
    raw: Any,
    *,
    path: str,
    environment: Mapping[str, str],
    protect_literal: bool = False,
) -> ExpandedConfigValue:
    """展开 MCP 配置字符串中的全部环境变量模板。

    Args:
        raw: YAML 中等待展开的字符串。
        path: 当前值在 YAML 配置中的完整字段位置，用于生成错误提示。
        environment: 变量取值使用的合并环境。
        protect_literal: 没有模板时是否也把完整值视为敏感信息。

    Returns:
        展开后的原文和需要脱敏的变量片段。
    """
    if not isinstance(raw, str):
        raise ConfigError(f"配置项 {path} 必须是字符串")

    secret_parts: list[SecretValue] = []

    def replace(match: re.Match[str]) -> str:
        """
        将单个环境变量占位符替换为对应的环境变量值。

        Args:
            match: 正则表达式匹配到的环境变量占位符。

        Returns:
            展开后的环境变量值。
        """
        variable = match.group(1)
        value = environment.get(variable)
        if not value:
            raise ConfigError(
                f"配置项 {path} 引用的环境变量 {variable} 不存在或为空"
            )
        secret_parts.append(SecretValue(value))
        return value

    expanded = _ENV_REFERENCE.sub(replace, raw)
    if "${" in expanded or "}" in expanded:
        raise ConfigError(f"配置项 {path} 包含无效的环境变量模板")
    if protect_literal and not secret_parts and expanded:
        secret_parts.append(SecretValue(expanded))
    return ExpandedConfigValue(expanded, tuple(secret_parts))


def _parse_string_list(raw: Any, *, path: str) -> tuple[str, ...]:
    """校验 YAML 中的 args 是字符串列表，然后转换成不可变元组

    Args:
        raw: YAML 中的原始列表值。
        path: 当前值在 YAML 配置中的完整字段位置，用于生成错误提示。

    Returns:
        保持原顺序的不可变字符串元组。
    """
    if raw is None:
        return ()
    if not isinstance(raw, list) or any(
        not isinstance(item, str) for item in raw
    ):
        raise ConfigError(f"配置项 {path} 必须是字符串列表")
    return tuple(raw)


def _parse_value_mapping(
    raw: Any,
    *,
    path: str,
    environment: Mapping[str, str],
    headers: bool,
) -> tuple[tuple[str, ExpandedConfigValue], ...]:
    """解析 MCP 环境或请求头映射。

    Args:
        raw: YAML 中的原始映射值。
        path: 映射字段的完整诊断路径。
        environment: 用于模板展开的合并环境。
        headers: 当前映射是否为 HTTP 请求头。

    Returns:
        保持声明顺序的键和展开值元组。
    """
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise ConfigError(f"配置项 {path} 必须是映射")

    parsed: list[tuple[str, ExpandedConfigValue]] = []
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise ConfigError(f"配置项 {path} 的键必须是非空字符串")
        if not headers and _ENVIRONMENT_NAME.fullmatch(key) is None:
            raise ConfigError(f"配置项 {path}.{key} 的环境变量名无效")
        # 所有 HTTP 请求头值都可能承载租户、会话或认证信息；即使没有
        # 使用环境变量模板，也统一纳入应用敏感值集合。
        protect_literal = headers
        parsed.append(
            (
                key,
                _expand_config_value(
                    value,
                    path=f"{path}.{key}",
                    environment=environment,
                    protect_literal=protect_literal,
                ),
            )
        )
    return tuple(parsed)


def _parse_mcp_server(
    name: str,
    raw: Any,
    environment: Mapping[str, str],
    *,
    layer_label: str,
) -> McpServerConfig:
    """解析一条 stdio 或 HTTP MCP Server 配置。

    Args:
        name: Server 在配置 map 中的键。
        raw: YAML 中的 Server 配置映射。
        environment: 用于变量展开的合并环境。
        layer_label: 用户级或项目级诊断前缀。

    Returns:
        与传输类型对应的不可变 Server 配置。
    """
    path = f"{layer_label}.mcp_servers.{name}"
    if _MCP_SERVER_NAME.fullmatch(name) is None:
        raise ConfigError(
            f"配置项 {path} 的 Server 名必须以英文字母开头，且只能包含"
            "英文字母、数字、下划线或连字符"
        )
    if not isinstance(raw, Mapping):
        raise ConfigError(f"配置项 {path} 必须是映射")

    transport = _required_string(raw, "type", path)
    if transport == "stdio":
        unknown = sorted(set(raw) - _STDIO_SERVER_FIELDS)
        if unknown:
            raise ConfigError(
                f"配置项 {path} 包含未知字段：{', '.join(unknown)}"
            )
        return StdioMcpServerConfig(
            name=name,
            command=_required_string(raw, "command", path),
            args=_parse_string_list(raw.get("args"), path=f"{path}.args"),
            env=_parse_value_mapping(
                raw.get("env"),
                path=f"{path}.env",
                environment=environment,
                headers=False,
            ),
        )
    if transport == "streamable_http":
        unknown = sorted(set(raw) - _HTTP_SERVER_FIELDS)
        if unknown:
            raise ConfigError(
                f"配置项 {path} 包含未知字段：{', '.join(unknown)}"
            )
        url = _parse_url(
            _required_string(raw, "url", path),
            f"{path}.url",
        )
        return HttpMcpServerConfig(
            name=name,
            url=url,
            headers=_parse_value_mapping(
                raw.get("headers"),
                path=f"{path}.headers",
                environment=environment,
                headers=True,
            ),
        )
    raise ConfigError(
        f"配置项 {path}.type 只支持 stdio 或 streamable_http"
    )


def _parse_provider_layer(
    raw: Mapping[str, Any],
    environment: Mapping[str, str],
    *,
    layer_label: str,
) -> tuple[ProviderConfig, ...]:
    """解析单层中可选的 Provider 列表。

    Args:
        raw: 当前配置层根映射。
        environment: 用于 API Key 解析的环境。
        layer_label: 当前配置层的诊断名称。

    Returns:
        当前层按声明顺序排列的 Provider 元组。
    """
    raw_providers = raw.get("providers")
    if raw_providers is None:
        return ()
    if not isinstance(raw_providers, list):
        raise ConfigError(f"配置项 {layer_label}.providers 必须是列表")

    providers = tuple(
        _parse_provider(
            provider,
            index,
            environment,
            layer_label=layer_label,
        )
        for index, provider in enumerate(raw_providers)
    )
    names = [provider.name for provider in providers]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigError(
            f"{layer_label} Provider 名称必须唯一，重复项："
            f"{', '.join(duplicates)}"
        )
    return providers


def _parse_mcp_layer(
    raw: Mapping[str, Any],
    environment: Mapping[str, str],
    *,
    layer_label: str,
) -> tuple[McpServerConfig, ...]:
    """从当前这一份配置文件里，读取并解析 mcp_servers 这一块内容

    Args:
        raw: 当前配置层根映射。
        environment: 用于模板展开的环境。
        layer_label: 当前配置层的诊断名称。

    Returns:
        当前层按声明顺序排列的 MCP Server 元组。
    """
    raw_servers = raw.get("mcp_servers")
    if raw_servers is None:
        return ()
    if not isinstance(raw_servers, Mapping):
        raise ConfigError(f"配置项 {layer_label}.mcp_servers 必须是映射")
    if any(not isinstance(name, str) for name in raw_servers):
        raise ConfigError(
            f"配置项 {layer_label}.mcp_servers 的 Server 名必须是字符串"
        )
    return tuple(
        _parse_mcp_server(
            name,
            server,
            environment,
            layer_label=layer_label,
        )
        for name, server in raw_servers.items()
    )


def _parse_optional_active(
    raw: Mapping[str, Any],
    *,
    layer_label: str,
) -> str | None:
    """解析单层中可选的 ``active`` 字段。

    Args:
        raw: 当前配置层根映射。
        layer_label: 当前配置层的诊断名称。

    Returns:
        去除首尾空白的 Provider 名；字段缺失时返回 ``None``。
    """
    if "active" not in raw:
        return None
    active = raw.get("active")
    if not isinstance(active, str) or not active.strip():
        raise ConfigError(f"配置项 {layer_label}.active 必须是非空字符串")
    return active.strip()


def _merge_named(
    base: tuple[Any, ...],
    override: tuple[Any, ...],
) -> tuple[Any, ...]:
    """按对象的 ``name`` 属性执行稳定整项覆盖。
       项目级优先，同名整体覆盖，新名称追加，原有顺序保持稳定。

    Args:
        base: 优先级较低、先进入结果的对象。
        override: 优先级较高、用于覆盖或追加的对象。

    Returns:
        同名原位替换、新名称追加后的不可变元组。
    """
    merged = list(base)
    positions = {item.name: index for index, item in enumerate(merged)}
    for item in override:
        index = positions.get(item.name)
        if index is None:
            positions[item.name] = len(merged)
            merged.append(item)
        else:
            merged[index] = item
    return tuple(merged)


def _parse_agent_layer(
    raw: Mapping[str, Any],
    *,
    layer_label: str,
) -> dict[str, float | int | bool]:
    """读取单层 ``agents`` 配置并只返回该层明确填写的字段。

    Args:
        raw: 当前用户级、项目级或单文件配置的根映射。
        layer_label: 错误消息中标识当前配置层的名字。

    Returns:
        已完成严格类型和值校验的字段字典。未填写 ``agents`` 时返回空字典。

    Raises:
        ConfigError: ``agents`` 不是映射、包含未知字段，或字段类型和值无效。
    """

    value = raw.get("agents")
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"配置项 {layer_label}.agents 必须是映射")
    if any(not isinstance(key, str) for key in value):
        raise ConfigError(f"配置项 {layer_label}.agents 的字段名必须是字符串")
    unknown = sorted(set(value) - _AGENT_FIELDS)
    if unknown:
        raise ConfigError(
            f"配置项 {layer_label}.agents 包含未知字段：{', '.join(unknown)}"
        )

    parsed: dict[str, float | int | bool] = {}
    if "auto_background_seconds" in value:
        seconds = value["auto_background_seconds"]
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or seconds < 0
        ):
            raise ConfigError(
                f"配置项 {layer_label}.agents.auto_background_seconds "
                "必须是非负数"
            )
        parsed["auto_background_seconds"] = float(seconds)
    if "agent_tool_timeout_seconds" in value:
        timeout = value["agent_tool_timeout_seconds"]
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ConfigError(
                f"配置项 {layer_label}.agents.agent_tool_timeout_seconds "
                "必须是正数"
            )
        parsed["agent_tool_timeout_seconds"] = float(timeout)
    if "max_background_tasks" in value:
        concurrency = value["max_background_tasks"]
        if (
            isinstance(concurrency, bool)
            or not isinstance(concurrency, int)
            or concurrency <= 0
        ):
            raise ConfigError(
                f"配置项 {layer_label}.agents.max_background_tasks "
                "必须是正整数"
            )
        parsed["max_background_tasks"] = concurrency
    if "enable_verification" in value:
        enabled = value["enable_verification"]
        if not isinstance(enabled, bool):
            raise ConfigError(
                f"配置项 {layer_label}.agents.enable_verification "
                "必须是布尔值"
            )
        parsed["enable_verification"] = enabled
    return parsed


def _merge_agent_settings(
    base: Mapping[str, float | int | bool],
    override: Mapping[str, float | int | bool],
) -> AgentSettings:
    """按单字段合并用户级与项目级子 Agent 配置。

    Args:
        base: 低优先级配置中明确填写的字段。
        override: 高优先级配置中明确填写的字段。

    Returns:
        缺失字段使用 ``AgentSettings`` 默认值、项目字段覆盖用户字段的
        完整配置对象。
    """

    values: dict[str, float | int | bool] = dict(base)
    values.update(override)
    try:
        return AgentSettings(**values)
    except ValueError as exc:
        # 单层字段已经完成类型检查；这里报告的是用户层与项目层合并后才
        # 能判断的时间关系，例如项目层把自动移交设得比专用超时更长。
        raise ConfigError(f"配置项 agents 无效：{exc}") from exc


def _validate_worktree_rule_text(
    value: object,
    *,
    field_name: str,
) -> str:
    """校验初始化来源只能指向仓库内的普通相对位置。

    Args:
        value: YAML 中读取出的路径或 ignored 模式值。
        field_name: 错误消息中显示的完整字段名。

    Returns:
        校验通过的原始字符串，保留调用方填写的 shell 风格通配符。

    Raises:
        ConfigError: 值不是非空字符串，使用反斜杠、绝对路径、空段、
            ``.``/``..`` 段，或指向 ``.git``、``.mycode`` 管理目录。
    """

    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"配置项 {field_name} 必须是非空字符串")
    if "\\" in value:
        raise ConfigError(f"配置项 {field_name} 必须使用正斜杠分隔路径")
    parts = value.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise ConfigError(f"配置项 {field_name} 必须是项目内相对路径")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ConfigError(f"配置项 {field_name} 必须是项目内相对路径")
    if parts[0].lower() in {".git", ".mycode"}:
        raise ConfigError(
            f"配置项 {field_name} 不能读取 .git 或 .mycode 管理目录"
        )
    return value


def _parse_worktree_rules(
    value: object,
    *,
    field_name: str,
    value_key: str,
) -> tuple[WorktreePathRule | WorktreeIgnoredRule, ...]:
    """解析一类 Worktree 初始化规则列表。

    Args:
        value: YAML 中 ``copy_files``、``symlink_directories`` 或
            ``copy_ignored`` 的原始值。
        field_name: 错误消息中显示的完整列表字段名。
        value_key: 每项保存目标文本的键，取 ``path`` 或 ``pattern``。

    Returns:
        按配置出现顺序排列的路径规则或 ignored 规则元组。

    Raises:
        ConfigError: 列表、映射、字段、路径文本或 ``required`` 类型无效。
    """

    if value is None:
        return ()
    if not isinstance(value, list):
        raise ConfigError(f"配置项 {field_name} 必须是列表")
    allowed = (
        _WORKTREE_RULE_FIELDS
        if value_key == "path"
        else _WORKTREE_IGNORED_RULE_FIELDS
    )
    parsed: list[WorktreePathRule | WorktreeIgnoredRule] = []
    for index, item in enumerate(value):
        item_field = f"{field_name}[{index}]"
        if not isinstance(item, Mapping):
            raise ConfigError(f"配置项 {item_field} 必须是映射")
        if any(not isinstance(key, str) for key in item):
            raise ConfigError(f"配置项 {item_field} 的字段名必须是字符串")
        unknown = sorted(set(item) - allowed)
        if unknown:
            raise ConfigError(
                f"配置项 {item_field} 包含未知字段：{', '.join(unknown)}"
            )
        if value_key not in item:
            raise ConfigError(f"配置项 {item_field}.{value_key} 必填")
        text_value = _validate_worktree_rule_text(
            item[value_key],
            field_name=f"{item_field}.{value_key}",
        )
        required = item.get("required", False)
        if not isinstance(required, bool):
            raise ConfigError(f"配置项 {item_field}.required 必须是布尔值")
        if value_key == "path":
            parsed.append(WorktreePathRule(path=text_value, required=required))
        else:
            parsed.append(
                WorktreeIgnoredRule(pattern=text_value, required=required)
            )
    return tuple(parsed)


def _parse_worktree_layer(
    raw: Mapping[str, Any],
    *,
    layer_label: str,
) -> dict[str, object]:
    """读取单层 ``worktrees`` 配置并保留该层明确填写的字段。

    Args:
        raw: 当前用户级、项目级或单文件配置的根映射。
        layer_label: 错误消息中标识当前配置层的名字。

    Returns:
        已完成严格校验的字段字典。标量只在本层出现时写入，规则列表保存为
        元组，供两层配置按顺序追加。

    Raises:
        ConfigError: ``worktrees`` 结构、字段、数值、规则或 Hooks 路径无效。
    """

    value = raw.get("worktrees")
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"配置项 {layer_label}.worktrees 必须是映射")
    if any(not isinstance(key, str) for key in value):
        raise ConfigError(
            f"配置项 {layer_label}.worktrees 的字段名必须是字符串"
        )
    unknown = sorted(set(value) - _WORKTREE_FIELDS)
    if unknown:
        raise ConfigError(
            f"配置项 {layer_label}.worktrees 包含未知字段：{', '.join(unknown)}"
        )

    parsed: dict[str, object] = {}
    for field_name in ("stale_after_hours", "cleanup_interval_seconds"):
        if field_name not in value:
            continue
        number = value[field_name]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or number <= 0
        ):
            raise ConfigError(
                f"配置项 {layer_label}.worktrees.{field_name} 必须是正数"
            )
        parsed[field_name] = float(number)

    for field_name, value_key in (
        ("copy_files", "path"),
        ("symlink_directories", "path"),
        ("copy_ignored", "pattern"),
    ):
        if field_name in value:
            parsed[field_name] = _parse_worktree_rules(
                value[field_name],
                field_name=f"{layer_label}.worktrees.{field_name}",
                value_key=value_key,
            )

    if "hooks_path" in value:
        hooks_path = value["hooks_path"]
        if hooks_path is not None:
            hooks_path = _validate_worktree_rule_text(
                hooks_path,
                field_name=f"{layer_label}.worktrees.hooks_path",
            )
        parsed["hooks_path"] = hooks_path
    return parsed


def _merge_worktree_settings(
    base: Mapping[str, object],
    override: Mapping[str, object],
) -> WorktreeSettings:
    """合并用户级和项目级 Worktree 配置。

    Args:
        base: 用户级配置中明确填写的字段。
        override: 项目级配置中明确填写的字段。

    Returns:
        标量和 Hooks 路径由项目级覆盖、三类规则按用户级再项目级顺序追加的
        完整 ``WorktreeSettings``。

    Raises:
        ConfigError: 合并后的字段无法构造有效配置模型。
    """

    values: dict[str, object] = {}
    for field_name in (
        "stale_after_hours",
        "cleanup_interval_seconds",
        "hooks_path",
    ):
        if field_name in override:
            values[field_name] = override[field_name]
        elif field_name in base:
            values[field_name] = base[field_name]
    for field_name in ("copy_files", "symlink_directories", "copy_ignored"):
        values[field_name] = tuple(base.get(field_name, ())) + tuple(
            override.get(field_name, ())
        )
    try:
        return WorktreeSettings(**values)  # type: ignore[arg-type]
    except ValueError as exc:
        raise ConfigError(f"配置项 worktrees 无效：{exc}") from exc


def _build_app_config(
    *,
    active: str | None,
    providers: tuple[ProviderConfig, ...],
    mcp_servers: tuple[McpServerConfig, ...],
    hooks: tuple[HookDefinition, ...] = (),
    agents: AgentSettings | None = None,
    worktrees: WorktreeSettings | None = None,
) -> AppConfig:
    """校验合并结果并构造最终应用配置。

    Args:
        active: 合并后选择的 Provider 名。
        providers: 合并后的 Provider 元组。
        mcp_servers: 合并后的 MCP Server 元组。
        hooks: 三层配置追加合并并完成校验后的 Hook。
        agents: 合并后的独立子 Agent 运行配置；未传时使用默认值。
        worktrees: 合并后的 Worktree 初始化和清理配置；未传时使用默认值。

    Returns:
        可以交给应用装配层使用的完整配置。
    """
    if not active:
        raise ConfigError("合并后的配置项 active 必须是非空字符串")
    if not providers:
        raise ConfigError("合并后的配置项 providers 必须是非空列表")
    names = [provider.name for provider in providers]
    if active not in names:
        raise ConfigError(f"配置项 active 未匹配任何 Provider：{active}")
    return AppConfig(
        active=active,
        providers=providers,
        mcp_servers=mcp_servers,
        hooks=hooks,
        agents=agents or AgentSettings(),
        worktrees=worktrees or WorktreeSettings(),
    )


def load_config(
    path: Path | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> AppConfig:
    """只加载一份完整配置文件

    Args:
        path: 配置路径；未提供时使用默认用户级路径。
        environment: 可替换进程环境的变量映射。

    Returns:
        校验完成的应用配置。
    """
    config_path = path or default_config_path()
    raw = _read_config_layer(
        config_path,
        label="",
        required=True,
    )
    env = os.environ if environment is None else environment
    active = _parse_optional_active(raw, layer_label="config")
    providers = _parse_provider_layer(
        raw,
        env,
        layer_label="config",
    )
    mcp_servers = _parse_mcp_layer(
        raw,
        env,
        layer_label="config",
    )
    hooks = parse_hook_layers(
        ((HookLayer.PROJECT, config_path, raw),)
    )
    agents = _merge_agent_settings(
        {},
        _parse_agent_layer(raw, layer_label="config"),
    )
    worktrees = _merge_worktree_settings(
        {},
        _parse_worktree_layer(raw, layer_label="config"),
    )
    return _build_app_config(
        active=active,
        providers=providers,
        mcp_servers=mcp_servers,
        hooks=hooks,
        agents=agents,
        worktrees=worktrees,
    )


def load_startup_config(
    *,
    working_directory: Path | None = None,
    environment: Mapping[str, str] | None = None,
    user_home: Path | None = None,
) -> AppConfig:
    """加载用户级和项目级两份配置，然后合并

    Args:
        working_directory: 项目配置、相对路径和 ``.env`` 使用的工作区。
        environment: 优先于 ``.env`` 的进程或测试环境。
        user_home: 可替换真实用户主目录的测试路径。

    Returns:
        Provider 和 MCP 设置按原规则合并、Hook 按三层追加后的完整配置。
    """
    current_directory = (
        Path.cwd()
        if working_directory is None
        else working_directory.resolve(strict=False)
    )
    home = (
        Path.home()
        if user_home is None
        else user_home.resolve(strict=False)
    )
    base_environment = os.environ if environment is None else environment
    merged_environment = _load_project_environment(
        current_directory / ".env",
        base_environment,
    )
    user_path = (home / ".mycode" / "config.yaml").resolve(strict=False)
    project_path = _resolve_project_config_path(
        current_directory,
        merged_environment,
    )
    project_required = bool(merged_environment.get(MYCODE_CONFIG_ENV, "").strip())
    local_path = (
        current_directory / LOCAL_HOOK_CONFIG_RELATIVE_PATH
    ).resolve(strict=False)

    if user_path == project_path:
        user_raw: dict[str, Any] = {}
        project_raw = _read_config_layer(
            project_path,
            label="项目级",
            required=project_required,
        )

    else:
        user_raw = _read_config_layer(
            user_path,
            label="用户级",
            required=False,
        )
        project_raw = _read_config_layer(
            project_path,
            label="项目级",
            required=project_required,
        )

    local_raw = _read_config_layer(
        local_path,
        label="本地级",
        required=False,
    )
    local_unknown = sorted(set(local_raw) - {"hooks"})
    if local_unknown:
        raise ConfigError(
            f"本地级配置 {local_path} 只允许 hooks，包含未知字段："
            + ", ".join(local_unknown)
        )
    hooks = parse_hook_layers(
        (
            (HookLayer.USER, user_path, user_raw),
            (HookLayer.PROJECT, project_path, project_raw),
            (HookLayer.LOCAL, local_path, local_raw),
        )
    )

    user_active = _parse_optional_active(
        user_raw,
        layer_label="user",
    )
    project_active = _parse_optional_active(
        project_raw,
        layer_label="project",
    )
    user_providers = _parse_provider_layer(
        user_raw,
        merged_environment,
        layer_label="user",
    )
    project_providers = _parse_provider_layer(
        project_raw,
        merged_environment,
        layer_label="project",
    )
    user_servers = _parse_mcp_layer(
        user_raw,
        merged_environment,
        layer_label="user",
    )
    project_servers = _parse_mcp_layer(
        project_raw,
        merged_environment,
        layer_label="project",
    )
    user_agents = _parse_agent_layer(
        user_raw,
        layer_label="user",
    )
    project_agents = _parse_agent_layer(
        project_raw,
        layer_label="project",
    )
    user_worktrees = _parse_worktree_layer(
        user_raw,
        layer_label="user",
    )
    project_worktrees = _parse_worktree_layer(
        project_raw,
        layer_label="project",
    )
    return _build_app_config(
        active=project_active or user_active,
        providers=_merge_named(user_providers, project_providers),
        mcp_servers=_merge_named(user_servers, project_servers),
        hooks=hooks,
        agents=_merge_agent_settings(user_agents, project_agents),
        worktrees=_merge_worktree_settings(user_worktrees, project_worktrees),
    )
