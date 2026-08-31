"""把三层 YAML 中的 Hook 转换成可直接执行的规则。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from mycode.errors import ConfigError
from mycode.hooks.conditions import parse_condition
from mycode.hooks.templates import parse_template
from mycode.models.hooks import (
    AgentHookAction,
    CommandHookAction,
    HookAction,
    HookDefinition,
    HookEvent,
    HookLayer,
    HookSource,
    HttpHookAction,
    PromptHookAction,
    HookTemplate,
)


_HOOK_FIELDS = {"id", "event", "if", "action", "once", "async", "reject"}
_ACTION_FIELDS = {
    "command": {"type", "command", "timeout"},
    "prompt": {"type", "message"},
    "http": {"type", "url", "method", "headers", "body"},
    "agent": {"type", "prompt"},
}


def _mapping(value: object, *, source: str) -> Mapping[str, object]:
    """要求一个 YAML 值是字符串键映射，并返回给后续字段解析。

    Args:
        value: YAML 读取器返回的待检查值。
        source: 该值在配置中的位置，用于错误提示。

    Returns:
        保留原声明顺序、键均为字符串的映射。
    """

    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise ConfigError(f"{source} 必须是字段名为字符串的映射")
    return value


def _check_fields(
    raw: Mapping[str, object],
    allowed: set[str],
    *,
    source: str,
) -> None:
    """拒绝配置对象中当前版本不认识的字段。

    Args:
        raw: 当前 Hook 或 action 的字段映射。
        allowed: 该对象类型允许出现的全部字段名。
        source: 当前对象在配置中的位置，用于错误提示。

    Returns:
        无返回值；发现未知字段时直接抛出 ConfigError。
    """

    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"{source} 包含未知字段：{', '.join(unknown)}")


def _boolean(raw: Mapping[str, object], name: str, *, source: str) -> bool:
    """读取一个默认关闭的 Hook 布尔控制项。

    Args:
        raw: 当前 Hook 的完整字段映射。
        name: 要读取的 ``once``、``async`` 或 ``reject`` 字段名。
        source: 当前 Hook 的配置位置，用于错误提示。

    Returns:
        字段不存在时返回 False；存在且类型正确时返回其布尔值。
    """

    value = raw.get(name, False)
    if not isinstance(value, bool):
        raise ConfigError(f"{source} 的 {name} 必须是布尔值")
    return value


def _timeout(value: object, *, source: str) -> float:
    """把正数或 ``10s`` 形式的命令超时转换为秒数。

    Args:
        value: YAML 中的数字或以 ``s`` 结尾的字符串。
        source: 当前 Hook 的配置位置，用于错误提示。

    Returns:
        大于零的浮点秒数。
    """

    raw = value[:-1] if isinstance(value, str) and value.endswith("s") else value
    if isinstance(raw, bool):
        raise ConfigError(f"{source} 的 timeout 必须是正数秒")
    try:
        seconds = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{source} 的 timeout 必须是正数秒或 10s 形式") from exc
    if seconds <= 0:
        raise ConfigError(f"{source} 的 timeout 必须大于 0")
    return seconds


def _parse_action(value: object, *, source: str) -> HookAction:
    """校验动作专属字段并返回相应的不可变动作对象。

    Args:
        value: YAML ``action`` 字段读取到的映射。
        source: 当前 Hook 的配置位置，用于错误提示。

    Returns:
        可直接交给 HookActionRunner 的 command、prompt、http 或 agent 动作。
    """

    raw = _mapping(value, source=f"{source} 的 action")
    action_type = raw.get("type")
    if not isinstance(action_type, str) or action_type not in _ACTION_FIELDS:
        raise ConfigError(
            f"{source} 的 action.type 只支持 command、prompt、http、agent"
        )
    _check_fields(raw, _ACTION_FIELDS[action_type], source=f"{source} 的 action")
    if action_type == "command":
        if "command" not in raw:
            raise ConfigError(f"{source} 的 command 动作缺少 command")
        timeout = (
            _timeout(raw["timeout"], source=source)
            if "timeout" in raw
            else None
        )
        return CommandHookAction(
            parse_template(raw["command"], source=f"{source} 的 command"),
            timeout,
        )
    if action_type == "prompt":
        if "message" not in raw:
            raise ConfigError(f"{source} 的 prompt 动作缺少 message")
        return PromptHookAction(
            parse_template(raw["message"], source=f"{source} 的 message")
        )
    if action_type == "agent":
        if "prompt" not in raw:
            raise ConfigError(f"{source} 的 agent 动作缺少 prompt")
        return AgentHookAction(
            parse_template(raw["prompt"], source=f"{source} 的 prompt")
        )

    if "url" not in raw:
        raise ConfigError(f"{source} 的 http 动作缺少 url")
    method = raw.get("method", "POST")
    if not isinstance(method, str) or not method.strip():
        raise ConfigError(f"{source} 的 http method 必须是非空字符串")
    raw_headers = raw.get("headers", {})
    headers = _mapping(raw_headers, source=f"{source} 的 headers")
    parsed_headers: list[tuple[str, HookTemplate]] = []
    for name, header_value in headers.items():
        if not name.strip():
            raise ConfigError(f"{source} 的请求头名称不能为空")
        parsed_headers.append(
            (
                name,
                parse_template(
                    header_value,
                    source=f"{source} 的请求头 {name}",
                ),
            )
        )
    body = (
        parse_template(raw["body"], source=f"{source} 的 body")
        if "body" in raw
        else None
    )
    return HttpHookAction(
        url=parse_template(raw["url"], source=f"{source} 的 url"),
        method=method.upper(),
        headers=tuple(parsed_headers),
        body=body,
    )


def _parse_hook(
    value: object,
    *,
    layer: HookLayer,
    path: Path,
    index: int,
) -> HookDefinition:
    """解析列表中的一条 Hook，并把来源加入错误和运行记录。

    Args:
        value: ``hooks`` 列表中的一个原始 YAML 对象。
        layer: 该规则来自用户、项目或本地配置层。
        path: 声明规则的实际配置文件路径。
        index: 规则在文件 ``hooks`` 列表中的零基位置。

    Returns:
        事件、条件、动作和执行控制均已校验的 Hook 定义。
    """

    position = f"#{index + 1}"
    preliminary = f"{path} 的 Hook {position}"
    raw = _mapping(value, source=preliminary)
    _check_fields(raw, _HOOK_FIELDS, source=preliminary)
    raw_id = raw.get("id", position)
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise ConfigError(f"{preliminary} 的 id 必须是非空字符串")
    source = HookSource(layer, path, index, raw_id.strip())
    label = source.label
    if "event" not in raw:
        raise ConfigError(f"{label} 缺少 event")
    if "action" not in raw:
        raise ConfigError(f"{label} 缺少 action")
    try:
        event = HookEvent(raw["event"])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} 包含未知事件：{raw['event']!r}") from exc
    once = _boolean(raw, "once", source=label)
    async_mode = _boolean(raw, "async", source=label)
    reject = _boolean(raw, "reject", source=label)
    if reject and event is not HookEvent.PRE_TOOL_USE:
        raise ConfigError(f"{label} 的 reject 只能用于 pre_tool_use")
    if async_mode and event is HookEvent.PRE_TOOL_USE:
        raise ConfigError(f"{label} 的 pre_tool_use 不能异步执行")
    condition = (
        parse_condition(raw["if"], source=label)
        if "if" in raw
        else None
    )
    return HookDefinition(
        source=source,
        event=event,
        condition=condition,
        action=_parse_action(raw["action"], source=label),
        once=once,
        async_mode=async_mode,
        reject=reject,
    )


def parse_hook_layer(
    raw: Mapping[str, object],
    *,
    layer: HookLayer,
    path: Path,
) -> tuple[HookDefinition, ...]:
    """解析一份已经由设置加载器读取的 YAML 配置。

    Args:
        raw: 该配置文件的顶层映射。
        layer: 文件属于用户、项目还是本地层。
        path: 实际配置路径，写入规则来源和错误信息。

    Returns:
        按文件声明顺序排列的已校验 Hook；没有 `hooks` 时返回空元组。
    """

    values = raw.get("hooks", ())
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ConfigError(f"{path} 的 hooks 必须是列表")
    return tuple(
        _parse_hook(value, layer=layer, path=path, index=index)
        for index, value in enumerate(values)
    )


def parse_hook_layers(
    layers: Sequence[tuple[HookLayer, Path, Mapping[str, object]]],
) -> tuple[HookDefinition, ...]:
    """按用户、项目、本地的传入顺序追加全部 Hook。

    Args:
        layers: 每项包含层级、真实文件路径和顶层 YAML 映射。

    Returns:
        可以原样交给 `HookEngine` 的规则元组。任何一层失败都不返回部分结果。
    """

    parsed: list[HookDefinition] = []
    for layer, path, raw in layers:
        parsed.extend(parse_hook_layer(raw, layer=layer, path=path))
    return tuple(parsed)
