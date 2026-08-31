"""三层权限 YAML 的加载、校验和本地永久授权写入。"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from mycode.errors import ConfigError
from mycode.models.permissions import (
    LoadedPermissionSettings,
    PermissionLayer,
    PermissionMode,
    PermissionScope,
    PermissionOperation,
)
from mycode.permissions.rules import (
    format_exact_rule_expression,
    parse_rule,
)

# 权限 YAML 根映射允许出现的字段，用于拒绝拼写错误或尚未支持的配置项。
_TOP_LEVEL_FIELDS = {"mode", "rules"}


class _UniqueKeyLoader(yaml.SafeLoader):
    """安全加载 YAML，并把重复映射键视为配置错误。"""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    """
    在 YAML 还保留完整键值节点时逐项构造字典，发现重复键就立即报错，避免权限规则被后面的同名配置覆盖

    Parameters:
        loader: yaml解析器
        node: PyYAML 解析 YAML 后产生的映射语法节点
            例如 YAML：
                mode: default
                rules: {}
            在还没有转换成字典前，node.value 大致保存：
                [
                    (mode节点, default节点),
                    (rules节点, 空映射节点),
                ]
        deep: 控制构造嵌套对象时是否进行深层构造
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

# 给 _UniqueKeyLoader 注册规则：以后遇到任何普通 YAML 字典节点时，都使用 _construct_unique_mapping() 将它转换成 Python 字典
_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _read_yaml_mapping(path: Path, scope: PermissionScope) -> dict[str, Any]:
    """
    从指定路径读取 UTF-8 权限 YAML 文件，使用支持重复键检查的 Loader
    完成解析，并校验根节点必须是字符串键组成的映射且不包含未知顶层字段。
    文件不存在或内容为空时返回空字典，校验通过后返回供后续权限层解析使用
    的普通字典。

    Parameters:
        path:当前权限层的配置文件路径
        scope: 当前权限层

    Returns:
        dict[str, Any]: 权限配置相关字典
    """
    label = f"{scope.value} 权限配置 {path}"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError) as exc:
        raise ConfigError(f"无法读取 {label}：{type(exc).__name__}") from exc
    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader)
    except ConfigError as exc:
        raise ConfigError(f"{label}：{exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{label} 的 YAML 格式错误：{exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{label} 顶层必须是映射")
    if any(not isinstance(key, str) for key in raw):
        raise ConfigError(f"{label} 顶层键必须是字符串")
    unknown = sorted(set(raw) - _TOP_LEVEL_FIELDS)
    if unknown:
        raise ConfigError(f"{label} 包含未知字段：{', '.join(unknown)}")
    return dict(raw)


def _parse_layer(path: Path, scope: PermissionScope) -> PermissionLayer:
    """
    读取并解析一层权限配置，生成经过校验的 PermissionLayer。

    Parameters:
        path: 当前权限层对应的 YAML 配置文件路径。
        scope: 当前配置所属的权限层，例如 USER、PROJECT 或 LOCAL。

    Returns:
        包含可选权限模式、已编译规则和来源路径的 PermissionLayer。
    """
    raw = _read_yaml_mapping(path, scope)
    raw_mode = raw.get("mode")

    # mode 不存在或者显式为 null，表示当前权限层不覆盖整体权限模式。
    # 例如：没有配置文件、用户选择永久允许只会追加rules:...，不会添加mode:...
    if raw_mode is None:
        mode = None
    else:
        try:
            mode = PermissionMode(raw_mode)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"{scope.value} 权限配置 {path} 的 mode "
                "只支持 strict、default 或 allow"
            ) from exc

    raw_rules = raw.get("rules", {})
    if not isinstance(raw_rules, Mapping):
        raise ConfigError(
            f"{scope.value} 权限配置 {path} 的 rules 必须是映射"
        )
    rules = []
    for expression, effect in raw_rules.items():
        if not isinstance(expression, str) or not isinstance(effect, str):
            raise ConfigError(
                f"{scope.value} 权限配置 {path} 的规则必须是字符串到 "
                "allow/deny 的映射"
            )
        rules.append(
            parse_rule(
                expression,
                effect,
                scope=scope,
                source=f"{scope.value} 权限配置 {path}",
            )
        )
    return PermissionLayer(scope, mode, tuple(rules), path)


def load_permission_settings(
    workspace_root: Path,
    *,
    user_home: Path | None = None,
) -> LoadedPermissionSettings:
    """
    加载用户级、项目级和本地级权限配置，并确定会话启动模式。

    工作区必须真实存在且为目录。函数根据工作区和用户主目录计算三份
    权限配置路径，分别解析成 PermissionLayer，然后按照
    LOCAL、PROJECT、USER 的优先级选择第一个明确配置的权限模式；
    三层都未配置 mode 时使用 DEFAULT。

    Parameters:
        workspace_root: 当前项目工作区根目录。
        user_home: 可选的用户主目录。未提供时使用 Path.home()；
            该参数主要用于测试时替换真实用户目录。

    Returns:
        包含初始权限模式、USER/PROJECT/LOCAL 三层配置及本地权限
        写入路径的 LoadedPermissionSettings。
    """
    # 将工作区解析为真实绝对路径，并要求该路径必须已经存在
    root = workspace_root.resolve(strict=True)

    # 权限配置必须绑定到一个项目目录，不能绑定到普通文件
    if not root.is_dir():
        raise ConfigError(f"权限工作区不是目录：{root}")

    # 当前系统用户的主目录
    home = Path.home() if user_home is None else user_home.resolve(strict=False)

    #设置不同层级的权限的yaml文件路径
    user_path = home / ".mycode" / "permissions.yaml"
    project_path = root / ".mycode" / "permissions.yaml"
    local_path = root / ".mycode" / "permissions.local.yaml"

    # 生成不同的层级的权限配置
    user = _parse_layer(user_path, PermissionScope.USER)
    project = _parse_layer(project_path, PermissionScope.PROJECT)
    local = _parse_layer(local_path, PermissionScope.LOCAL)

    # 按 LOCAL、PROJECT、USER 的优先级寻找第一个明确设置的 mode，要是都没有就设置mode为PermissionMode.DEFAULT
    initial_mode = next(
        (
            layer.mode
            for layer in (local, project, user)
            if layer.mode is not None
        ),
        PermissionMode.DEFAULT,
    )
    # 组装成权限配置加载完成后的不可变快照返回出去
    return LoadedPermissionSettings(
        initial_mode=initial_mode,
        user=user,
        project=project,
        local=local,
        local_path=local_path,
    )


class LocalPermissionStore:
    """
    把用户选择的永久允许的内容作为规则写入<项目>/.mycode/permissions.local.yaml
    """
    def __init__(self, local_path: Path) -> None:
        self._local_path = local_path.resolve(strict=False)

    def allow_permanently(
        self,
        operation: PermissionOperation,
    ) -> PermissionLayer:
        """
        将当前操作作为精确允许规则，持久化到项目本地权限配置。

        每次都从磁盘重新读取最新的 LOCAL 配置，在保留现有模式和
        规则的基础上追加当前操作的精确 allow 规则。合并后的配置先写入
        同目录临时文件，并使用正式权限加载流程完成业务校验；校验通过后，
        再通过 os.replace() 原子替换正式文件。

        保存成功后重新解析正式文件并返回最新 PermissionLayer，使调用方
        可以立即替换内存中的 LOCAL 权限快照，无需重启程序。

        Parameters:
            operation: 用户选择永久允许的具体权限操作。保存时只授权相同
                权限工具和相同 match_value，不自动扩大为通配授权。

        Returns:
            从正式本地权限文件重新解析得到的最新 LOCAL PermissionLayer。
            """
        # 每次永久授权都会重新读取磁盘上的最新内容，而不是只修改程序启动时加载的旧快照
        raw = _read_yaml_mapping(self._local_path, PermissionScope.LOCAL)
        # 如果 raw 已有 rules，返回已有值，如果没有，则添加 rules: {} 并返回新字典
        raw_rules = raw.setdefault("rules", {})
        if not isinstance(raw_rules, Mapping):
            raise ConfigError(
                f"local 权限配置 {self._local_path} 的 rules 必须是映射"
            )
        # 复制原有规则
        rules = dict(raw_rules)
        # 生成精确允许规则
        rules[format_exact_rule_expression(operation)] = "allow"
        # 把合并结果放回顶层配置
        raw["rules"] = rules

        # 确保父目录存在，没有就创建
        parent = self._local_path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(
                f"无法创建本地权限配置目录 {parent}：{type(exc).__name__}"
            ) from exc

        # 记录临时文件路径
        temporary_path: Path | None = None
        try:
            # 创建同目录的临时文件
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{self._local_path.name}.",
                suffix=".tmp",
                dir=parent,
                delete=False,
            ) as handle:
                # 保持临时文件的路径
                temporary_path = Path(handle.name)
                # 序列化yaml
                yaml.safe_dump(
                    raw,
                    handle,
                    allow_unicode=True,
                    sort_keys=False,
                )
                # 刷新缓冲区
                handle.flush()
                # 要求操作系统刷新文件
                os.fsync(handle.fileno())
            # 使用正式解析流程校验临时文件
            parsed = _parse_layer(temporary_path, PermissionScope.LOCAL)
            # 用临时文件替换原始文件
            os.replace(temporary_path, self._local_path)
            temporary_path = None
        except ConfigError:
            raise
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise ConfigError(
                f"无法更新本地权限配置 {self._local_path}："
                f"{type(exc).__name__}"
            ) from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

        # 临时文件的 source_path 不是最终来源，替换成功后重新读取正式文件。
        del parsed
        return _parse_layer(self._local_path, PermissionScope.LOCAL)
