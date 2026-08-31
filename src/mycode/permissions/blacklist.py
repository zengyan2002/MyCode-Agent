"""不可由配置或人工授权放开的危险命令黑名单。"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DangerousCommandMatch:
    """
    对外返回模型返回的命令和黑名单命令的匹配结果
    """
    # 规则的标识
    rule_id: str
    # 面向用户的拒绝原因
    reason: str


@dataclass(frozen=True)
class _DangerousCommandRule:
    """
    黑名单中的一条规则，把规则信息和已经编译好的正则放在一起
    """
    rule_id: str
    reason: str
    pattern: re.Pattern[str]


def _compile(pattern: str) -> re.Pattern[str]:
    """
    返回响应的正则表达式（忽略大小写，并且忽略换行）
    """
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


_RULES = (
    # 可拦截rm -rf /   rm -fr /  RM -RF /  echo before; rm -rf /  sudo rm -rf /
    _DangerousCommandRule(
        "remove-root",
        "禁止强制递归删除根目录",
        _compile(
            r"(?:^|[;&|]\s*|\bsudo\s+)\brm\s+"
            r"-(?=[A-Za-z]*r)(?=[A-Za-z]*f)[A-Za-z]+\s+/"
            r"(?:\s*(?:$|[;&|]))"
        ),
    ),
    _DangerousCommandRule(
        "format-device",
        "禁止格式化磁盘设备",
        _compile(r"\bmkfs(?:\.[A-Za-z0-9_+-]+)?\s+/dev/"),
    ),
    #拦截 Linux、WSL 和类 Unix 环境中，使用 dd 直接写入 /dev/ 裸设备的命令
    _DangerousCommandRule(
        "copy-to-device",
        "禁止直接写入磁盘设备",
        _compile(r"\bdd\b.*?\bof\s*=\s*/dev/"),
    ),
    # 禁止开放根目录权限
    _DangerousCommandRule(
        "chmod-root",
        "禁止递归开放根目录权限",
        _compile(r"\bchmod\s+-R\s+777\s+/(?:\s|$|[;&|])"),
    ),
    # 防止fork炸弹 进程数量会快速指数增长，最终耗尽系统进程、CPU或内存资源。
    _DangerousCommandRule(
        "fork-bomb",
        "禁止执行 fork bomb",
        _compile(
            r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"
        ),
    ),
    # 禁止下载远程内容后，直接交给 Shell 执行
    _DangerousCommandRule(
        "download-to-shell",
        "禁止把远程下载内容直接交给 shell 执行",
        _compile(r"\b(?:curl|wget)\b.*?\|\s*(?:ba)?sh\b"),
    ),
    # 禁止重定向覆盖磁盘设备
    _DangerousCommandRule(
        "redirect-to-device",
        "禁止通过重定向覆盖磁盘设备",
        _compile(r">\s*/dev/sd[A-Za-z0-9]*\b"),
    ),
    # 禁止 Windows 原生命令清空、格式化或重新分区磁盘。
    _DangerousCommandRule(
        "windows-disk-management",
        "禁止使用 Windows 磁盘管理命令破坏磁盘数据",
        _compile(
            r"(?:"
            r"\bclear-disk\b|"
            r"\bformat-volume\b|"
            r"(?:^|[&|]\s*)\bformat(?:\.com)?\s+[A-Za-z]:|"
            r"\bdiskpart(?:\.exe)?\b"
            r")"
        ),
    ),
    # Windows 原生裸磁盘通常使用 \\.\PhysicalDriveN。
    _DangerousCommandRule(
        "windows-raw-device-write",
        "禁止直接写入 Windows 裸磁盘设备",
        _compile(
            r"(?:"
            r"\bdd\b.*?\bof\s*=\s*[\"']?\\\\\.\\physicaldrive\d+\b|"
            r">\s*[\"']?\\\\\.\\physicaldrive\d+\b"
            r")"
        ),
    ),
    # 禁止把 PowerShell 下载结果直接交给 Invoke-Expression。
    _DangerousCommandRule(
        "download-to-powershell",
        "禁止把远程下载内容直接交给 PowerShell 执行",
        _compile(
            r"(?:"
            r"\b(?:iwr|irm|invoke-webrequest|invoke-restmethod)\b"
            r".*?\|\s*(?:iex|invoke-expression)\b|"
            r"\b(?:iex|invoke-expression)\b"
            r".*?\b(?:downloadstring|iwr|irm|invoke-webrequest|"
            r"invoke-restmethod)\b"
            r")"
        ),
    ),
    # 拦截最明确的 PowerShell 无限创建进程循环。
    _DangerousCommandRule(
        "powershell-process-bomb",
        "禁止执行 PowerShell 进程炸弹",
        _compile(
            r"\bwhile\s*\(\s*\$true\s*\)\s*\{"
            r"[^}]*\bstart-process\b[^}]*\}"
        ),
    ),
)


def match_dangerous_command(command: str) -> DangerousCommandMatch | None:
    """
    按 _RULES 中的顺序逐条检查
    """
    for rule in _RULES:
        if rule.pattern.search(command) is not None:
            return DangerousCommandMatch(rule.rule_id, rule.reason)
    return None
