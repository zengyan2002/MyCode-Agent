"""应用异常类型与面向用户输出的敏感信息脱敏。"""

from __future__ import annotations

import re
from collections.abc import Iterable

from mycode.models.config import SecretValue


class MyCodeError(Exception):
    """经过脱敏后可安全展示给用户的异常基类。"""


class ConfigError(MyCodeError):
    """配置无法加载或未通过校验。"""


class TransportError(MyCodeError):
    """无法连接或读取远端服务。"""


class AuthenticationError(TransportError):
    """远端服务拒绝了身份认证。"""


class ServiceError(TransportError):
    """远端服务返回错误。"""


class HttpServiceError(ServiceError):
    """保存 Provider 返回的 HTTP 状态码和已经脱敏的错误说明。"""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.safe_message = message
        super().__init__(message)


class HttpStatusError(ServiceError):
    """远端返回非认证类 HTTP 错误，并保留受限响应体供协议适配器判断。"""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        detail = f"：{body}" if body else ""
        super().__init__(f"服务请求失败（HTTP {status_code}）{detail}")


class ContextWindowExceededError(ServiceError):
    """Provider 明确拒绝了超过模型上下文窗口的请求。"""


class ArtifactError(MyCodeError):
    """当前会话的工具正文或用户原文无法安全写入、提交或清理。"""


class StreamProtocolError(MyCodeError):
    """流式响应不符合当前选择的协议。"""


class ConcurrentTurnError(MyCodeError):
    """同一个 Agent 循环仍在运行时又启动了第二个回合。"""


_AUTH_PATTERNS = (
    # 即使某个密钥没有进入 SecretValue 列表，也尽量清理最常见的两种
    # 认证头格式。这里只处理展示文本，不会修改真实网络请求头。
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)([^\s,;]+)"),
    re.compile(r"(?i)(x-api-key\s*:\s*)([^\s,;]+)"),
)


def redact_secrets(
    message: str,
    secrets: Iterable[SecretValue] = (),
) -> str:
    """移除已加载密钥和常见认证请求头中的敏感值。"""

    # 先替换已知密钥，再处理认证头。前者覆盖服务端在任意上下文中回显
    # 密钥的情况，后者为未知或未登记的头值兜底。
    redacted = str(message)
    for secret in secrets:
        value = secret.reveal()
        if value:
            redacted = redacted.replace(value, "***")
    for pattern in _AUTH_PATTERNS:
        redacted = pattern.sub(r"\1***", redacted)
    return redacted
