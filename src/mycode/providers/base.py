"""仅使用协议中立领域模型表达的 Provider 契约。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from mycode.models.provider import ProviderEvent, ProviderRequest


class Provider(Protocol):
    # Provider 隔离 AgentLoop 与 OpenAI、Anthropic 等具体协议：
    # 它把统一请求转换成对应服务的 JSON，再把流式响应转换成统一事件。
    # 每次响应必须且只能以一个 ProviderCompleted 事件结束。
    def stream(
        self,
        request: ProviderRequest,
    ) -> AsyncIterator[ProviderEvent]:
        """发送对话历史，并产出归一化后的流式事件。"""
