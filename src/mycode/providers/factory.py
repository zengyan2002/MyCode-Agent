"""按照配置协议构建 Provider。"""

from __future__ import annotations

from mycode.models.config import Protocol, ProviderConfig
from mycode.providers.anthropic import AnthropicProvider
from mycode.providers.base import Provider
from mycode.providers.openai import OpenAIProvider
from mycode.providers.transport import HttpTransport


def create_provider(
    config: ProviderConfig,
    transport: HttpTransport,
) -> Provider:
    # 协议选择集中在装配阶段，避免 AgentLoop 中出现 OpenAI/Anthropic
    # 分支；新增协议时也只需扩展配置枚举、适配器和这里的映射。
    if config.protocol is Protocol.ANTHROPIC:
        return AnthropicProvider(config, transport)
    if config.protocol is Protocol.OPENAI:
        return OpenAIProvider(config, transport)
    raise ValueError(f"不支持的 Provider 协议：{config.protocol}")
