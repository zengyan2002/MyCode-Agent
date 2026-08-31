"""记录一次 Agent 运行还可以请求模型多少次，以及每次请求的实际用途。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mycode.models.provider import ProviderUsage


class ModelCallPurpose(str, Enum):
    """说明一次真实 Provider 请求用于 Agent 决策还是上下文压缩。"""

    AGENT = "agent"
    COMPACTION = "compaction"


@dataclass(frozen=True)
class ModelCallRecord:
    """保存一次已经发给 Provider 的请求序号、用途和 Token 统计。

    Attributes:
        model_call_number: 该请求在当前 Agent 运行中的序号，从 1 开始。
        purpose: 请求用于 Agent 决策或上下文压缩。
        usage: Provider 返回的 Token 统计；请求失败或服务未报告时为 ``None``。
    """

    model_call_number: int
    purpose: ModelCallPurpose
    usage: ProviderUsage | None

    def __post_init__(self) -> None:
        """拒绝无效序号、用途和用量类型。

        Returns:
            字段可以用于用量事件时不返回数据。

        Raises:
            ValueError: 序号不是正整数，或用途、用量类型不正确。
        """

        if (
            isinstance(self.model_call_number, bool)
            or not isinstance(self.model_call_number, int)
            or self.model_call_number <= 0
        ):
            raise ValueError("模型调用序号必须是正整数")
        if not isinstance(self.purpose, ModelCallPurpose):
            raise ValueError("模型调用用途无效")
        if self.usage is not None and not isinstance(self.usage, ProviderUsage):
            raise ValueError("模型调用用量必须是 ProviderUsage 或 None")


class ModelCallBudget:
    """管理一条主 Agent 消息或一次子 Agent 任务可以发出的 Provider 请求。

    ``AgentTurnRunner`` 在真正发送请求前调用 :meth:`begin`，请求结束或失败
    后调用 :meth:`finish`。上下文压缩与普通 Agent 请求共享同一个实例，因而
    二者不会分别突破总上限。

    Attributes:
        _max_model_calls: 本次运行允许发给 Provider 的请求总数。
        _started: 已经取得序号、因此已经计入预算的请求及其用途。
        _finished: 已经形成调用记录的请求序号。
    """

    def __init__(self, max_model_calls: int) -> None:
        """创建一次运行独享的模型调用预算。

        Args:
            max_model_calls: 当前运行最多可以真实请求 Provider 的次数。

        Returns:
            不返回数据；新预算从零次已用额度开始。

        Raises:
            ValueError: 上限不是正整数。
        """

        if (
            isinstance(max_model_calls, bool)
            or not isinstance(max_model_calls, int)
            or max_model_calls <= 0
        ):
            raise ValueError("最大模型调用次数必须是正整数")
        self._max_model_calls = max_model_calls
        self._started: dict[int, ModelCallPurpose] = {}
        self._finished: set[int] = set()

    @property
    def max_model_calls(self) -> int:
        """返回本次运行允许的 Provider 请求总数。"""

        return self._max_model_calls

    @property
    def used_model_calls(self) -> int:
        """返回已经取得序号、会计入总上限的 Provider 请求数。"""

        return len(self._started)

    @property
    def remaining_model_calls(self) -> int:
        """返回当前运行尚可发给 Provider 的请求数。"""

        return self._max_model_calls - self.used_model_calls

    @property
    def finalization_required(self) -> bool:
        """判断下一次请求是否已经是必须无工具回答的最后一次。"""

        return self.remaining_model_calls == 1

    def begin(
        self,
        purpose: ModelCallPurpose,
        *,
        preserve_final_call: bool = False,
    ) -> int:
        """在发送 Provider 请求前消费一次额度并返回本次请求序号。

        Args:
            purpose: 这次请求用于 Agent 决策还是上下文压缩。
            preserve_final_call: ``True`` 时不允许消费最后一次额度；上下文
                压缩使用它为正式回答保留一次请求。

        Returns:
            当前运行中从 1 开始的模型调用序号。

        Raises:
            ValueError: ``purpose`` 不是有效用途。
            RuntimeError: 额度已经耗尽，或调用方要求保留最后一次额度。
        """

        if not isinstance(purpose, ModelCallPurpose):
            raise ValueError("模型调用用途无效")
        remaining = self.remaining_model_calls
        if remaining <= 0:
            raise RuntimeError("已达到最大模型调用次数")
        if preserve_final_call and remaining == 1:
            raise RuntimeError("最后一次模型调用已为正式回答保留")
        number = self.used_model_calls + 1
        self._started[number] = purpose
        return number

    def finish(
        self,
        model_call_number: int,
        usage: ProviderUsage | None,
    ) -> ModelCallRecord:
        """结束一次已开始的请求并返回可发送给用量消费者的记录。

        Args:
            model_call_number: :meth:`begin` 返回的当前运行请求序号。
            usage: Provider 返回的 Token 统计；失败或未报告时为 ``None``。

        Returns:
            保存同一序号、原用途和实际用量的 ``ModelCallRecord``。

        Raises:
            ValueError: 序号未开始、已经结束，或用量类型无效。
        """

        if model_call_number not in self._started:
            raise ValueError("不能结束尚未开始的模型调用")
        if model_call_number in self._finished:
            raise ValueError("同一次模型调用不能重复结束")
        if usage is not None and not isinstance(usage, ProviderUsage):
            raise ValueError("模型调用用量必须是 ProviderUsage 或 None")
        self._finished.add(model_call_number)
        return ModelCallRecord(
            model_call_number,
            self._started[model_call_number],
            usage,
        )
