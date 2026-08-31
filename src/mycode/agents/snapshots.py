"""记录主 Agent 实际发送的 Provider 请求和对应完整响应。"""

from __future__ import annotations

from dataclasses import dataclass

from mycode.models.messages import AssistantMessage
from mycode.models.provider import ProviderRequest
from mycode.models.tools import ToolView


@dataclass(frozen=True)
class ParentRunSnapshot:
    """Fork 子 Agent 创建时复制的父请求前缀。

    Attributes:
        request: 主 Agent 实际发给 Provider 的请求，包括已经渲染完成的
            PromptContext、消息和工具定义。
        tool_view: 与该请求工具定义完全对应的本地可见名快照。
        response: Provider 对该请求返回的完整 AssistantMessage。
    """

    request: ProviderRequest
    tool_view: ToolView
    response: AssistantMessage


class ParentRunRecorder:
    """在一个主 Agent 回合内保存最近一次完整父请求和响应。

    AgentTurnRunner 在发送请求前记录请求，收到 ProviderCompleted 后补上
    响应。AgentTool 随后在同一批工具执行期间读取快照。该类不复制或持有
    SessionManager 等可变会话对象。

    Attributes:
        _request: 主 Agent 最近一次真正发给 Provider 的完整请求。
        _tool_view: 与该请求 definitions 对应的本地工具可见范围。
        _response: Provider 对该请求返回的完整助手消息。
    """

    def __init__(self) -> None:
        """创建尚未记录任何 Provider 请求的 Recorder。

        Returns:
            不返回数据；三个字段初始均为 ``None``。
        """

        self._request: ProviderRequest | None = None
        self._tool_view: ToolView | None = None
        self._response: AssistantMessage | None = None

    def record_request(
        self,
        request: ProviderRequest,
        tool_view: ToolView,
    ) -> None:
        """保存即将发送的真实请求和同一批工具视图。

        Args:
            request: AgentTurnRunner 已经完成上下文压缩和提示拼装的请求。
            tool_view: ToolRegistry 为该请求解析出的最终可见工具名。

        Returns:
            不返回数据；旧请求和旧响应会被本次请求替换。
        """

        self._request = request
        self._tool_view = tool_view
        self._response = None

    def record_response(self, response: AssistantMessage) -> None:
        """为最近记录的 Provider 请求补上完整助手响应。

        Args:
            response: ProviderCompleted 中包含文字、thinking 和工具调用的
                完整消息。

        Returns:
            不返回数据；之后 :meth:`snapshot` 可以产生 Fork 输入。

        Raises:
            RuntimeError: 尚未记录请求就收到响应。
        """

        if self._request is None or self._tool_view is None:
            raise RuntimeError("尚未记录父 Agent 请求，不能记录响应")
        self._response = response

    def snapshot(self) -> ParentRunSnapshot:
        """返回最近一次完整请求与响应的不可变快照。

        Returns:
            Fork 构造消息和工具交集所需的 ParentRunSnapshot。

        Raises:
            RuntimeError: 当前没有成对的请求和响应。
        """

        if (
            self._request is None
            or self._tool_view is None
            or self._response is None
        ):
            raise RuntimeError("当前没有可供 Fork 使用的完整父运行快照")
        return ParentRunSnapshot(
            request=self._request,
            tool_view=self._tool_view,
            response=self._response,
        )

    def clear(self) -> None:
        """清除回合内保存的父请求和响应引用。

        Returns:
            不返回数据；回合结束或取消时调用。
        """

        self._request = None
        self._tool_view = None
        self._response = None
