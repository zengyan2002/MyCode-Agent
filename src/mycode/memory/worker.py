"""在后台按顺序处理已完成的对话，并从中提取长期记忆
本模块负责管理待处理的对话队列、调用模型生成记忆修改、执行一次有条件的
重试、写入笔记文件并记录处理状态。后台处理不会阻止用户开始下一轮对话
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

from mycode.agent.cancellation import CancellationToken
from mycode.constants import MEMORY_SHUTDOWN_TIMEOUT_SECONDS
from mycode.errors import (
    AuthenticationError,
    ContextWindowExceededError,
    HttpServiceError,
    ServiceError,
    TransportError,
    redact_secrets,
)
from mycode.memory.extraction import (
    MemoryExtractionCodec,
    MemoryResponseFormatError,
)
from mycode.models.memory import (
    CompletedTurn,
    MemoryWorkerStatus,
    MemoryWorkerStatusKind,
)
from mycode.memory.store import MemoryStore
from mycode.models.config import SecretValue
from mycode.providers.runner import ProviderRequestRunner

_STOP = object()


def _is_retryable_request_error(error: Exception) -> bool:
    """判断记忆提取的首次请求是否遇到了允许重试的网络或 HTTP 错误。"""

    if isinstance(error, HttpServiceError):
        return error.status_code in (408, 429) or 500 <= error.status_code <= 599
    if isinstance(
        error,
        (AuthenticationError, ContextWindowExceededError, ServiceError),
    ):
        return False
    return isinstance(error, TransportError)


class MemoryExtractionWorker:
    """用一个 FIFO 消费任务依次更新当前用户和项目的笔记文件。"""

    def __init__(
        self,
        request_runner: ProviderRequestRunner,
        store: MemoryStore,
        *,
        secrets: Iterable[SecretValue] = (),
    ) -> None:
        # 负责真正发送模型请求，并收集完整响应
        self._runner = request_runner
        # 负责读取和写入长期记忆文件
        self._store = store
        # 负责记忆提取请求和响应之间的转换
        self._codec = MemoryExtractionCodec()
        # 保存需要从错误提示中替换掉的已知敏感值，例如 API Key
        self._secrets = tuple(secrets)
        # 等待后台处理的对话队列
        self._queue: asyncio.Queue[CompletedTurn | object] = asyncio.Queue()
        # 处理结果队列，供界面读取后台记忆任务的结果
        self._statuses: asyncio.Queue[MemoryWorkerStatus] = asyncio.Queue()
        # 保存正在运行的后台异步任务
        self._task: asyncio.Task[None] | None = None
        # 保存当前正在处理的记忆请求所使用的取消信号
        self._current_cancellation: CancellationToken | None = None
        # 记录后台 Worker 是否已经进入关闭状态
        self._stopping = False

    def start(self) -> None:
        """启动后台记忆任务，依次处理等待提取记忆的对话回合"""

        if self._stopping:
            raise RuntimeError("记忆后台任务已经停止")
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    def enqueue(self, turn: CompletedTurn) -> None:
        """把刚完成的一轮对话交给后台记忆任务排队处理，然后立刻返回"""

        if self._stopping:
            raise RuntimeError("记忆后台任务已经停止接收新回合")
        if self._task is None:
            raise RuntimeError("记忆后台任务尚未启动")
        self._queue.put_nowait(turn)

    async def next_status(self) -> MemoryWorkerStatus:
        """等待并返回下一条供界面显示的后台结果。"""

        return await self._statuses.get()

    async def _process(self, turn: CompletedTurn) -> None:
        """从一轮已完成的对话中提取长期记忆，并记录处理结果

        函数读取现有笔记，把本轮对话和笔记内容发送给模型，再将模型返回的创建、更新或删除操作写入记忆文件。遇到允许重试的网络错误时重新发送
        原请求一次；模型返回格式错误时发送一次带错误说明的纠正请求。处理成功、无需更新或处理失败后，都会把对应状态放入状态队列，供界面显示

        Args:
            turn: 刚刚正常结束的一轮对话，包含会话 ID 和本轮产生的全部消息
        """
        cancellation = CancellationToken()
        self._current_cancellation = cancellation
        try:
            # 读取当前全部长期记忆
            snapshot = self._store.load_snapshot()
            # 把本轮对话和现有记忆整理成发送给提取模型的请求
            original_request = self._codec.build_request(turn, snapshot)
            try:
                # 发送请求，并等待拿到完成结果
                completed = await self._runner.collect(
                    original_request,
                    cancellation,
                )
                # 解析完成结果
                update = self._codec.parse(completed)
            except MemoryResponseFormatError as exc:
                # 模型正常结束，但返回内容不能转换成一整批记忆操作，可以重试一次
                correction = self._codec.build_correction_request(
                    original_request,
                    completed,
                    exc,
                )
                completed = await self._runner.collect(correction, cancellation)
                update = self._codec.parse(completed)
            except Exception as exc:
                if not _is_retryable_request_error(exc):
                    raise
                # 说明是可以重试的网络问题，重试一次
                completed = await self._runner.collect(
                    original_request,
                    cancellation,
                )
                update = self._codec.parse(completed)
            if not update.operations:
                status = MemoryWorkerStatus(
                    MemoryWorkerStatusKind.NO_ACTION,
                    turn.session_id,
                    "本轮没有需要长期记录的内容",
                )
            else:
                # 将一批记忆写入笔记文件，并重新生成两份索引
                self._store.apply(update)
                # 上述记忆文件处理成功
                status = MemoryWorkerStatus(
                    MemoryWorkerStatusKind.SUCCEEDED,
                    turn.session_id,
                    f"已更新 {len(update.operations)} 条长期笔记",
                )
        except Exception as exc:
            message = redact_secrets(str(exc), self._secrets)
            status = MemoryWorkerStatus(
                MemoryWorkerStatusKind.FAILED,
                turn.session_id,
                f"自动笔记提取失败：{message}",
            )
        finally:
            self._current_cancellation = None
        # 把本轮记忆处理结果立即放进异步状态队列，不等待
        self._statuses.put_nowait(status)

    async def _run(self) -> None:
        """依次处理队列中等待提取长期记忆的对话回合

        函数持续从任务队列中取出已完成的对话回合，并调用 ``_process`` 处理。收到停止标记时结束运行。每取出一项，无论处理成功、失败还是停止，都会
        通知队列该项已经处理完毕。
        """
        while True:
            item = await self._queue.get()
            try:
                if item is _STOP:
                    return
                assert isinstance(item, CompletedTurn)
                await self._process(item)
            finally:
                # 通知队列,刚才通过 get() 取出的那一项，现在已经处理完了
                self._queue.task_done()

    def _discard_queue(self) -> None:
        """取出并丢弃队列中仍在等待的所有项目

        已经被后台任务取出并开始处理的回合不在队列中，因此本函数不会处理或
        取消它。当前请求需要由调用方通过取消令牌另行取消
        """
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            else:
                self._queue.task_done()

    async def close(
        self,
        timeout_seconds: float = MEMORY_SHUTDOWN_TIMEOUT_SECONDS,
    ) -> bool:
        """止后台记忆任务，并限时等待队列中的对话处理完毕

        函数先停止接收新的对话回合，再把停止标记放到队尾，让后台任务处理完已经排队的回合后退出。如果超过等待时间仍未结束，则取消当前模型请求
        和后台任务，并丢弃队列中尚未处理的回合

        Args:
            timeout_seconds: 最多等待后台任务正常结束的秒数，不能小于零

        Returns:
            后台任务正常结束或尚未启动时返回 True；等待超时、任务被取消或尚未结束时返回 False
        """

        if timeout_seconds < 0:
            raise ValueError("记忆后台关闭超时不能为负数")
        if self._stopping:
            if self._task is None:
                return True
            return self._task.done() and not self._task.cancelled()
        self._stopping = True
        if self._task is None:
            return True
        self._queue.put_nowait(_STOP)
        try:
            await asyncio.wait_for(
                asyncio.shield(self._task),
                timeout=timeout_seconds,
            )
            return True
        except TimeoutError:
            if self._current_cancellation is not None:
                self._current_cancellation.cancel()
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._discard_queue()
            return False
