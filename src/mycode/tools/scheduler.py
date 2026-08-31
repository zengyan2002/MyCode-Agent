"""按模型顺序、有界调度单次响应中的工具调用。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence

from mycode.agent.cancellation import CancellationToken
from mycode.hooks.runtime import HookRunScope
from mycode.models.events import (
    AgentRunOptions,
    ToolResultEvent,
    ToolStartedEvent,
)
from mycode.models.messages import ToolCall
from mycode.models.tools import (
    ToolActivationState,
    ToolAccess,
    ToolErrorCode,
    ToolExecutionResult,
    ToolInvocation,
)
from mycode.tools.interceptors import (
    AfterToolObserver,
    BeforeToolInterceptor,
    ToolRunContext,
    notify_observers,
)
from mycode.tools.executor import ToolExecutor
from mycode.tools.registry import ToolRegistry

# ToolScheduler 是无状态工厂；每批模型工具调用都会创建独立 session，
# 运行中的 Task、结果和取消收口状态绝不在多个 Agent 轮次间共享。
class ToolScheduler:
    def __init__(
        self,
        registry: ToolRegistry,
        executor: ToolExecutor,
        *,
        interceptors: Sequence[BeforeToolInterceptor] = (),
        observers: Sequence[AfterToolObserver] = (),
    ) -> None:
        self._registry = registry
        self._executor = executor
        self._interceptors = tuple(interceptors)
        self._observers = tuple(observers)

    @property
    def tool_activation(self) -> ToolActivationState:
        """返回工具执行上下文中当前 Agent 独享的 MCP 激活状态。

        Returns:
            tool_search 修改、下一轮 Provider 工具视图读取的同一对象。
        """

        return self._executor.context.tool_activation

    # 将模型调用冻结为带权限、轮次和原始索引的执行记录。这里只创建会话，
    # 不会立即启动工具，调用方因此能在任意取消路径统一 finalize。
    def schedule(
        self,
        calls: tuple[ToolCall, ...],
        *,
        model_call_number: int,
        options: AgentRunOptions,
        cancellation: CancellationToken,
        hook_scope: HookRunScope,
        visible_tool_names: frozenset[str] | None = None,
    ) -> ToolScheduleSession:
        """为模型本轮返回的工具调用创建调度会话。

        Args:
            calls: 模型按顺序返回的工具调用。
            model_call_number: 当前 Agent 模型轮次，从 1 开始。
            options: Plan 模式、读取并发数和轮次上限。
            cancellation: 当前用户请求的取消信号。
            visible_tool_names: Provider 本轮实际收到的工具名。None 只用于
                尚未接入 ToolView 的兼容调用方。
            hook_scope: 发起本批调用的主会话或 Skill fork Hook 状态。

        Returns:
            尚未启动工具的 ToolScheduleSession。

        Raises:
            ValueError: calls 为空。
        """

        if not calls:
            raise ValueError("工具调度至少需要一个调用")
        # 未知工具按写类处理：即使后续执行器会返回“未知工具”，也不能让
        # 未声明分类的调用绕过写屏障或仅规划模式拦截。
        invocation_list: list[ToolInvocation] = []
        for index, call in enumerate(calls):
            tool = self._registry.get(call.name)
            access = (
                tool.definition.access
                if tool is not None
                else ToolAccess.WRITE
            )

            invocation_list.append(
                ToolInvocation(
                    call=call,
                    access=access,
                    model_call_number=model_call_number,
                    call_index=index,
                )
            )
        invocations = tuple(invocation_list)

        return ToolScheduleSession(
            invocations,
            self._executor,
            self._interceptors,
            self._observers,
            options,
            cancellation,
            hook_scope,
            visible_tool_names,
        )

#负责执行模型本轮返回的一批工具调用，并管理执行顺序、并发、取消和结果收尾。
class ToolScheduleSession:
    """
    管理一次模型响应中整批工具调用的执行生命周期。
    连续的只读工具按并发上限执行，写工具严格串行执行；
    执行过程中通过 stream() 产出工具开始和完成事件；
    正常结束或取消后通过 finalize() 回收活跃任务、补齐结果，
    并按模型声明工具调用的原始顺序返回最终结果。
    """
    def __init__(
        self,
        invocations: tuple[ToolInvocation, ...],
        executor: ToolExecutor,
        interceptors: tuple[BeforeToolInterceptor, ...],
        observers: tuple[AfterToolObserver, ...],
        options: AgentRunOptions,
        cancellation: CancellationToken,
        hook_scope: HookRunScope,
        visible_tool_names: frozenset[str] | None,
    ) -> None:
        self._invocations = invocations
        self._executor = executor
        self._interceptors = interceptors
        self._observers = observers
        self._options = options
        self._cancellation = cancellation
        self._hook_scope = hook_scope
        # None 保留旧调用行为；集合表示必须严格限制在本轮模型可见名字中。
        self._visible_tool_names = visible_tool_names
        # 并发读工具按完成顺序产生事件，但结果以原始 call_index 为键保存；
        # finalize 时再按模型声明顺序排列，保证回灌历史稳定。
        self._results: dict[int, ToolExecutionResult] = {}
        #记录当前正在运行的异步工具任务，以及每个任务对应的工具调用信息。
        self._active: dict[
            asyncio.Task[ToolExecutionResult], ToolInvocation
        ] = {}
        #记录一次工具调度事件流的生命周期
        self._stream_started = False
        self._stream_finished = False
        # finalize 必须幂等，因为正常路径、取消分支和生成器 finally 都可能
        # 尝试收口同一 session，工具副作用和结果都不能重复产生。
        self._finalized: tuple[ToolExecutionResult, ...] | None = None

    @property
    def invocations(self) -> tuple[ToolInvocation, ...]:
        return self._invocations

    # 只合并连续的 READ 调用。每个 WRITE 单独成组，既等待前一读批次完成，
    # 也阻止后一读批次提前开始，因此天然形成顺序屏障。
    def _groups(self) -> tuple[tuple[ToolInvocation, ...], ...]:
        """
            举例
            0：read_file     READ
            1：search_code   READ
            2：write_file    WRITE
            3：read_file     READ
            得到
            groups = (
            (read_file, search_code),  # 读取组
            (write_file,),             # 写入组
            (read_file,),              # 读取组
            )
        """
        groups: list[tuple[ToolInvocation, ...]] = []
        reads: list[ToolInvocation] = []
        for invocation in self._invocations:
            if invocation.access is ToolAccess.READ:
                reads.append(invocation)
                continue
            if reads:
                groups.append(tuple(reads))
                reads.clear()
            groups.append((invocation,))
        if reads:
            groups.append(tuple(reads))
        return tuple(groups)

    # 拦截器在 ToolExecutor 之前运行，策略拒绝不会触发具体工具副作用；
    # 拒绝结果仍采用统一 ToolExecutionResult，Agent 无需维护另一套协议。
    # 串联拦截器-执行器-观察器
    async def _execute(
        self,
        invocation: ToolInvocation,
    ) -> ToolExecutionResult:
        started = time.monotonic()
        context = ToolRunContext(
            invocation=invocation,
            options=self._options,
            hook_scope=self._hook_scope,
        )
        try:
            if (
                self._visible_tool_names is not None
                and invocation.call.name not in self._visible_tool_names
            ):
                return self._failure(
                    invocation,
                    ToolErrorCode.BLOCKED,
                    "该工具不在当前 Skill 允许的工具范围内",
                    started,
                )
            for interceptor in self._interceptors:
                decision = await interceptor.before_tool(context)
                if not decision.allowed:
                    assert decision.error_code is not None
                    assert decision.message is not None
                    return self._failure(
                        invocation,
                        decision.error_code,
                        decision.message,
                        started,
                    )
            result = await self._executor.execute(invocation.call)
            # observer 只观察真正经过 Executor 的调用；被 Plan 模式拦截的
            # 调用没有执行事实，不应被审计为“已执行”。
            await notify_observers(
                self._observers,
                context,
                result,
            )
            return result
        except asyncio.CancelledError:
            raise
        except Exception:
            # 调度扩展点的未知异常不能击穿 Agent。固定错误文本避免把工具
            # 参数、路径或环境细节意外回灌给模型。
            return self._failure(
                invocation,
                ToolErrorCode.INTERNAL_ERROR,
                "工具因内部调度错误而未能执行",
                started,
            )

    def _failure(
        self,
        invocation: ToolInvocation,
        code: ToolErrorCode,
        message: str,
        started: float | None = None,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_call_id=invocation.call.id,
            tool_name=invocation.call.name,
            success=False,
            content="",
            error_code=code,
            error_message=message,
            timed_out=code is ToolErrorCode.TIMEOUT,
            truncated=False,
            original_size_bytes=0,
            duration_ms=(
                0
                if started is None
                else max(0, round((time.monotonic() - started) * 1000))
            ),
        )

    #执行本轮工具批次中的一个写工具调用。
    #不允许并发，一次只能执行一个
    async def _stream_write(
        self,
        invocation: ToolInvocation,
    ) -> AsyncIterator[ToolStartedEvent | ToolResultEvent]:
        # 写任务一次只启动一个。开始事件在创建任务前发出，使 UI 能在长操作开始时立即反馈，同时保持模型声明的严格顺序。

        #如果用户取消当前任务，不执行任何工具调用
        if self._cancellation.is_cancelled:
            return

        #产出工具开始事件
        yield ToolStartedEvent(invocation)

        #因为yield会暂停函数，暂停期间，取消状态可能发生变化，恢复后必须再检查一遍
        # 如果用户取消当前任务，不执行任何工具调用
        if self._cancellation.is_cancelled:
            return

        #启动一个异步工具任务，并保存task对象，方便后面同时监听工具完成和用户取消
        task = asyncio.create_task(self._execute(invocation))

        #将当前正在执行的异步任务放进_active中
        self._active[task] = invocation

        #创建一个取消监听器
        cancel_waiter = asyncio.create_task(self._cancellation.wait())
        try:
            #同时等工具执行和取消监听器，谁先有消息就先处理谁
            done, _ = await asyncio.wait(
                {task, cancel_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
            #取消监听器先收到用户的取消任务
            if cancel_waiter in done and not task.done():
                # 此处不直接取消 Task；统一由 finalize(reason) 取消并等待，
                # 防止事件流和收口路径同时清理同一个工具。
                return
            #任务先执行完
            result = await task
            #删除键为 task 的记录；如果字典里找不到这个任务，就返回 None，不要抛出 KeyError。
            self._active.pop(task, None)
            self._results[invocation.call_index] = result
            yield ToolResultEvent(invocation, result)
        finally:
            #清理等待用户取消指令的等待器
            cancel_waiter.cancel()
            await asyncio.gather(cancel_waiter, return_exceptions=True)

    #执行当前连续只读分组中的多个工具调用，并发上限由 max_read_concurrency 控制，同时不断产出开始事件和结果事件。
    async def _stream_reads(
        self,
        invocations: tuple[ToolInvocation, ...],
    ) -> AsyncIterator[ToolStartedEvent | ToolResultEvent]:
        # 显式维护 active 集合，而不是一次 gather 整批任务，以便限制并发数，并让每个完成结果都能立即作为事件向上游发送。

        #准备要执行的任务列表   后面通过queued.pop(0)不断从前面取任务
        queued = list(invocations)

        #异步队列，用来装已经执行完成的 Task
        completed_tasks: asyncio.Queue[
            asyncio.Task[ToolExecutionResult]
        ] = asyncio.Queue(maxsize=self._options.max_read_concurrency)

        #用户取消等待器，取消监听任务，
        cancel_waiter = asyncio.create_task(self._cancellation.wait())
        try:
            #只要还有待启动或正在运行的工具，并且任务没有被取消，就继续循环。
            while (queued or self._active) and not self._cancellation.is_cancelled:
                #在并发上限允许的情况下，尽可能多地启动读取工具。
                while (
                    queued
                    and len(self._active) < self._options.max_read_concurrency
                    and not self._cancellation.is_cancelled
                ):
                    #取出待完成工具队列中的第一个工具
                    invocation = queued.pop(0)

                    #暂停函数，并返回当前工具开始事件
                    yield ToolStartedEvent(invocation)

                    #如果用户取消，退出当前循环，不再启动更多的读取工具
                    if self._cancellation.is_cancelled:
                        break
                    #创建并执行
                    task = asyncio.create_task(self._execute(invocation))
                    #讲当前执行的任务放到正在运行的列表中
                    self._active[task] = invocation
                    # 回调只把已完成 Task 放入队列；读取结果和修改 _results仍在当前协程中串行完成，避免回调竞争共享状态。
                    def on_done(finished_task):
                        #立即向队列放入元素，不使用 await。
                        completed_tasks.put_nowait(finished_task)

                    # 给任务注册完成回调。任务成功、抛异常或被取消时，都会调用它。
                    task.add_done_callback(on_done)

                #如果没有在运行的工具或者用户取消，就直接退出循环
                if not self._active or self._cancellation.is_cancelled:
                    break

                # 创建监听任务，等待从完成队列中取出下一个已完成的工具 Task；
                completion_waiter = asyncio.create_task(
                    completed_tasks.get()
                )

                # 同时等待“下一个读取工具完成”和“收到取消信号”；谁先发生就执行谁
                done, _ = await asyncio.wait(
                    {completion_waiter, cancel_waiter},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # 如果先发生取消事件
                if cancel_waiter in done:
                    completion_waiter.cancel()
                    await asyncio.gather(
                        completion_waiter,
                        return_exceptions=True,
                    )
                    break

                #先发生工具任务事件
                task = completion_waiter.result()
                invocation = self._active.pop(task)
                result = await task
                self._results[invocation.call_index] = result
                yield ToolResultEvent(invocation, result)
        finally:
            cancel_waiter.cancel()
            await asyncio.gather(cancel_waiter, return_exceptions=True)

    async def stream(
        self,
    ) -> AsyncIterator[ToolStartedEvent | ToolResultEvent]:
        """
        按调度分组执行本批工具调用，并流式产出工具生命周期事件。

        连续的只读工具组成一个并发执行组，由 _stream_reads() 负责执行；
        写工具单独成组，由 _stream_write() 串行执行。不同分组按照原始
        调用顺序依次处理，从而保证写操作不会与前后的读取操作交叉执行。

        每个工具执行过程中会依次产出：

        - ToolStartedEvent：工具开始执行；
        - ToolResultEvent：工具执行完成，并携带执行结果。

        同一个 ToolScheduleSession 的事件流只能消费一次，防止调用方重复
        遍历导致写工具重复执行。取消发生后不再启动后续分组，尚未完成的
        调用由 finalize() 根据终止原因补齐结果。

        Yields:
            ToolStartedEvent | ToolResultEvent:
                工具开始事件或工具结果事件。
        """
        # 同一 session 的事件流只能消费一次，否则写工具可能重复产生副作用。
        if self._stream_started:
            raise RuntimeError("同一工具调度事件流只能消费一次")
        self._stream_started = True
        try:
            for group in self._groups():
                if self._cancellation.is_cancelled:
                    break
                #当前组的第一个为只读工具，则该组都是只读
                if group[0].access is ToolAccess.READ:
                    async for event in self._stream_reads(group):
                        yield event
                else:
                    async for event in self._stream_write(group[0]):
                        yield event
        finally:
            # finished 只表示事件迭代已经离开，不等于所有工具都有结果；
            # 中途取消时 finalize 仍必须接收明确终止原因。
            self._stream_finished = True

    async def finalize(
        self,
        reason: ToolErrorCode | None = None,
    ) -> tuple[ToolExecutionResult, ...]:
        """结束本批工具调用，并返回按原始调用顺序排列的结果。

        Args:
            reason: 异常结束的错误码；None 表示正常结束。

        Returns:
            本批所有工具调用的结果。
        """

        #防止重复收尾
        if self._finalized is not None:
            return self._finalized

        #reason为None就代表调用方声称这是正常结束
        #正常结束得满足所有事件流都结束以及所有调用都有结果
        if reason is None and (
            not self._stream_finished
            or len(self._results) != len(self._invocations)
        ):
            raise RuntimeError("未完成的工具调度必须提供终止原因")

        active = tuple(self._active)
        if reason is not None:
            # 先取消所有活跃任务，再 gather 等待工具完成自己的清理逻辑，
            # 尤其是 execute_command 对整个子进程树的回收。
            for task in active:
                task.cancel()
        if active:
            settled = await asyncio.gather(*active, return_exceptions=True)
            for task, outcome in zip(active, settled, strict=True):
                invocation = self._active.pop(task)
                if isinstance(outcome, ToolExecutionResult):
                    self._results[invocation.call_index] = outcome

        #这部分是在异常收尾时，给没有结果的工具调用补上一个统一的失败结果。
        if reason is not None:
            for invocation in self._invocations:
                if invocation.call_index not in self._results:
                    self._results[invocation.call_index] = self._failure(
                        invocation,
                        reason,
                        "工具调用在完成前被取消",
                    )

        # 无论并发完成顺序如何，最终结果都严格按原始 call_index 排列，
        # 以满足 Provider 对 tool_calls/tool_results 配对顺序的要求。
        self._finalized = tuple(
            self._results[index] for index in range(len(self._invocations))
        )
        return self._finalized
