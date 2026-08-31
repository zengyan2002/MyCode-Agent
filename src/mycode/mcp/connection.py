"""单个 MCP Server 的传输、会话和资源生命周期。"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from contextlib import AsyncExitStack
from enum import Enum
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, Tool as SdkTool

from mycode.models.config import (
    HttpMcpServerConfig,
    McpServerConfig,
    StdioMcpServerConfig,
)
from mycode.models.json_types import JsonValue


class McpConnectionState(str, Enum):
    """描述单个 MCP 连接的生命周期状态。"""
    # 连接对象刚创建，还没有开始连接
    NEW = "new"
    # 正在建立连接
    OPENING = "opening"
    # 连接已经完全可用，只有处于当前状态可以调用工具
    READY = "ready"
    # 连接建立失败，或者持有连接的后台任务意外终止
    FAILED = "failed"
    # 连接已经关闭，底层资源开始或已经释放
    CLOSED = "closed"


class McpConnectionStage(str, Enum):
    """标识连接建立过程中最近执行的阶段。"""
    # 正在建立底层连接
    CONNECT = "connect"
    # 底层传输已经打开，正在执行 MCP 初始化握手
    INITIALIZE = "initialize"
    # MCP 初始化成功，正在获取 Server 提供的工具列表
    LIST_TOOLS = "list_tools"
    # 所有启动步骤都已完成：底层传输已打开，MCP 初始化已完成，工具列表已获取，ClientSession 可以使用
    READY = "ready"
    # 连接已经进入关闭状态
    CLOSED = "closed"


class McpConnection:
    """维护一个 MCP Server 的独立会话及其全部底层资源。"""

    def __init__(
        self,
        config: McpServerConfig,
    ) -> None:
        """创建尚未打开的单 Server 连接。

        Args:
            config: 已完成校验的单台 MCP Server 配置，决定使用 stdio
                还是 Streamable HTTP。

        Returns:
            None。
        """
        # 当前连接对应的不可变 Server 配置。
        self._config = config
        # 串行化打开和关闭状态变化，不限制并发工具调用。异步锁，用来防止“打开”和“关闭”同时修改连接状态
        self._state_lock = asyncio.Lock()
        # 完成初始化后可用于枚举和调用工具的 SDK 会话。
        self._session: ClientSession | None = None
        # 后台异步任务，专门负责打开 MCP 连接、一直保持连接、收到关闭通知后释放连接资源。
        self._owner_task: asyncio.Task[None] | None = None
        # 通知资源所有者执行正常关闭的事件。
        self._close_requested: asyncio.Event | None = None
        # 关闭通知：_run_owner() 等待该事件  aclose() 设置该事件以触发资源清理
        self._state = McpConnectionState.NEW
        # 启动期间最近进入的可报告阶段。
        self._stage = McpConnectionStage.CONNECT

    @property
    def server_name(self) -> str:
        """返回配置中的 Server 名。

        Returns:
            用于工具命名空间和诊断的 Server 名。
        """
        return self._config.name

    @property
    def state(self) -> McpConnectionState:
        """返回连接当前生命周期状态。

        Returns:
            当前连接状态枚举。
        """
        return self._state

    @property
    def stage(self) -> McpConnectionStage:
        """返回启动过程最近进入的阶段。

        Returns:
            可用于生成安全启动报告的阶段枚举。
        """
        return self._stage

    async def _open_transport(
        self,
        stack: AsyncExitStack,
    ) -> tuple[Any, Any]:
        """判断当前 Server 配置的是 stdio 还是 streamable_http，然后真正打开对应连接，最后返回 MCP 消息的读取流和写入流。

        Args:
            stack: 拥有本次连接全部资源的异步退出栈。

        Returns:
            SDK 传输提供的读写流及可选会话 ID 读取器。
        """
        if isinstance(self._config, StdioMcpServerConfig):
            # 如果是stdio的配置
            child_environment = dict(os.environ)
            child_environment.update(
                (key, value.reveal()) for key, value in self._config.env
            )
            parameters = StdioServerParameters(
                command=self._config.command,
                args=list(self._config.args),
                env=child_environment,
            )
            errlog = open(os.devnull, "w", encoding="utf-8")
            # 把“以后关闭 errlog 的操作”登记到资源栈，现在不会关闭
            stack.callback(errlog.close)
            # 创建 stdio 传输上下文管理器，进入上下文，启动 MCP 子进程并得到读写流，返回读写流
            return await stack.enter_async_context(
                stdio_client(parameters, errlog=errlog)
            )

        # 把配置中的请求头转化为python字典
        headers = {
            key: value.reveal() for key, value in self._config.headers
        }
        # 创建HTTP客户端
        client = await stack.enter_async_context(
            httpx.AsyncClient(headers=headers)
        )
        # 建立 MCP 通信通道
        streams = await stack.enter_async_context(
            streamable_http_client(
                self._config.url,
                http_client=client,
            )
        )
        # 返回读写流
        return streams[0], streams[1]

    async def _run_owner(
        self,
        opened: asyncio.Future[tuple[SdkTool, ...]],
        close_requested: asyncio.Event,
    ) -> None:
        """一个长期运行的后台任务：它创建并持有 MCP 连接，把启动结果通过 opened 交出去，然后一直等到 close_requested 发出关闭通知，最后统一释放资源。

        Args:
            opened: 向调用方交付工具列表或启动异常的结果 Future。
            close_requested: 管理器发出正常关闭请求时设置的事件。

        Returns:
            None。
        """
        # 创建资源清理栈，用来管理HTTP 客户端或 stdio 子进程、MCP 传输、ClientSession
        stack = AsyncExitStack()
        # 记录启动或运行过程中出现的异常
        opening_error: BaseException | None = None
        # 记录后台任务是否因为正常关闭而被取消
        owner_cancelled = False
        # 连接是否已经成功进入可用状态
        ready = False
        try:
            # 打开 MCP 传输
            read_stream, write_stream = await self._open_transport(stack)
            # 创建 MCP 会话，session 明确是 ClientSession
            session = ClientSession(read_stream, write_stream)

            # 进入会话上下文，并把关闭操作登记到资源栈
            await stack.enter_async_context(session)

            self._session = session

            # MCP初始化握手
            # 这里直接调用MCP库的初始化方法，起始底层实现是三步，
            # 第一步：Client → Server：发送 initialize 请求；第二步：Server → Client：返回协议版本、Server 信息和能力；第三步：Client → Server：发送 notifications/initialized 通知
            self._stage = McpConnectionStage.INITIALIZE
            await session.initialize()

            # 工具发现
            # 记录现阶段的_stage
            self._stage = McpConnectionStage.LIST_TOOLS
            # 用来保存从所有分页中取得的工具
            tools: list[SdkTool] = []
            # 传给下一次 tools/list 请求的续页游标；首次请求为 None
            cursor: str | None = None
            # 创建一个集合，记录已经见过的分页游标
            seen_cursors: set[str] = set()
            # 循环取出每页的工具
            while True:
                page = await session.list_tools(cursor=cursor)
                tools.extend(page.tools)
                next_cursor = page.nextCursor
                if next_cursor is None:
                    break
                if next_cursor in seen_cursors:
                    raise RuntimeError(
                        f"MCP Server {self.server_name} 返回了重复分页游标"
                    )
                seen_cursors.add(next_cursor)
                cursor = next_cursor
            # 设置连接进入可用状态
            ready = True
            self._state = McpConnectionState.READY
            self._stage = McpConnectionStage.READY
            opened.set_result(tuple(tools))
            # 让当前后台任务停在这里，等待别人发出“可以关闭连接了”的通知
            await close_requested.wait()
        except asyncio.CancelledError:
            owner_cancelled = close_requested.is_set()
            if not owner_cancelled:
                opening_error = RuntimeError(
                    f"MCP Server {self.server_name} 的 SDK 会话意外取消"
                )
                self._state = McpConnectionState.FAILED
        except BaseException as exc:
            opening_error = exc
            if not ready:
                self._state = McpConnectionState.FAILED
        finally:
            self._session = None
            close_error: BaseException | None = None
            try:
                # 反向关闭并清理之前登记到 stack 里的所有资源
                await stack.aclose()
            except BaseException as exc:
                close_error = exc

            if not opened.done():
                # 如果如果 opened 还没有拿到最终结果，就必须给它一个结果
                if owner_cancelled:

                    opened.cancel()
                else:
                    opened.set_exception(
                        opening_error
                        or close_error
                        or RuntimeError(
                            f"MCP Server {self.server_name} 启动失败"
                        )
                    )
            if ready and close_error is not None:
                raise close_error

    async def open_and_list_tools(self) -> tuple[SdkTool, ...]:
        """启动当前 MCP Server 连接，并在连接可用后返回它提供的全部工具

        Returns:
            按远端分页顺序汇总的 SDK 工具定义元组。
        """
        if self._state is not McpConnectionState.NEW:
            # 检查连接是第一次打开
            raise RuntimeError(
                f"MCP Server {self.server_name} 的连接不能重复打开"
            )
        self._state = McpConnectionState.OPENING
        self._stage = McpConnectionStage.CONNECT
        # 取得当前事件循环
        loop = asyncio.get_running_loop()
        # 创建一个尚未完成的 Future
        opened: asyncio.Future[tuple[SdkTool, ...]] = (
            loop.create_future()
        )
        # 创建一个异步事件，作为“关闭连接”信号
        close_requested = asyncio.Event()
        self._close_requested = close_requested
        # 把 _run_owner() 启动成独立的后台异步任务
        self._owner_task = asyncio.create_task(
            self._run_owner(opened, close_requested)
        )
        # shield防止open_and_list_tools() 被取消的同时，opened被取消使self._run_owner报错
        return await asyncio.shield(opened)

    async def call_tool(
        self,
        remote_name: str,
        arguments: Mapping[str, JsonValue],
    ) -> CallToolResult:
        """通过已初始化会话调用一个远端工具。

        Args:
            remote_name: Server 原始工具名，不含本地命名空间。
            arguments: 已通过本地 JSON Schema 校验的参数。

        Returns:
            SDK 返回的 MCP 工具调用结果。
        """
        session = self._session
        if self._state is not McpConnectionState.READY or session is None:
            raise RuntimeError(
                f"MCP Server {self.server_name} 的连接尚未完成初始化"
            )
        return await session.call_tool(remote_name, dict(arguments))

    async def aclose(self) -> None:
        """幂等关闭当前连接拥有的会话、传输和客户端。

        Returns:
            None。
        """

        # 获取状态锁 确保多个任务同时调用 aclose() 时，只有一个任务执行真正的关闭流程。
        async with self._state_lock:
            # 如果已经关闭就直接返回
            if self._state is McpConnectionState.CLOSED:
                return
            # 保存关闭前的状态
            previous_state = self._state
            # 取得后台任务
            owner_task = self._owner_task
            # 取出关闭事件
            close_requested = self._close_requested
            # 立即禁止继续调用工具
            self._session = None
            # 将连接状态标记为已关闭
            self._state = McpConnectionState.CLOSED
            self._stage = McpConnectionStage.CLOSED

            if close_requested is not None:
                # 如果后台任务已经创建，就设置关闭事件
                close_requested.set()
            if (
                previous_state is McpConnectionState.OPENING
                and owner_task is not None
                and not owner_task.done()
            ):
                # 如果还在启动，向后台任务发出“请取消”的请求
                owner_task.cancel()
            if (
                owner_task is not None
                and owner_task is not asyncio.current_task()
            ):
                await owner_task
            self._owner_task = None
            self._close_requested = None
