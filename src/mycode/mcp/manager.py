"""并发发现 MCP Server、稳定注册工具并管理连接缓存。"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from mcp.types import Tool as SdkTool

from mycode.errors import redact_secrets
from mycode.mcp.connection import McpConnection, McpConnectionStage
from mycode.mcp.tool import McpToolAdapter
from mycode.models.config import McpServerConfig, SecretValue
from mycode.models.tools import ToolSource
from mycode.tools.registry import ToolRegistry

class McpStartupStage(str, Enum):
    """标识 MCP 启动或关闭问题发生的阶段。"""

    CONNECT = "connect"
    INITIALIZE = "initialize"
    LIST_TOOLS = "list_tools"
    REGISTER_TOOLS = "register_tools"
    CLOSE = "close"


@dataclass(frozen=True)
class McpIssue:
    """描述一个可安全向用户展示的 MCP 局部问题。"""

    # 发生问题的配置 Server 名。
    server_name: str
    # 问题所在的连接、枚举、注册或关闭阶段。
    stage: McpStartupStage
    # 已脱敏且不包含原始异常详情的说明。
    message: str


@dataclass(frozen=True)
class McpStartupReport:
    """汇总 MCP 启动后仍可用的连接、工具和局部问题。"""

    # 保存启动成功、至少注册了一个工具，并且仍然保持连接的 MCP Server名称
    connected_servers: tuple[str, ...]
    # 保存成功注册到 MyCode工具注册表中的本地工具名
    registered_tools: tuple[str, ...]
    # 保存 MCP启动过程中发生的问题
    issues: tuple[McpIssue, ...]


@dataclass(frozen=True)
class McpCloseReport:
    """汇总 MCP 管理器关闭连接的实际结果。"""

    # 已成功关闭的 Server 名
    closed_servers: tuple[str, ...]
    # 记录关闭失败的 Server 信息及失败原因
    issues: tuple[McpIssue, ...]


@dataclass(frozen=True)
class _DiscoveryOutcome:
    """MCP 管理器尝试连接一台 MCP Server后，记录这台 Server的连接对象、工具列表和错误信息"""

    # 本次发现使用的连接对象。
    connection: McpConnection
    # 成功枚举的远端工具；发现失败时为空
    tools: tuple[SdkTool, ...]
    # 发现失败时生成的安全问题；成功时为空。
    issue: McpIssue | None


class _ManagerState(str, Enum):
    """表示 MCP 管理器从创建、启动到关闭的当前状态"""
    # 管理器刚创建，尚未连接任何 MCP Server，可以调用 start()
    NEW = "new"

    # 正在并发连接 MCP Server、获取工具并注册到工具表
    STARTING = "starting"

    # 启动流程已经结束；可用 Server 已保存，合法工具已完成注册
    # 部分 Server 启动失败也可能进入该状态
    STARTED = "started"

    # 管理器已经关闭或启动被取消，连接资源已开始或已经完成清理
    # 不能再次调用 start()
    CLOSED = "closed"


def _stage_for_connection(stage: McpConnectionStage) -> McpStartupStage:
    """把连接内部阶段转换成 Manager错误报告使用的“问题发生阶段”

    Args:
        stage: 连接最近进入的生命周期阶段。

    Returns:
        对应的公开 MCP 启动阶段。
    """
    if stage is McpConnectionStage.INITIALIZE:
        return McpStartupStage.INITIALIZE
    if stage is McpConnectionStage.LIST_TOOLS:
        return McpStartupStage.LIST_TOOLS
    return McpStartupStage.CONNECT


class McpManager:
    """并行发现多个 MCP Server，并缓存存在可用工具的连接。"""

    def __init__(
        self,
        configs: tuple[McpServerConfig, ...],
        *,
        timeout_seconds: float = 30.0,
        secrets: Iterable[SecretValue] = (),
    ) -> None:
        """创建尚未启动的 MCP 管理器。

        Args:
            configs: 两层配置合并后的 MCP Server 配置。
            timeout_seconds: 每个 Server 从连接到枚举结束的总秒数上限。
            secrets: 生成工具结果和报告时需要移除的敏感值。

        Returns:
            None。
        """
        if timeout_seconds <= 0:
            raise ValueError("MCP Server 启动超时时间必须为正数")
        # 按配置合并顺序保存的 Server 配置。
        self._configs = configs
        # 单 Server 连接、初始化和工具枚举的超时上限。
        self._timeout_seconds = timeout_seconds
        # 用于远端输出和报告的敏感值快照。
        self._secrets = tuple(secrets)
        # 串行化启动和关闭，避免重复注册或交叉清理。
        self._state_lock = asyncio.Lock()
        # 当前管理器生命周期状态。
        self._state = _ManagerState.NEW
        # 只保存至少注册成功一个工具的活动连接。
        self._connections: dict[str, McpConnection] = {}
        # 首次关闭后缓存的稳定报告。
        self._close_report: McpCloseReport | None = None

    def _safe_message(self, message: str) -> str:
        """从报告文本中移除所有已知敏感值。

        Args:
            message: 等待进入用户可见报告的文本。

        Returns:
            已应用统一敏感值替换的安全文本。
        """
        return redact_secrets(message, self._secrets)

    async def _close_quietly(self, connection: McpConnection) -> bool:
        """尝试关闭连接并把异常收口为布尔结果。

        Args:
            connection: 需要局部清理的 Server 连接。

        Returns:
            关闭成功返回 ``True``，发生异常返回 ``False``。
        """
        try:
            await connection.aclose()
        except Exception:
            return False
        return True

    async def _discover_one(
        self,
        config: McpServerConfig,
    ) -> _DiscoveryOutcome:
        """在总超时内完成单个 Server 的连接和工具枚举。

        Args:
            config: 当前 Server 的已校验配置。

        Returns:
            成功后返回工具列表、MCP连接对象、发现失败时生成的安全问题
        """
        # 根据config获取MCP Server连接对象
        connection = McpConnection(config)
        try:
            # 等待获取工具列表，超时直接抛出异常
            tools = await asyncio.wait_for(
                connection.open_and_list_tools(),
                timeout=self._timeout_seconds,
            )
            return _DiscoveryOutcome(connection, tools, None)
        except asyncio.CancelledError:
            # 如果中途用户取消，则尝试关闭连接
            await self._close_quietly(connection)
            raise
        except TimeoutError:
            # 取得超时时的最近连接状态，并转换为MCP管理器的错误
            stage = _stage_for_connection(connection.stage)
            await self._close_quietly(connection)
            issue = McpIssue(
                server_name=config.name,
                stage=stage,
                message=self._safe_message(
                    f"MCP Server {config.name} 启动超过 "
                    f"{self._timeout_seconds:g} 秒，已跳过"
                ),
            )
            return _DiscoveryOutcome(connection, (), issue)
        except Exception:
            stage = _stage_for_connection(connection.stage)
            await self._close_quietly(connection)
            issue = McpIssue(
                server_name=config.name,
                stage=stage,
                message=self._safe_message(
                    f"MCP Server {config.name} 在 {stage.value} 阶段失败，"
                    "已跳过"
                ),
            )
            return _DiscoveryOutcome(connection, (), issue)

    def _register_server_tools(
        self,
        outcome: _DiscoveryOutcome,
        registry: ToolRegistry,
    ) -> tuple[list[str], list[McpIssue]]:
        """
        检查并注册一个 MCP Server 提供的工具
        工具会先按名称排序，然后逐个检查名称、参数格式和重名情况。
        检查通过的工具会注册到 MyCode 的工具表中；检查失败的工具会被跳过，
        并记录对应的问题

        Args:
            outcome: 成功完成枚举的单 Server 结果
            registry: MyCode 的工具注册表

        Returns:
            成功注册的工具名称列表，以及注册过程中发现的问题列表
        """
        # 保存最后成功注册的工具名
        registered: list[str] = []
        # 记录注册工具过程中发现的问题
        issues: list[McpIssue] = []
        # 记录已经处理过的远端工具名
        seen_remote_names: set[str] = set()
        for remote_tool in sorted(outcome.tools, key=lambda item: item.name):
            if remote_tool.name in seen_remote_names:
                issues.append(
                    McpIssue(
                        server_name=outcome.connection.server_name,
                        stage=McpStartupStage.REGISTER_TOOLS,
                        message=(
                            f"MCP Server {outcome.connection.server_name} "
                            f"重复声明工具 {remote_tool.name}，已跳过"
                        ),
                    )
                )
                continue
            seen_remote_names.add(remote_tool.name)
            try:
                adapter = McpToolAdapter.from_remote_tool(
                    outcome.connection.server_name,
                    remote_tool,
                    outcome.connection,
                    secrets=self._secrets,
                )
            except (TypeError, ValueError):
                issues.append(
                    McpIssue(
                        server_name=outcome.connection.server_name,
                        stage=McpStartupStage.REGISTER_TOOLS,
                        message=(
                            f"MCP Server {outcome.connection.server_name} "
                            f"的工具 {remote_tool.name} 名称无效，已跳过"
                        ),
                    )
                )
                continue

            registered_name = adapter.definition.name
            if registry.get(registered_name) is not None:
                issues.append(
                    McpIssue(
                        server_name=outcome.connection.server_name,
                        stage=McpStartupStage.REGISTER_TOOLS,
                        message=self._safe_message(
                            f"MCP 工具 {registered_name} 与已有工具重名，"
                            "已跳过"
                        ),
                    )
                )
                continue
            try:
                registry.register(adapter, source=ToolSource.MCP)
            except ValueError:
                issues.append(
                    McpIssue(
                        server_name=outcome.connection.server_name,
                        stage=McpStartupStage.REGISTER_TOOLS,
                        message=self._safe_message(
                            f"MCP 工具 {registered_name} 的定义或输入 "
                            "Schema 无效，已跳过"
                        ),
                    )
                )
                continue
            registered.append(registered_name)
        return registered, issues

    async def start(
            self,
            registry: ToolRegistry,
    ) -> McpStartupReport:
        """连接配置中的所有 MCP Server，并把可用工具注册到 MyCode。

        多台 Server 会同时连接和获取工具列表。获取完成后，再按照 Server 名称依次注册工具。某台 Server 或某个工具失败时会跳过，不影响其他 Server。

        Args:
            registry: MyCode 的工具表，MCP 工具将注册到这里。

        Returns:
            启动结果，包括仍保持连接的 Server 名、成功注册的工具名，
            以及连接或注册过程中遇到的问题。

        """

        # 上锁保证启动和关闭不能同时执行；一次启动没结束前，另一次启动或关闭只能等待
        async with self._state_lock:
            if self._state is not _ManagerState.NEW:
                raise RuntimeError("MCP 管理器不能重复启动")
            # 标记为正在启动
            self._state = _ManagerState.STARTING
            # 根据self._configs，为每个MCP Server创建个任务
            tasks = [
                asyncio.create_task(self._discover_one(config))
                for config in self._configs
            ]
            # 等待所有Server完成
            outcomes: list[_DiscoveryOutcome] = []
            try:
                if tasks:
                    outcomes = list(await asyncio.gather(*tasks))
            except asyncio.CancelledError:
                for task in tasks:
                    task.cancel()
                # 等所有已取消的任务完成清理，忽略它们最终抛出的异常
                await asyncio.gather(*tasks, return_exceptions=True)
                # 将MCP 服务器状态设置为已关闭
                self._state = _ManagerState.CLOSED
                raise

            # 把连接失败、初始化失败或获取工具失败的问题收集起来
            issues = [
                outcome.issue
                for outcome in outcomes
                if outcome.issue is not None
            ]
            # 按 Server 名注册工具
            registered_tools: list[str] = []
            # 遍历每个Server的连接及工具枚举结果
            for outcome in sorted(
                    outcomes,
                    key=lambda item: item.connection.server_name,
            ):
                if outcome.issue is not None:
                    continue
                registered, tool_issues = self._register_server_tools(
                    outcome,
                    registry,
                )
                # 将当前MCP Server成功注册的工具放到已注册工具中
                registered_tools.extend(registered)
                # 将当前MCP Server注册过程中发现的问题放到总问题列表中
                issues.extend(tool_issues)
                # 获取当前MCP Server的名字
                server_name = outcome.connection.server_name
                if registered:
                    # 有至少一个成功注册的工具，则当前的MCP连接放到 self._connections中
                    self._connections[server_name] = outcome.connection
                    continue
                # 所有工具均注册失败，则关闭当前MCP的连接
                closed = await self._close_quietly(outcome.connection)
                # 关闭连接失败
                if not closed:
                    issues.append(
                        McpIssue(
                            server_name=server_name,
                            stage=McpStartupStage.CLOSE,
                            message=(
                                f"MCP Server {server_name} 没有可注册工具，"
                                "且关闭连接失败"
                            ),
                        )
                    )
            # 设置MCP管理器的状态为启动流程已经结束；可用 Server 已保存，合法工具已完成注册
            self._state = _ManagerState.STARTED
            # 返回MCP启动后仍可用的连接、工具和局部问题汇总
            return McpStartupReport(
                connected_servers=tuple(sorted(self._connections)),
                registered_tools=tuple(registered_tools),
                issues=tuple(issues),

            )

    async def aclose(self) -> McpCloseReport:
        """依次关闭当前保存的所有 MCP Server 连接。

        一个 Server 关闭失败时会记录问题，并继续关闭其他 Server。
        第一次关闭的结果会被保存；以后重复调用时直接返回同一份报告，
        不会再次关闭连接。

        Returns:
            关闭成功的 Server 名，以及关闭失败的 Server 和错误说明。
        """
        async with self._state_lock:
            #
            if self._close_report is not None:
                return self._close_report
            connections = tuple(sorted(self._connections.items()))
            # 清除self._connections字典
            self._connections.clear()
            # 记录关闭成功的Server
            closed: list[str] = []
            # 记录关闭失败的Server及失败原因
            issues: list[McpIssue] = []
            for server_name, connection in connections:
                try:
                    # 关闭连接
                    await connection.aclose()
                except Exception:
                    issues.append(
                        McpIssue(
                            server_name=server_name,
                            stage=McpStartupStage.CLOSE,
                            message=self._safe_message(
                                f"MCP Server {server_name} 关闭失败"
                            ),
                        )
                    )
                else:
                    closed.append(server_name)
            self._state = _ManagerState.CLOSED
            self._close_report = McpCloseReport(
                closed_servers=tuple(closed),
                issues=tuple(issues),
            )
            return self._close_report
