"""执行 Hook 配置中的命令、提示、HTTP 和子 Agent 占位动作。"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path

import httpx

from mycode.constants import (
    HOOK_HTTP_TIMEOUT_SECONDS,
    HOOK_OUTPUT_LIMIT_CHARS,
)
from mycode.hooks.runtime import HookRunScope
from mycode.hooks.templates import expand_template
from mycode.models.hooks import (
    AgentHookAction,
    CommandHookAction,
    HookAction,
    HookActionResult,
    HookContext,
    HttpHookAction,
    PromptHookAction,
)
from mycode.tools.processes import terminate_async_process_tree


_LOGGER = logging.getLogger(__name__)


def _limited(value: str) -> str:
    """把动作输出限制到可回灌模型的固定长度。

    Args:
        value: command stdout、stderr 或 HTTP 响应正文。

    Returns:
        未超限时返回原文；超限时返回截断正文和明确的截断标记。
    """

    if len(value) <= HOOK_OUTPUT_LIMIT_CHARS:
        return value
    return value[:HOOK_OUTPUT_LIMIT_CHARS] + "\n[Hook 输出已截断]"


class HookActionRunner:
    """直接执行当前版本支持的四类 Hook 动作。

    CLI 为整个应用创建一个实例。命令在工作区执行，HTTP 客户端由该实例
    独立持有，提示动作写入调用方传入的 scope；它不会创建额外的动作工厂。

    Attributes:
        workspace_root: Shell 命令执行时使用的当前目录。
        _http: 复用连接并执行 Hook HTTP 请求的专用异步客户端。
        _closed: HTTP 客户端是否已经关闭。
    """

    def __init__(self, workspace_root: Path) -> None:
        """保存工作区并创建 Hook 专用 HTTP 客户端。

        Args:
            workspace_root: MyCode 启动时确定的项目根目录。

        Returns:
            None。HTTP 连接在后续动作中复用，应用退出时由 `close` 关闭。
        """

        self.workspace_root = workspace_root
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(HOOK_HTTP_TIMEOUT_SECONDS)
        )
        self._closed = False

    async def run(
        self,
        action: HookAction,
        context: HookContext,
        scope: HookRunScope,
    ) -> HookActionResult:
        """执行一条已经通过启动校验的动作。

        Args:
            action: 配置加载器生成的具体动作对象。
            context: 当前事件提供给模板的真实字段。
            scope: prompt 和 once 状态所属的主会话或 fork。

        Returns:
            Hook 引擎用于记录状态或生成拒绝原因的有限结果。
        """

        try:
            if isinstance(action, CommandHookAction):
                return await self._run_command(action, context)
            if isinstance(action, PromptHookAction):
                message = expand_template(action.message, context)
                if not message.strip():
                    return HookActionResult(False, error="提示词展开后为空")
                scope.instruction_manager.enqueue_hook_notification(message)
                return HookActionResult(True, output=message)
            if isinstance(action, HttpHookAction):
                return await self._run_http(action, context)
            assert isinstance(action, AgentHookAction)
            expand_template(action.prompt, context)
            _LOGGER.info("当前版本尚未接入子 Agent 运行")
            return HookActionResult(True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return HookActionResult(
                False,
                error=f"动作执行异常：{type(exc).__name__}",
            )

    async def _run_command(
        self,
        action: CommandHookAction,
        context: HookContext,
    ) -> HookActionResult:
        """启动平台 Shell，等待命令完成并返回受限输出。

        Args:
            action: 包含命令模板和可选超时的已校验动作。
            context: 展开命令模板时读取的当前事件数据。

        Returns:
            退出码为零时返回成功及 stdout；失败或超时时返回短原因和有限输出。
        """

        command = expand_template(action.command, context)
        options: dict[str, object] = {}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=self.workspace_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **options,
        )
        try:
            communicate = process.communicate()
            if action.timeout_seconds is None:
                stdout, stderr = await communicate
            else:
                stdout, stderr = await asyncio.wait_for(
                    communicate,
                    timeout=action.timeout_seconds,
                )
        except TimeoutError:
            await terminate_async_process_tree(process)
            return HookActionResult(False, error="命令执行超时")
        except asyncio.CancelledError:
            await asyncio.shield(terminate_async_process_tree(process))
            raise
        output = stdout.decode("utf-8", errors="replace").strip()
        error_output = stderr.decode("utf-8", errors="replace").strip()
        combined = _limited(output or error_output)
        if process.returncode != 0:
            return HookActionResult(
                False,
                output=combined,
                error=f"命令退出码为 {process.returncode}",
            )
        return HookActionResult(True, output=combined)

    async def _run_http(
        self,
        action: HttpHookAction,
        context: HookContext,
    ) -> HookActionResult:
        """展开 HTTP 模板并发送一次不重试的请求。

        Args:
            action: 包含 URL、方法、请求头和可选正文模板的已校验动作。
            context: 展开各 HTTP 模板时读取的当前事件数据。

        Returns:
            2xx 响应返回成功和有限正文；其他状态返回失败、状态码和有限正文。
        """

        response = await self._http.request(
            action.method,
            expand_template(action.url, context),
            headers={
                name: expand_template(value, context)
                for name, value in action.headers
            },
            content=(
                expand_template(action.body, context)
                if action.body is not None
                else None
            ),
        )
        body = _limited(response.text.strip())
        if not 200 <= response.status_code < 300:
            return HookActionResult(
                False,
                output=body,
                error=f"HTTP 请求返回 {response.status_code}",
            )
        return HookActionResult(True, output=body)

    async def close(self) -> None:
        """关闭 Hook 专用 HTTP 连接池。

        Returns:
            None。重复调用不会再次关闭客户端。
        """

        if self._closed:
            return
        self._closed = True
        await self._http.aclose()
