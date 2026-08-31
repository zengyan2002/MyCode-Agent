"""执行目录型 Skill 在 tool.json 中声明的专属脚本。"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from collections.abc import Mapping

from mycode.models.json_types import JsonValue
from mycode.models.tools import ToolDefinition, ToolErrorCode
from mycode.models.skills import SkillToolSpec
from mycode.tools.base import ToolContext, ToolOutput
from mycode.tools.processes import terminate_async_process_tree

_READ_CHUNK_BYTES = 64 * 1024
_STDERR_DIAGNOSTIC_BYTES = 64 * 1024
_SAFE_ENVIRONMENT_NAMES = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "LANG",
    "LC_ALL",
)


class _OutputLimitExceeded(Exception):
    """表示脚本 stdout 已经超过 tool.json 声明的字节上限。"""


class SkillSubprocessTool:
    """把一个 SkillToolSpec 变成可以由 ToolExecutor 调用的工具。

    SkillService 为每个专属工具创建一个实例。execute 不经过 Shell，
    参数通过 stdin JSON 传入，结果只接受 stdout 中的单个 JSON 值。
    """

    def __init__(self, spec: SkillToolSpec) -> None:
        """保存已经通过 Parser 校验的专属工具定义。

        Args:
            spec: 包含命令、Schema、读写类别和输出限制的 SkillToolSpec。
        """

        # Parser 已确认脚本真实路径留在 Skill 根目录内。
        self._spec = spec
        # 注册表和 Provider 使用的稳定工具定义。
        self._definition = ToolDefinition(
            name=spec.name,
            description=spec.description,
            input_schema=spec.input_schema,
            access=spec.access,
        )

    @property
    def definition(self) -> ToolDefinition:
        """返回发送给模型的工具定义。

        Returns:
            从 tool.json 生成的名称、说明、Schema 和读写类别。
        """

        return self._definition

    async def execute(
        self,
        arguments: Mapping[str, JsonValue],
        context: ToolContext,
    ) -> ToolOutput:
        """启动专属脚本，并按 JSON stdin/stdout 协议返回结果。

        Args:
            arguments: 已由 ToolRegistry 按 inputSchema 校验的模型参数。
            context: 提供当前工作区根目录；子进程会把它作为 cwd。

        Returns:
            退出码为 0 且 stdout 是合法 JSON 时返回成功；启动失败、非零
            退出、无效 JSON 或输出超限时返回带明确错误码的 ToolOutput。

        Raises:
            asyncio.CancelledError: 用户取消或 ToolExecutor 超时时，在终止
                整个子进程树后继续向上抛出。
        """

        command = (
            self._spec.command[0],
            str(self._spec.entry_path),
            *self._spec.command[2:],
        )
        environment = self._minimal_environment()
        environment["MYCODE_SKILL_DIR"] = str(self._spec.skill_root)
        options: dict[str, object]
        if os.name == "nt":
            options = {
                "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
            }
        else:
            options = {"start_new_session": True}
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=context.workspace_root,
                env=environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **options,
            )
        except OSError:
            return ToolOutput.fail(
                ToolErrorCode.IO_ERROR,
                "Skill 专属工具无法启动",
            )

        input_bytes = json.dumps(
            dict(arguments),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        stdin_task = asyncio.create_task(
            self._write_stdin(process, input_bytes)
        )
        stdout_task = asyncio.create_task(
            self._read_stdout(process)
        )
        stderr_task = asyncio.create_task(
            self._read_stderr(process)
        )
        wait_task = asyncio.create_task(process.wait())
        tasks = (stdin_task, stdout_task, stderr_task, wait_task)
        try:
            _, stdout, stderr_result, exit_code = await asyncio.gather(
                *tasks
            )
        except _OutputLimitExceeded:
            await terminate_async_process_tree(process)
            await self._settle_tasks(
                process,
                tasks,
                drain_remaining_stdout=True,
            )
            return ToolOutput.fail(
                ToolErrorCode.IO_ERROR,
                (
                    "Skill 专属工具输出超过 "
                    f"{self._spec.max_output_bytes} 字节限制"
                ),
            )
        except asyncio.CancelledError:
            await asyncio.shield(terminate_async_process_tree(process))
            await asyncio.shield(
                self._settle_tasks(
                    process,
                    tasks,
                    drain_remaining_stdout=False,
                )
            )
            raise

        stderr, stderr_size = stderr_result
        stderr_text = stderr.decode("utf-8", errors="replace")
        metadata = {
            "exit_code": exit_code,
            "stderr_size_bytes": stderr_size,
        }
        if exit_code != 0:
            message = f"Skill 专属工具以状态码 {exit_code} 退出"
            if stderr_text:
                message += f"：{stderr_text}"
            return ToolOutput.fail(
                ToolErrorCode.COMMAND_FAILED,
                message,
                metadata=metadata,
            )
        try:
            parsed = json.loads(stdout.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return ToolOutput.fail(
                ToolErrorCode.COMMAND_FAILED,
                "Skill 专属工具的标准输出不是有效 JSON",
                metadata=metadata,
            )
        content = json.dumps(
            parsed,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return ToolOutput.ok(
            content,
            metadata=metadata,
            original_size_bytes=len(stdout),
        )

    def _minimal_environment(self) -> dict[str, str]:
        """复制运行脚本所需的常见系统环境变量。

        Returns:
            不包含 MyCode secrets 的新环境字典。MYCODE_SKILL_DIR 由 execute
            在返回后单独加入。
        """

        return {
            name: os.environ[name]
            for name in _SAFE_ENVIRONMENT_NAMES
            if name in os.environ
        }

    async def _write_stdin(
        self,
        process: asyncio.subprocess.Process,
        payload: bytes,
    ) -> None:
        """把一个 JSON 值写入脚本 stdin，然后关闭输入。

        Args:
            process: 当前正在运行的 Skill 子进程。
            payload: 已编码成 UTF-8 的 JSON 参数。

        Returns:
            None。数据刷新到底层管道并关闭 stdin 后结束。
        """

        if process.stdin is None:
            return
        process.stdin.write(payload)
        await process.stdin.drain()
        process.stdin.close()
        await process.stdin.wait_closed()

    async def _read_stdout(
        self,
        process: asyncio.subprocess.Process,
    ) -> bytes:
        """读取 stdout，并在超过工具上限时立刻停止。

        Args:
            process: 当前正在运行的 Skill 子进程。

        Returns:
            未超过限制的完整 stdout 字节。

        Raises:
            _OutputLimitExceeded: 累计字节数超过 maxOutputBytes。
        """

        if process.stdout is None:
            return b""
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await process.stdout.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > self._spec.max_output_bytes:
                raise _OutputLimitExceeded
            chunks.append(chunk)
        return b"".join(chunks)

    async def _read_stderr(
        self,
        process: asyncio.subprocess.Process,
    ) -> tuple[bytes, int]:
        """排空 stderr，只保留开头一小段用于失败诊断。

        Args:
            process: 当前正在运行的 Skill 子进程。

        Returns:
            最多 64 KiB 的诊断字节，以及脚本实际写出的总字节数。
        """

        if process.stderr is None:
            return b"", 0
        kept = bytearray()
        total = 0
        while True:
            chunk = await process.stderr.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            remaining = _STDERR_DIAGNOSTIC_BYTES - len(kept)
            if remaining > 0:
                kept.extend(chunk[:remaining])
        return bytes(kept), total

    async def _settle_tasks(
        self,
        process: asyncio.subprocess.Process,
        tasks: tuple[asyncio.Task[object], ...],
        *,
        drain_remaining_stdout: bool,
    ) -> None:
        """在进程终止后排空管道并回收交换任务。

        Args:
            process: 已经收到终止请求的 Skill 子进程。
            tasks: stdin 写入、stdout/stderr 读取和进程等待任务。
            drain_remaining_stdout: stdout 读取任务因超限退出时为 True，
                此时函数继续读到 EOF，确保 Windows 管道 transport 正常关闭。

        Returns:
            None。全部任务已经结束，异常被清理路径消化。
        """

        stdin_task, stdout_task, _, _ = tasks
        if not stdin_task.done():
            stdin_task.cancel()
        if (
            drain_remaining_stdout
            and stdout_task.done()
            and process.stdout is not None
        ):
            await process.stdout.read()
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=3,
            )
        except TimeoutError:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
