"""跨平台、有输出上限的非交互式 Shell 命令执行。"""

from __future__ import annotations

import asyncio
import os
import subprocess
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

from mycode.models.json_types import JsonValue
from mycode.models.tools import ToolAccess, ToolDefinition, ToolErrorCode
from mycode.tools.base import ToolContext, ToolOutput
from mycode.tools.processes import terminate_process_tree


_EXECUTE_COMMAND = ToolDefinition(
    name="execute_command",
    description=(
        "仅在没有合适专用工具时，使用系统 Shell 在工作区中执行一条非交互式命令。"
    ),
    input_schema={
        "type": "object",
        "properties": {"command": {"type": "string", "minLength": 1}},
        "required": ["command"],
        "additionalProperties": False,
    },
    # Shell 能产生任意副作用，因此无论具体命令文本看起来是否只读，都必须
    # 归为 WRITE，并在 Plan 模式中统一拦截。
    access=ToolAccess.WRITE,
)


class _CommandCapture:
    """收集一条命令产生的完整 stdout 和 stderr。

    两个读取线程会同时调用 ``add``。锁只用于保护两个缓冲区和计数，避免
    并发写入破坏内容；结果大小控制由后续上下文管理器负责。
    """

    def __init__(self) -> None:
        # 命令标准输出按实际到达顺序追加到这里。
        self.stdout = bytearray()
        # 命令错误输出按实际到达顺序追加到这里。
        self.stderr = bytearray()
        self._lock = threading.Lock()

    def add(self, channel: str, chunk: bytes) -> None:
        """把一次管道读取追加到对应输出缓冲区。"""

        with self._lock:
            if channel == "stdout":
                self.stdout.extend(chunk)
            else:
                self.stderr.extend(chunk)

#在指定工作区中启动一条非交互式 Shell 命令，捕获正常输出和错误输出，并为后续取消整个进程树做好准备。
def _start_shell_process(command: str, workspace: Path) -> subprocess.Popen[bytes]:
    """在指定工作区启动一条非交互式 Shell 命令。

    命令的标准输入被关闭，标准输出和错误输出通过管道交给调用方读取。
    新进程会进入独立的进程组，方便取消时终止它及其子进程。

    Args:
        command: 需要交给 Shell 执行的命令文本。
        workspace: 命令使用的工作目录。

    Returns:
        用于读取输出、等待结束或终止命令的进程对象。
    """
    #定义保存操作系统专用启动参数的字典。
    options: dict[str, object]

    #判断当前是否为Windows
    if os.name == "nt":
        # 是Windows  启动命令时，为它创建一个新的进程组。
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        # 不是的话，让新进程进入一个独立的会话和进程组。
        options = {"start_new_session": True}

    #在 workspace 目录中，通过 Shell 启动 command；不允许它读取用户输入；把正常输出和错误输出交给程序捕获；让它进入独立进程组；最后把进程控制对象返回给调用方。
    return subprocess.Popen(
        command,
        shell=True,
        cwd=workspace,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **options,
    )


def _drain_pipe(
    #要读取的输出管道
    pipe: BinaryIO | None,
    capture: _CommandCapture,
    channel: str,
) -> None:
    """持续读取命令的一个输出管道，并保存全部字节。

    函数每次最多读取 64 KiB，直到管道关闭或没有更多数据，最后关闭管道。

    Args:
        pipe: 命令的 stdout 或 stderr 管道；为 None 时直接返回。
        capture: 用于保存命令输出的收集对象。
        channel: 输出类型，只能是 ``stdout`` 或 ``stderr``。
    """
    if pipe is None:
        return
    try:
        while True:
            #每次最多读取64 KiB
            chunk = pipe.read(64 * 1024)

            if not chunk:
                break

            # 即使内存预算耗尽也要继续排空管道，不能因为输出即将被截断，就让子进程因管道写满而死锁。
            capture.add(channel, chunk)
    finally:
        pipe.close()


def _wait_and_capture(
    process: subprocess.Popen[bytes],
    capture: _CommandCapture,
) -> int:
    """
    等待命令结束，同时收集命令的标准输出和错误输出

    函数启动两个线程，分别读取 stdout 和 stderr，避免其中一个管道
    因无人读取而写满。命令结束后，函数等待两个线程读完剩余内容。
    读取到的内容直接写入传入的 capture 对象

    Args:
        process: 已经启动的 Shell 进程对象，提供 stdout、stderr 和退出状态。
        capture: 保存命令输出的对象。函数执行结束后，其 stdout 和 stderr属性中分别保存命令产生的标准输出和错误输出。

    Returns:
        命令结束时返回的退出码。通常 0 表示成功，非 0 表示命令执行失败
    """

    #创建stdout读取线程，但是未执行
    stdout_reader = threading.Thread(
        target=_drain_pipe,
        args=(process.stdout, capture, "stdout"),
        #表示是守护线程  守护线程不会单独阻止整个 Python 程序退出。
        daemon=True,
    )

    #创建stderr读取线程
    stderr_reader = threading.Thread(
        target=_drain_pipe,
        args=(process.stderr, capture, "stderr"),
        daemon=True,
    )

    #开启两个读取线程
    stdout_reader.start()
    stderr_reader.start()

    #等待命令结束
    exit_code = process.wait()

    #将两个管道的剩余内容排空
    stdout_reader.join()
    stderr_reader.join()
    return exit_code

def _command_content(stdout: bytes, stderr: bytes) -> str:
    """把命令的标准输出和错误输出整理成一段文本

    函数分别使用 UTF-8 解码 stdout 和 stderr。遇到无法解码的字节时，
    使用替换字符代替，避免整个工具结果解码失败。返回文本使用标题区分两种输出，后续会作为工具执行结果交给模型。

    Args:
        stdout: 从命令标准输出管道读取到的全部字节。
        stderr: 从命令错误输出管道读取到的全部字节。

    Returns:
        包含 stdout 和 stderr 两个部分的字符串。
    """
    # 命令输出不强制要求 UTF-8；非法字节使用替换字符，保证回灌内容始终是
    # 合法文本，同时保留 stdout/stderr 的明确边界。
    return (
        f"stdout:\n{stdout.decode('utf-8', errors='replace')}\n"
        f"stderr:\n{stderr.decode('utf-8', errors='replace')}"
    )


class ExecuteCommandTool:
    """在工作区执行非交互 Shell 命令并返回完整文本输出。"""

    @property
    def definition(self) -> ToolDefinition:
        return _EXECUTE_COMMAND

    #接收模型提供的Shell命令，在工作区中启动命令，异步等待执行、收集受限的stdout和stderr，处理用户取消，最后返回结构化工具结果。
    async def execute(
        self,
        arguments: Mapping[str, JsonValue],
        context: ToolContext,
    ) -> ToolOutput:
        """在工作区中执行模型提交的 Shell 命令，并收集命令输出。

        函数在线程中启动并等待 Shell 进程，避免阻塞 Agent 的事件循环。命令执行期间会同时读取 stdout 和 stderr。命令正常结束后，根据
        退出码返回成功或失败结果，并附带完整输出、退出码和输出字节数。

        如果任务被取消，函数会先终止 Shell 进程及其子进程，等待读取线程结束，然后继续向调用方抛出取消异常。

        Args:
            arguments: 已通过参数校验的工具参数，其中 command 是模型提交的Shell 命令文本
            context: 本次工具调用使用的环境信息，其中 workspace_root 是命令的工作目录。

        Returns:
            命令退出码为 0 时返回成功结果；启动失败或退出码非 0 时返回失败结果。结果中包含 stdout、stderr、退出码和两种输出的字节数。
        """

        #拿到模型传来的命令
        command = str(arguments["command"])

        #创建命令启动任务
        start_shell_task = asyncio.create_task(
            asyncio.to_thread(
                _start_shell_process,
                command,
                context.workspace_root,
            )
        )
        try:
            # 取消 execute() 时不要取消启动任务。这样即使取消发生在 Shell启动期间，下面的异常处理仍能拿到进程对象并关闭它
            process = await asyncio.shield(start_shell_task)
        except asyncio.CancelledError:
            process = await asyncio.shield(start_shell_task)
            await asyncio.to_thread(terminate_process_tree, process)
            raise
        except OSError:
            return ToolOutput.fail(
                ToolErrorCode.IO_ERROR,
                "命令无法启动",
            )

        # 创建stdout和stderr捕获器
        capture = _CommandCapture()

        #创建等待和捕获任务
        wait_task = asyncio.create_task(
            asyncio.to_thread(_wait_and_capture, process, capture)
        )
        termination_error: str | None = None
        try:
            # 工作线程并发排空两个管道，事件循环则继续响应会话取消与超时。
            exit_code = await asyncio.shield(wait_task)
        except asyncio.CancelledError:
            termination_error = await asyncio.to_thread(
                terminate_process_tree,
                process,
            )
            # 在继续抛出取消异常前先回收读取线程，避免残留进程以及
            # 事件循环关闭后的管道警告。
            try:
                await asyncio.wait_for(
                    asyncio.shield(wait_task),
                    timeout=5,
                )
            except TimeoutError:
                pass
            raise

        content = _command_content(capture.stdout, capture.stderr)
        stdout_size = len(capture.stdout)
        stderr_size = len(capture.stderr)
        original_size = stdout_size + stderr_size
        metadata = {
            "exit_code": exit_code,
            "stdout_size_bytes": stdout_size,
            "stderr_size_bytes": stderr_size,
        }
        if termination_error is not None:
            metadata["termination_error"] = termination_error

        if exit_code == 0:
            #正常结束
            return ToolOutput.ok(
                content,
                metadata=metadata,
                original_size_bytes=original_size,
            )
        return ToolOutput.fail(
            ToolErrorCode.COMMAND_FAILED,
            f"命令以状态码 {exit_code} 退出",
            content=content,
            metadata=metadata,
            original_size_bytes=original_size,
        )
