"""终止普通命令工具和 Skill 工具启动的整个子进程树。"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess


def terminate_process_tree(
    process: subprocess.Popen[bytes],
) -> str | None:
    """终止同步 Popen 进程以及它启动的子进程。

    ExecuteCommandTool 在用户取消或全局超时时调用本函数。Windows 上先
    发送 CTRL_BREAK，再用 taskkill 结束整棵树；其他系统使用进程组信号。

    Args:
        process: 由命令工具启动、进入独立进程组的 Popen 对象。

    Returns:
        没有发现终止错误时返回 None；终止失败时返回不含命令参数的说明。
    """

    if process.poll() is not None:
        return None
    error: str | None = None
    try:
        if os.name == "nt":
            try:
                os.kill(process.pid, signal.CTRL_BREAK_EVENT)
                process.wait(timeout=0.5)
                return None
            except (OSError, subprocess.TimeoutExpired):
                pass
            completed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode != 0 and process.poll() is None:
                process.kill()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
    except (OSError, subprocess.SubprocessError) as exc:
        error = f"无法完全终止进程树：{type(exc).__name__}"
        try:
            process.kill()
        except OSError:
            pass
    return error


async def terminate_async_process_tree(
    process: asyncio.subprocess.Process,
) -> str | None:
    """终止 asyncio 子进程以及它启动的子进程。

    SkillSubprocessTool 在输出超限、超时或用户取消时调用。该函数会等待
    进程退出，避免事件循环关闭后残留管道或后台脚本。

    Args:
        process: 由 asyncio.create_subprocess_exec 启动并进入独立进程组
            的进程对象。

    Returns:
        进程树已经结束时返回 None；终止过程中出错时返回简短说明。
    """

    if process.returncode is not None:
        return None
    error: str | None = None
    try:
        if os.name == "nt":
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
                await asyncio.wait_for(asyncio.shield(process.wait()), 0.5)
                return None
            except (OSError, ProcessLookupError, TimeoutError):
                pass

            def taskkill() -> subprocess.CompletedProcess[bytes]:
                """调用 Windows taskkill 结束指定 PID 的整棵进程树。"""

                return subprocess.run(
                    [
                        "taskkill",
                        "/PID",
                        str(process.pid),
                        "/T",
                        "/F",
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                    check=False,
                    creationflags=getattr(
                        subprocess,
                        "CREATE_NO_WINDOW",
                        0,
                    ),
                )

            completed = await asyncio.to_thread(taskkill)
            if completed.returncode != 0 and process.returncode is None:
                process.kill()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(asyncio.shield(process.wait()), 2)
        except TimeoutError:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            await asyncio.wait_for(asyncio.shield(process.wait()), 2)
    except (OSError, ProcessLookupError, subprocess.SubprocessError) as exc:
        error = f"无法完全终止进程树：{type(exc).__name__}"
        try:
            process.kill()
            await asyncio.shield(process.wait())
        except (OSError, ProcessLookupError):
            pass
    return error
