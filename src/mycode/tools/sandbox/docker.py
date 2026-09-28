"""用本机 Docker 执行命令；创建前记名，结束后核实容器已删除。"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import threading
from dataclasses import asdict, replace
from pathlib import Path

from mycode.errors import ConfigError, redact_secrets
from mycode.models.config import AppConfig, CommandSandboxSettings, SecretValue
from mycode.models.operations import OperationState
from mycode.models.tools import ToolErrorCode
from mycode.persistence.operations import _process_alive
from mycode.tools.base import ToolContext, ToolOutput
from mycode.tools.processes import terminate_process_tree
from mycode.tools.sandbox.records import (
    SandboxRunRecord, load_records, record_directory, remove_record, save_record,
)
from mycode.tools.sandbox.snapshot import SnapshotError, prepare_snapshot


PROFILE_ENV = "MYCODE_COMMAND_SANDBOX"
RUNTIME_NOTICE = (
    "内置 execute_command 在本机 Docker Linux 沙箱的 /workspace 中使用 /bin/sh 执行。"
    "每次复制宿主当前工作区的普通文件，排除 .env、配置、.git、.mycode、依赖缓存和已知密钥。"
    "容器断网，依赖必须预装；命令创建或修改的文件仅在本次容器存在，不回写宿主，也不跨调用保留。"
    "持久修改源码请使用 write_file/edit_file；这些文件工具仍使用宿主路径。"
    "Skill、Hook 和 MCP 进程未被这项 Shell 沙箱设置隔离。"
)
OUTPUT_LIMIT = 16 * 1024 * 1024


class DockerError(RuntimeError):
    """Docker 管理请求失败，不能据此断言任务没有启动。"""


async def _settle(task: asyncio.Task):
    """等待有自身截止时间的清理/启动任务，重复取消不丢失它的结果。"""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


def export_profile(config: AppConfig) -> str:
    """给团队 Host 传递已冻结的非敏感设置，不传凭证。"""
    return json.dumps({"settings": asdict(config.sandbox),
        "protected_paths": [str(p) for p in config.loaded_config_paths]}, ensure_ascii=False)


def inherit_profile(config: AppConfig, environment: dict[str, str]) -> AppConfig:
    """仅由团队 Host 入口调用，防止成员重新加载配置后退回本地执行。"""
    raw = environment.get(PROFILE_ENV)
    if not raw:
        return config
    try:
        data = json.loads(raw)
        values = dict(data["settings"])
        values["exclude"] = tuple(values.get("exclude", ()))
        settings = CommandSandboxSettings(**values)
        paths = tuple(Path(p) for p in data["protected_paths"])
        if any(not p.is_absolute() for p in paths):
            raise ValueError("配置保护路径不是绝对路径")
    except (ValueError, TypeError, KeyError) as exc:
        raise ConfigError("团队 Host 的沙箱设置无效") from exc
    return replace(config, sandbox=settings,
                   loaded_config_paths=tuple(dict.fromkeys((*config.loaded_config_paths, *paths))))


class _Capture:
    """同时排空两条管道，超过总上限后只计数，不再占用内存。"""

    def __init__(self) -> None:
        self.data = {"stdout": bytearray(), "stderr": bytearray()}
        self.sizes = {"stdout": 0, "stderr": 0}
        self.lock = threading.Lock()

    def drain(self, pipe, channel: str) -> None:
        try:
            while chunk := pipe.read(65536):
                with self.lock:
                    self.sizes[channel] += len(chunk)
                    remaining = OUTPUT_LIMIT - sum(len(v) for v in self.data.values())
                    self.data[channel].extend(chunk[:max(0, remaining)])
        finally:
            pipe.close()

    def wait(self, process: subprocess.Popen) -> int:
        threads = [threading.Thread(target=self.drain, args=(pipe, channel), daemon=True)
                   for pipe, channel in ((process.stdout, "stdout"), (process.stderr, "stderr"))]
        for thread in threads:
            thread.start()
        result = process.wait()
        for thread in threads:
            thread.join()
        return result


class DockerCommandRunner:
    """保存可信启动设置；各次调用的容器和工作区记录相互独立。"""

    def __init__(self, settings: CommandSandboxSettings, control_root: Path,
                 protected_paths: tuple[Path, ...] = (), secrets: tuple[SecretValue, ...] = ()) -> None:
        self.settings = settings
        self.control_root = control_root.resolve()
        self.protected_paths = protected_paths
        self.secrets = secrets
        self.endpoint = ""
        self.image_id = ""
        self.executable = "docker"
        self._initialize_lock = asyncio.Lock()

    async def _manage(self, args: list[str], *, endpoint: str | None = None) -> str:
        prefix = [self.executable]
        address = self.endpoint if endpoint is None else endpoint
        if address:
            prefix += ["--host", address]
        task = asyncio.create_task(asyncio.to_thread(subprocess.run,
            [*prefix, *args], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=5, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await _settle(task)
            except (OSError, subprocess.SubprocessError):
                pass
            raise
        except (OSError, subprocess.SubprocessError) as exc:
            raise DockerError("Docker 管理请求未能完成") from exc
        if result.returncode:
            raise DockerError(f"Docker {args[0]} 请求失败")
        return result.stdout.decode("utf-8")

    async def initialize(self) -> None:
        """启动时验证本机 Linux 引擎与镜像，清理确认已退出所有者的容器。"""
        async with self._initialize_lock:
            if not self.endpoint:
                executable = shutil.which("docker")
                if not executable:
                    raise DockerError("未安装 Docker CLI，沙箱不会退回宿主执行")
                self.executable = executable
                if os.environ.get("DOCKER_HOST") and not os.environ.get("DOCKER_CONTEXT"):
                    endpoint = os.environ["DOCKER_HOST"]
                else:
                    context = os.environ.get("DOCKER_CONTEXT") or (await self._manage(["context", "show"])).strip()
                    data = json.loads(await self._manage(["context", "inspect", context]))
                    endpoint = data[0]["Endpoints"]["docker"]["Host"]
                if not endpoint.startswith(("unix:///", "npipe:////./pipe/")):
                    raise DockerError("第一版只支持本机 Docker endpoint")
                self.endpoint = endpoint
            info = json.loads(await self._manage(["info", "--format", "{{json .}}"] ))
            if info.get("OSType") != "linux":
                raise DockerError("沙箱要求 Docker Linux 引擎")
            if not self.image_id:
                images = json.loads(await self._manage(["image", "inspect", self.settings.image]))
                image = images[0]
                config = image["Config"]
                if ((config.get("Labels") or {}).get("mycode.sandbox.protocol") != "v1"
                        or config.get("Volumes")):
                    raise DockerError("镜像不符合 MyCode 沙箱 v1 协议或声明了额外卷")
                self.image_id = image["Id"]
            await self.recover()

    async def _inspect(self, record: SandboxRunRecord) -> dict | None:
        # 用成功的 list 区分“确实不存在”和“引擎不可达”，不解析本地化 stderr。
        listed = await self._manage(["container", "ls", "-a", "--filter",
            f"name=^/{record.container_name}$", "--format", "{{.ID}}"], endpoint=record.endpoint)
        if not listed.strip():
            return None
        data = json.loads(await self._manage(["container", "inspect", record.container_name],
                                            endpoint=record.endpoint))[0]
        labels = data["Config"].get("Labels") or {}
        if (any(labels.get(k) != v for k, v in record.labels.items())
                or (record.container_id and data["Id"] != record.container_id)):
            raise DockerError("同名容器归属不匹配，拒绝删除")
        return data

    async def cleanup(self, record: SandboxRunRecord) -> None:
        """只删除经过身份核对的容器，确认不存在后再移除输入。"""
        record_directory(self.control_root, record)
        if not record.endpoint.startswith(("unix:///", "npipe:////./pipe/")):
            raise DockerError("记录中的 Docker endpoint 不是本机地址")
        try:
            async with asyncio.timeout(10):
                data = await self._inspect(record)
                if data is None and record.stage == "creating":
                    raise DockerError("创建请求结果未知，暂时查不到容器不能证明请求已结束")
                if data is not None:
                    await self._manage(["container", "rm", "-f", data["Id"]], endpoint=record.endpoint)
                if await self._inspect(record) is not None:
                    raise DockerError("容器仍然存在")
                remove_record(self.control_root, record)
        except (DockerError, OSError, ValueError, TimeoutError):
            record.cleanup_error = f"未能确认容器和工作副本已清理，沙箱 ID：{record.sandbox_id}"
            save_record(self.control_root, record)
            raise DockerError(record.cleanup_error)

    async def recover(self) -> None:
        """不接管活跃调用；旧进程退出后仅清理外部资源，不重跑业务操作。"""
        for record in load_records(self.control_root):
            if record.cleanup_error and record.owner_pid == os.getpid():
                await self.cleanup(record)
            elif _process_alive(record.owner_pid) is False:
                await self.cleanup(record)

    def _create_args(self, command: str, record: SandboxRunRecord, input_path: Path) -> list[str]:
        s = self.settings
        args = ["create", "--name", record.container_name, "--pull=never", "--read-only",
            "--network=none", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--user=10001:10001", "--init", "--restart=no", "--no-healthcheck", "--log-driver=none",
            "--cpus", str(s.cpus), "--memory", f"{s.memory_mb}m", "--memory-swap", f"{s.memory_mb}m",
            "--pids-limit", str(s.pids_limit), "--workdir=/workspace",
            "--tmpfs", f"/workspace:rw,exec,nosuid,nodev,size={s.workspace_mb}m,uid=10001,gid=10001,mode=0700",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m,uid=10001,gid=10001,mode=0700",
            "--mount", f"type=bind,source={input_path},target=/input,readonly",
            "--entrypoint=python"]
        for key, value in record.labels.items():
            args += ["--label", f"{key}={value}"]
        return [*args, record.image_id, "/opt/mycode-sandbox/bootstrap.py", command]

    def _start(self, record: SandboxRunRecord) -> subprocess.Popen:
        options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
                   if os.name == "nt" else {"start_new_session": True})
        return subprocess.Popen([self.executable, "--host", record.endpoint,
            "start", "--attach", record.container_name],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=False, **options)

    async def run(self, command: str, context: ToolContext) -> ToolOutput:
        operation = context.operation_record
        if operation is None or operation.state is not OperationState.RUNNING:
            return ToolOutput.fail(ToolErrorCode.BLOCKED, "Docker 命令缺少已领取的操作记录")
        try:
            await self.initialize()
        except DockerError as exc:
            return ToolOutput.fail(ToolErrorCode.BLOCKED, f"Docker 沙箱不可用：{exc}；未执行宿主命令")
        except (OSError, ValueError, KeyError) as exc:
            return ToolOutput.fail(ToolErrorCode.BLOCKED, f"Docker 沙箱不可用：{type(exc).__name__}；未执行宿主命令")
        workspace = context.workspace_root
        record = SandboxRunRecord.create(self.control_root, workspace, operation, self.image_id, self.endpoint)
        save_record(self.control_root, record)
        input_path = record_directory(self.control_root, record) / "input"
        cancel_copy = threading.Event()
        copy_task = asyncio.create_task(asyncio.to_thread(prepare_snapshot, workspace, input_path,
            self.settings, self.protected_paths, self.secrets, cancel_copy))
        try:
            snapshot = await asyncio.shield(copy_task)
        except asyncio.CancelledError:
            cancel_copy.set()
            try:
                await _settle(copy_task)
            except (OSError, SnapshotError):
                pass
            remove_record(self.control_root, record)
            raise
        except (OSError, SnapshotError) as exc:
            remove_record(self.control_root, record)
            reason = redact_secrets(str(exc), self.secrets)
            return ToolOutput.fail(ToolErrorCode.BLOCKED,
                f"工作副本准备失败，命令未启动：{reason}")
        # --mount 的 CSV 语法不能直接表示逗号路径，拒绝而不是误挂载。
        if "," in str(input_path):
            remove_record(self.control_root, record)
            return ToolOutput.fail(ToolErrorCode.BLOCKED, "Docker 输入副本路径暂不支持逗号")
        capture = _Capture()
        start_task = wait_task = None
        process = None
        state = None
        error = None
        cancelled = False
        try:
            record.stage = "creating"
            save_record(self.control_root, record)
            record.container_id = (await self._manage(self._create_args(command, record, input_path))).strip()
            record.stage = "created"
            save_record(self.control_root, record)
            start_task = asyncio.create_task(asyncio.to_thread(self._start, record))
            process = await asyncio.shield(start_task)
            record.stage = "running"
            save_record(self.control_root, record)
            wait_task = asyncio.create_task(asyncio.to_thread(capture.wait, process))
            await asyncio.shield(wait_task)
            data = await self._inspect(record)
            if data is None or data["State"]["Status"] != "exited":
                raise DockerError("未取得命令的最终退出状态")
            state = data["State"]
            record.stage = "exited"
            save_record(self.control_root, record)
        except asyncio.CancelledError:
            cancelled = True
        except (DockerError, OSError, ValueError, KeyError):
            error = "Docker 执行状态未能确认"
        finally:
            if start_task is not None and process is None:
                try:
                    process = await _settle(start_task)
                except OSError:
                    pass
            cleanup_task = asyncio.create_task(self.cleanup(record))
            try:
                await _settle(cleanup_task)
            except (DockerError, OSError, ValueError):
                error = "Docker 容器清理未能确认，请检查沙箱追踪记录"
            if process is not None:
                terminate_task = asyncio.create_task(asyncio.to_thread(terminate_process_tree, process))
                if await _settle(terminate_task):
                    error = "Docker 客户端进程未能完全退出"
            if wait_task is not None:
                async def finish_readers():
                    await asyncio.wait_for(asyncio.shield(wait_task), 5)
                try:
                    await _settle(asyncio.create_task(finish_readers()))
                except TimeoutError:
                    error = "Docker 输出管道尚未关闭"
        if cancelled or asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        metadata = {"backend": "docker", "sandbox_id": record.sandbox_id,
            "image_id": record.image_id, "container_workdir": "/workspace",
            "workspace_mode": "ephemeral_copy", "host_writeback": False,
            "cleanup_status": "failed" if record.cleanup_error else "removed",
            "excluded_count": snapshot.excluded_count,
            "excluded_paths": list(snapshot.excluded_relative_paths),
            "stdout_size_bytes": capture.sizes["stdout"], "stderr_size_bytes": capture.sizes["stderr"],
            "output_truncated": sum(capture.sizes.values()) > OUTPUT_LIMIT}
        content = "stdout:\n" + capture.data["stdout"].decode("utf-8", errors="replace")
        content += "\nstderr:\n" + capture.data["stderr"].decode("utf-8", errors="replace")
        content += "\n[Docker 临时副本；无网络；文件未回写宿主。]"
        if metadata["output_truncated"]:
            content += "\n[输出超过 16MiB，已截断。]"
        if error:
            return ToolOutput.fail(ToolErrorCode.IO_ERROR, error, content=content, metadata=metadata)
        assert state is not None
        metadata.update(exit_code=state["ExitCode"], oom_killed=state.get("OOMKilled", False))
        if state["ExitCode"] == 0 and not state.get("OOMKilled"):
            return ToolOutput.ok(content, metadata=metadata)
        return ToolOutput.fail(ToolErrorCode.COMMAND_FAILED,
            f"容器命令退出码 {state['ExitCode']}" + ("（内存超限）" if state.get("OOMKilled") else ""),
            content=content, metadata=metadata)
