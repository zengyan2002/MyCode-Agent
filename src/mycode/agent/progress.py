"""比较当前 Turn 的近期工具结果，提醒模型停止重复调查。"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from enum import Enum

from mycode.models.messages import ToolCall
from mycode.models.tools import (
    ToolAccess, ToolErrorCode, ToolExecutionResult, ToolInvocation,
)


PROGRESS_NOTICE = (
    "最近连续的工具轮次重复得到了相同结果。请检查当前假设，避免原样重复最近的工具和参数，"
    "尝试新的文件、搜索范围或验证方式。如果证据已经足够，请直接完成回答。"
)
PROGRESS_REPLAN = (
    "当前工具路径已连续多轮重复得到相同结果。下一步请先重新规划：明确尚未解决的问题，"
    "判断旧路径为何没有推进，再选择新的证据来源、参数或工具。"
    "如果没有新的有效路径，请基于现有证据完成回答。"
)


class ProgressSignal(str, Enum):
    """说明本轮没有重复，还是出现了相同结果的一般重复或两动作交替。"""

    NONE = "none"
    REPEATED_ACTION = "repeated_action"
    OSCILLATION = "oscillation"


class ProgressSeverity(str, Enum):
    """决定下一次请求无需提示、轻量提醒或要求重新规划。"""

    NONE = "none"
    NOTICE = "notice"
    REPLAN = "replan"


@dataclass(frozen=True)
class ToolStepFingerprint:
    """保存一次工具调用的位置和结果摘要，不保留参数或结果正文。"""

    model_call_number: int
    call_index: int
    tool_name: str
    access: ToolAccess
    action_hash: str
    result_hash: str
    success: bool
    error_code: ToolErrorCode | None


@dataclass(frozen=True)
class ProgressDecision:
    """告诉 Runner 本轮重复情况、下次提示和是否首次需要显示警告。"""

    signal: ProgressSignal
    severity: ProgressSeverity
    no_progress_streak: int
    message: str | None
    emit_warning: bool


class ProgressMonitor:
    """保存一个 Turn 最近的工具摘要，累计整轮重复得到相同结果的次数。"""

    def __init__(
        self, *, max_steps: int, warn_after_rounds: int, replan_after_rounds: int,
    ) -> None:
        self._steps: deque[ToolStepFingerprint] = deque(maxlen=max_steps)
        self._no_progress_streak = 0
        self._warn_after_rounds = warn_after_rounds
        self._replan_after_rounds = replan_after_rounds

    @property
    def no_progress_streak(self) -> int:
        """返回连续整轮重复得到相同结果的次数。"""
        return self._no_progress_streak

    def observe_round(
        self,
        invocations: tuple[ToolInvocation, ...],
        results: tuple[ToolExecutionResult, ...],
    ) -> ProgressDecision:
        """按声明顺序观察正常结束的一批工具，返回下一次请求应使用的提醒。

        任意新动作、变化结果或成功写入都会清零本轮计数。失败结果也参与
        比较；Runner 不应把取消或异常收尾时补出的结果传入这里。
        """
        all_redundant = bool(invocations)
        for invocation, result in zip(invocations, results, strict=True):
            action_hash = action_fingerprint(invocation.call)
            result_hash = result_fingerprint(result)
            previous = next(
                (step for step in reversed(self._steps) if step.action_hash == action_hash),
                None,
            )
            if (
                previous is None
                or previous.result_hash != result_hash
                or (invocation.access is ToolAccess.WRITE and result.success)
            ):
                all_redundant = False
            self._steps.append(ToolStepFingerprint(
                model_call_number=invocation.model_call_number,
                call_index=invocation.call_index,
                tool_name=invocation.call.name,
                access=invocation.access,
                action_hash=action_hash,
                result_hash=result_hash,
                success=result.success,
                error_code=result.error_code,
            ))

        previous_streak = self._no_progress_streak
        self._no_progress_streak = previous_streak + 1 if all_redundant else 0
        signal = ProgressSignal.REPEATED_ACTION if all_redundant else ProgressSignal.NONE
        if all_redundant and len(self._steps) >= 4:
            a, b, repeated_a, repeated_b = tuple(self._steps)[-4:]
            if (
                a.action_hash != b.action_hash
                and a.action_hash == repeated_a.action_hash
                and b.action_hash == repeated_b.action_hash
                and a.result_hash == repeated_a.result_hash
                and b.result_hash == repeated_b.result_hash
            ):
                signal = ProgressSignal.OSCILLATION

        severity = ProgressSeverity.NONE
        message = None
        if self._no_progress_streak >= self._replan_after_rounds:
            severity, message = ProgressSeverity.REPLAN, PROGRESS_REPLAN
        elif self._no_progress_streak >= self._warn_after_rounds:
            severity, message = ProgressSeverity.NOTICE, PROGRESS_NOTICE
        return ProgressDecision(
            signal=signal,
            severity=severity,
            no_progress_streak=self._no_progress_streak,
            message=message,
            emit_warning=(previous_streak < self._replan_after_rounds <= self._no_progress_streak),
        )


def action_fingerprint(call: ToolCall) -> str:
    """对工具名和参数求摘要；调用 ID 和对象键顺序不影响结果。"""
    arguments = json.dumps(
        call.arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(f"{call.name}\0{arguments}".encode("utf-8")).hexdigest()


def result_fingerprint(result: ToolExecutionResult) -> str:
    """对执行结果求摘要，排除每次调用都会变化的 ID 和耗时。"""
    payload = {
        "tool_name": result.tool_name,
        "success": result.success,
        "content": result.content,
        "error_code": result.error_code.value if result.error_code is not None else None,
        "error_message": result.error_message,
        "timed_out": result.timed_out,
        "truncated": result.truncated,
        "original_size_bytes": result.original_size_bytes,
        "metadata": result.metadata,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
