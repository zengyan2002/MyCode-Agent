"""工具结果过长时，把完整内容保存到 artifact，并在对话中只保留文件路径和内容预览"""

from __future__ import annotations

from dataclasses import dataclass, replace

from mycode.constants import TOOL_RESULT_PREVIEW_CHARS
from mycode.context.artifacts import ArtifactRecord, ArtifactStore
from mycode.errors import ArtifactError
from mycode.models.tools import ToolExecutionResult


@dataclass(frozen=True)
class ToolResultSaveFailure:
    """记录一份没有保存成功的工具结果及失败原因。"""

    tool_call_id: str
    tool_name: str
    reason: str


@dataclass(frozen=True)
class ToolResultCompactionOutcome:
    """记录一批工具结果处理完成后，要写入对话的结果和本次落盘失败信息"""

    results: tuple[ToolExecutionResult, ...]
    failures: tuple[ToolResultSaveFailure, ...]


class ToolResultCompactor:
    """处理一次 Assistant 响应产生的全部工具结果

    AgentLoop 在把本轮工具结果写入对话记录前调用 compact_batch。这个类会
    检查单个结果和整批结果的字符数；超过上限时，把较大的完整正文保存到
    artifact，并在对话中只保留文件路径和内容预览。
    """

    def __init__(
        self,
        artifacts: ArtifactStore,
        *,
        result_threshold_chars: int,
        batch_threshold_chars: int,
    ) -> None:
        """设置工具结果的保存位置和字符数上限。

        这里只保存配置，不会立即处理或写入任何工具结果。

        Args:
            artifacts: 负责把完整工具结果保存到 artifact 文件的对象。
            result_threshold_chars: 单个结果超过这个字符数时，将完整正文保存到文件。
            batch_threshold_chars: 一批结果的正文总量超过这个字符数时，优先把较大的结果保存到文件。
        """
        if result_threshold_chars <= 0 or batch_threshold_chars <= 0:
            raise ValueError("工具结果压缩阈值必须为正数")
        self._artifacts = artifacts
        self._result_threshold_chars = result_threshold_chars
        self._batch_threshold_chars = batch_threshold_chars

    def _preview(
        self,
        result: ToolExecutionResult,
        record: ArtifactRecord,
        safe_content: str,
    ) -> str:
        """生成写入对话的 artifact 文件说明和正文预览。

        正文较短时完整显示；正文较长时只显示开头和结尾。

        Args:
            result: 本次工具执行结果，用于读取工具名称和调用 ID。
            record: 已保存 artifact 的文件路径和原始正文大小。
            safe_content: 替换敏感信息后的完整工具结果正文，用于截取预览。

        Returns:
            包含工具名称、调用 ID、原始大小、文件路径和正文预览的字符串。
    """

        header = "\n".join(
            (
                "[工具结果已保存到工作区]",
                f"工具：{result.tool_name}",
                f"调用 ID：{result.tool_call_id}",
                f"原始字符数：{record.original_chars}",
                f"原始 UTF-8 字节数：{record.original_bytes}",
                f"文件：{record.relative_path}",
                "",
            )
        )
        # 开头和结尾各自最多保留的字符数
        limit = TOOL_RESULT_PREVIEW_CHARS
        if len(safe_content) <= limit * 2:
            # 总数小于limit * 2显示全部内容
            body = f"[内容预览]\n{safe_content}"
        else:
            # 否则，开头保留limit个字符，结尾保留limit个字符
            body = (
                f"[开头预览]\n{safe_content[:limit]}\n\n"
                f"[结尾预览]\n{safe_content[-limit:]}"
            )
        return (
            f"{header}{body}\n\n"
            "需要中间内容时，请使用 read_file 的 offset_bytes 和 "
            "limit_bytes 分段读取该文件。"
        )

    def compact_batch(
        self,
        results: tuple[ToolExecutionResult, ...],
    ) -> ToolResultCompactionOutcome:
        """压缩一批完整工具结果，并保持模型原调用顺序。

        Args:
            results: ToolScheduleSession 按 call_index 排好的完整执行结果。

        Returns:
            按原顺序处理后的结果，以及本批首次发生的保存失败。
        """
        # 统计字符数超过限制的工具索引，这些工具产生的结果内容要落盘
        spill_indexes = {
            index
            for index, result in enumerate(results)
            if len(result.content) > self._result_threshold_chars
        }
        # 统计字符数没超过限制的工具索引
        remaining = [
            index
            for index in range(len(results))
            if index not in spill_indexes
        ]
        # 统计所有没超过限制的工具产生的结果的内容总字符数
        total = sum(len(results[index].content) for index in remaining)

        if total > self._batch_threshold_chars:
            # 总字符数超过阈值
            # 按正文长度从大到小排列，然后按照从大到小进行落盘直到小于等于阈值为止
            for index in sorted(
                remaining,
                key=lambda item: (-len(results[item].content), item),
            ):
                spill_indexes.add(index)
                total -= len(results[index].content)
                if total <= self._batch_threshold_chars:
                    break

        # 当前批次最终要写进对话的结果
        processed: list[ToolExecutionResult] = []

        # 当前批次有哪些结果没能保存到 artifact
        failures: list[ToolResultSaveFailure] = []

        for index, result in enumerate(results):
            if index not in spill_indexes:
                processed.append(result)
                continue
            try:
                record = self._artifacts.save_tool_result(
                    result.tool_call_id,
                    result.tool_name,
                    result.content,
                )
            except ArtifactError as exc:
                processed.append(result)
                failures.append(
                    ToolResultSaveFailure(
                        result.tool_call_id,
                        result.tool_name,
                        str(exc),
                    )
                )
                continue
            safe = self._artifacts.safe_text(result.content)
            processed.append(
                replace(
                    result,
                    content=self._preview(result, record, safe),
                    truncated=True,
                    original_size_bytes=record.original_bytes,
                )
            )

        return ToolResultCompactionOutcome(
            tuple(processed),
            tuple(failures),
        )
