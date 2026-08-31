"""MyCode 当前会话的工具结果和对话历史压缩能力。"""

from mycode.context.artifacts import ArtifactRecord, ArtifactStore
from mycode.context.manager import (
    CompactionMode,
    CompactionOutcome,
    CompactionOutcomeKind,
    ContextManager,
    RestoredContext,
)
from mycode.context.tool_results import ToolResultCompactor

__all__ = [
    "ArtifactRecord",
    "ArtifactStore",
    "CompactionMode",
    "CompactionOutcome",
    "CompactionOutcomeKind",
    "ContextManager",
    "RestoredContext",
    "ToolResultCompactor",
]
