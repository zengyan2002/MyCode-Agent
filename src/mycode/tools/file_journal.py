"""由真实文件写线程记录目标内容和结束时间，供超时后的读取核查使用。"""

import hashlib
from pathlib import Path

from mycode.models.operations import FileWriteExpectation
from mycode.persistence.operations import OperationStore


class FileWriteJournal:
    def __init__(self, store: OperationStore, operation_id: str,
                 owner_token: str, attempt: int) -> None:
        self.store = store
        self.operation_id = operation_id
        self.owner_token = owner_token
        self.attempt = attempt

    def record_expected(self, path: Path, content: bytes) -> None:
        """必须在首次修改文件前成功返回；否则本次写入不应开始。"""
        self.store.save_file_expectation(FileWriteExpectation(
            self.operation_id, self.owner_token, self.attempt, str(path),
            hashlib.sha256(content).hexdigest(), len(content)))

    def mark_finished(self) -> None:
        """写入及其清理结束后调用，不能用协程取消代替这个标记。"""
        self.store.finish_file_writer(self.operation_id, self.owner_token)
