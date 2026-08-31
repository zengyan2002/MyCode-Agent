"""为单个 Agent 缓存已经读取并解码的 UTF-8 文件正文。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileCacheEntry:
    """保存一次文件读取结果及再次使用前需要核对的元数据。

    Attributes:
        canonical_path: ``resolve`` 后用于缓存键和失效操作的真实路径。
        content: 已经按 UTF-8 解码的完整文件正文。
        size: 读取时文件系统报告的字节数。
        mtime_ns: 读取时文件系统报告的纳秒修改时间。
    """

    canonical_path: Path
    content: str
    size: int
    mtime_ns: int


class AgentFileCache:
    """在一个 Agent 生命周期内复用未变化文件的完整正文。

    每个主 Agent、定义式子 Agent 和 Fork 子 Agent 都创建自己的实例。
    该类不启动文件监听；每次命中前重新读取 ``stat``，大小或修改时间变化
    时立即重新读取文件。

    Attributes:
        _entries: 键是规范化绝对路径，值是正文和读取时的大小、修改时间。
    """

    def __init__(self) -> None:
        """创建没有任何条目的文件缓存。

        Returns:
            不返回数据。
        """

        self._entries: dict[Path, FileCacheEntry] = {}

    def read_text(self, path: Path) -> str:
        """读取 UTF-8 文件，并在元数据未变化时复用已解码正文。

        Args:
            path: 已通过工作区边界检查的文件路径。

        Returns:
            文件的完整 Unicode 正文。

        Raises:
            OSError: 文件无法 stat 或读取。
            UnicodeDecodeError: 文件不是合法 UTF-8 文本。
        """

        canonical = path.resolve()
        stat = canonical.stat()
        cached = self._entries.get(canonical)
        if (
            cached is not None
            and cached.size == stat.st_size
            and cached.mtime_ns == stat.st_mtime_ns
        ):
            return cached.content

        content = canonical.read_bytes().decode("utf-8")
        self._entries[canonical] = FileCacheEntry(
            canonical_path=canonical,
            content=content,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )
        return content

    def invalidate(self, path: Path) -> None:
        """移除一个路径的旧正文，使下次读取一定访问文件系统。

        Args:
            path: 可能已经被 write_file 或 edit_file 改变的路径。

        Returns:
            不返回数据；路径没有缓存条目时也视为成功。
        """

        self._entries.pop(path.resolve(), None)

    def clear(self) -> None:
        """清除当前 Agent 累积的全部文件正文。

        Returns:
            不返回数据；独立运行结束时调用以释放正文占用的内存。
        """

        self._entries.clear()

    @property
    def entry_count(self) -> int:
        """返回当前缓存中的文件数量。

        Returns:
            尚未失效或清理的路径条目数。
        """

        return len(self._entries)
