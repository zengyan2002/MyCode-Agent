"""在容器临时目录准备工作副本，然后用 Linux Shell 执行一条命令。"""

import os
import shutil
import sys


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("沙箱只接受一条命令参数")
    shutil.copytree("/input", "/workspace", dirs_exist_ok=True)
    os.chdir("/workspace")
    os.execv("/bin/sh", ["/bin/sh", "-c", sys.argv[1]])


if __name__ == "__main__":
    main()
