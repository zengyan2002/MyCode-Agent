"""应用配置加载。"""

from mycode.settings.permissions import (
    LocalPermissionStore,
    load_permission_settings,
)

__all__ = [
    "LocalPermissionStore",
    "load_permission_settings",
]
