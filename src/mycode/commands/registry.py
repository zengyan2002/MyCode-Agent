"""保存启动时登记的命令，并提供只读查找和补全。"""

from __future__ import annotations

from mycode.commands.models import Command


class CommandRegistry:
    """保存静态命令和可以热替换的 Skill 命令。

    CLI 在启动阶段通过 register 登记静态命令并冻结；SkillService 随后
    只能调用 replace_skill_commands 整批替换动态层。查找、帮助和补全
    会把两个层次当成同一份命令列表。
    """

    def __init__(self) -> None:
        """创建尚未冻结的空注册表。

        Returns:
            无返回值；新实例可以在启动阶段接收命令登记。
        """

        # 按登记顺序保存内置静态命令，freeze 后不再改变。
        self._static_commands: dict[str, Command] = {}
        # 静态正式名称和别名使用同一份查找索引。
        self._static_lookup: dict[str, Command] = {}
        # 每次 reload 原子替换的 Skill 命令，顺序由 Catalog 名称排序决定。
        self._skill_commands: dict[str, Command] = {}
        # 动态 Skill 正式名称和别名使用的查找索引。
        self._skill_lookup: dict[str, Command] = {}
        # True 表示静态登记结束；动态 Skill 层仍允许受控整批替换。
        self._frozen = False

    @staticmethod
    def _normalized(value: str, *, label: str) -> str:
        """检查一个登记名称并返回不区分大小写的索引键。

        Args:
            value: 不含斜杠的正式名称或别名。
            label: 错误信息中使用的“命令名称”或“命令别名”。

        Returns:
            对 ``value`` 执行 ``casefold`` 后得到的索引键。
        """

        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or "/" in value
            or any(character.isspace() for character in value)
        ):
            raise ValueError(f"{label}必须是不含斜杠和空白的非空文本")
        return value.casefold()

    def register(self, command: Command) -> None:
        """校验并登记一条命令及其全部别名。

        Args:
            command: 包含正式名称、别名、帮助文字和 Handler 的命令定义。

        Returns:
            无返回值；成功后名称和别名可以立即用于查找。
        """

        if self._frozen:
            raise RuntimeError("命令注册表已经冻结")
        if command.skill:
            raise ValueError("Skill 命令必须通过 replace_skill_commands 登记")
        self._validate_command(command)
        normalized_name, lookup_entries = self._entries_for(command)
        self._reject_conflicts(
            command,
            lookup_entries,
            self._static_lookup,
        )
        self._static_commands[normalized_name] = command
        for normalized in lookup_entries:
            self._static_lookup[normalized] = command

    def _validate_command(self, command: Command) -> None:
        """检查一条命令本身的类型、描述和用法。

        Args:
            command: 静态登记或动态替换中的命令定义。

        Returns:
            None。字段合法时继续，非法时抛出异常。

        Raises:
            TypeError: 传入值不是 Command。
            ValueError: 描述或用法为空。
        """

        if not isinstance(command, Command):
            raise TypeError("只能登记 Command 实例")
        if not command.description.strip():
            raise ValueError(f"命令 {command.name!r} 必须提供简短描述")
        if not command.usage.strip():
            raise ValueError(f"命令 {command.name!r} 必须提供用法")

    def _entries_for(
        self,
        command: Command,
    ) -> tuple[str, dict[str, str]]:
        """规范化一条命令的正式名称和别名。

        Args:
            command: 等待登记的命令。

        Returns:
            规范正式名称，以及“规范键到原始写法”的字典。

        Raises:
            ValueError: 名称格式非法或同一命令内部重复。
        """

        entries = ((command.name, "命令名称"),) + tuple(
            (alias, "命令别名") for alias in command.aliases
        )
        seen_in_command: dict[str, str] = {}
        for value, label in entries:
            normalized = self._normalized(value, label=label)
            previous_value = seen_in_command.get(normalized)
            if previous_value is not None:
                raise ValueError(
                    f"命令 {command.name!r} 内名称重复："
                    f"{previous_value!r} 与 {value!r}"
                )
            seen_in_command[normalized] = value
        normalized_name = self._normalized(
            command.name,
            label="命令名称",
        )
        return normalized_name, seen_in_command

    def _reject_conflicts(
        self,
        command: Command,
        entries: dict[str, str],
        lookup: dict[str, Command],
    ) -> None:
        """检查名称和别名是否已经属于其他命令。

        Args:
            command: 正在登记的命令，用在错误信息中。
            entries: 当前命令的规范键和原始写法。
            lookup: 需要检查的静态或临时动态索引。

        Returns:
            None。没有冲突时保持索引不变。

        Raises:
            ValueError: 任一名称或别名已经被占用。
        """

        for normalized, value in entries.items():
            existing = lookup.get(normalized)
            if existing is not None:
                raise ValueError(
                    f"名称 {value!r} 冲突：已经属于命令 "
                    f"{existing.name!r}，不能再登记给 {command.name!r}"
                )

    def replace_skill_commands(
        self,
        commands: tuple[Command, ...],
    ) -> None:
        """整批替换动态 Skill 命令，失败时保留旧动态层。

        Args:
            commands: Catalog 为当前有效 Skill 生成的完整命令列表。

        Returns:
            None。全部命令通过校验后，帮助、查找和补全立即看到新列表。

        Raises:
            ValueError: 命令未标记为 Skill，或与静态/其他 Skill 名称冲突。
        """

        next_commands: dict[str, Command] = {}
        next_lookup: dict[str, Command] = {}
        for command in commands:
            self._validate_command(command)
            if not command.skill:
                raise ValueError(
                    f"动态命令 {command.name!r} 必须标记为 Skill"
                )
            normalized_name, entries = self._entries_for(command)
            self._reject_conflicts(
                command,
                entries,
                self._static_lookup,
            )
            self._reject_conflicts(command, entries, next_lookup)
            next_commands[normalized_name] = command
            for normalized in entries:
                next_lookup[normalized] = command

        self._skill_commands = next_commands
        self._skill_lookup = next_lookup

    def freeze(self) -> None:
        """结束启动登记并禁止本次进程继续修改命令集合。

        Returns:
            无返回值；调用后查找和补全仍可继续使用。
        """

        self._frozen = True

    def find(self, name: str) -> Command | None:
        """按正式名称或别名查找一条命令。

        Args:
            name: 不含斜杠的用户输入名称，大小写不限。

        Returns:
            命中的 ``Command``；没有登记该名称时返回 ``None``。
        """

        if not isinstance(name, str):
            return None
        normalized = name.casefold()
        return self._static_lookup.get(normalized) or self._skill_lookup.get(
            normalized
        )

    @property
    def visible_commands(self) -> tuple[Command, ...]:
        """返回按登记顺序排列的全部非隐藏命令。

        Returns:
            `/help` 和补全界面可以展示的不可变命令元组。
        """

        return tuple(
            command
            for command in (
                *self._static_commands.values(),
                *self._skill_commands.values(),
            )
            if not command.hidden
        )

    @property
    def static_names(self) -> frozenset[str]:
        """返回全部静态正式名称和别名的规范键。

        Returns:
            Parser 可以用来拒绝 Skill 撞名的不可变名称集合。
        """

        return frozenset(self._static_lookup)

    def complete(self, prefix: str) -> tuple[str, ...]:
        """查找以给定文本开头的可见正式名称和别名。

        Args:
            prefix: 用户已经输入的命令前缀，可以包含开头的 `/`。

        Returns:
            带 `/`、去重并按不区分大小写顺序排列的候选元组。
        """

        normalized_prefix = prefix.removeprefix("/").casefold()
        matches: set[str] = set()
        for command in self.visible_commands:
            for value in (command.name, *command.aliases):
                if value.casefold().startswith(normalized_prefix):
                    matches.add("/" + value)
        return tuple(sorted(matches, key=lambda value: (value.casefold(), value)))
