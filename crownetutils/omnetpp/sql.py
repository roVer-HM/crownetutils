from __future__ import annotations

import sqlite3 as sq
from typing import Any, List, Protocol


class SqlOp:
    """
    Helper class to build `WHERE` clause for matching a
    column against a single or multiple values which
    may contain placeholder strings `%`.
    Use classmethods `OR` or `AND` for the respective boolean operator
    needed.
    """

    def __init__(self, operator, group):
        self._operator = operator
        self._group = group if isinstance(group, list) else [group]

    @classmethod
    def OR(cls, items):
        return cls("or", items)

    @classmethod
    def AND(cls, items):
        return cls("and", items)

    def get_names(self):
        return self._group

    def append_path(self, path: str) -> SqlOp:
        path = path if path.startswith(".") else f".{path}"
        for i in range(len(self._group)):
            self._group[i] = f"{self._group[i]}{path}"
        return self

    def drop_path_elements(self, num: int = 1) -> SqlOp:
        for i in range(len(self._group)):
            path_elements = self._group[i].split(".")
            if len(path_elements) < num:
                raise ValueError(
                    f"path is to sort. cannot drop {num} elements: {self._group[i]}"
                )
            self._group[i] = ".".join(path_elements[0 : len(path_elements) - 1])
        return self

    def apply(self, table, column):
        ret = []
        for i in self._group:
            if "%" in i:
                ret.append(f"{table}.{column} like '{i}'")
            else:
                ret.append(f"{table}.{column} = '{i}'")
        if len(ret) == 1:
            return ret[0]
        else:
            ret = f" {self._operator} ".join(ret)
            return f"({ret})"

    def append_suffix(self, suffix: str):
        self._group = [f"{i}{suffix}" for i in self._group]

    def info_str(self) -> str:
        _items = ", ".join(self._group)
        return f"{self._operator}[{_items}]"

    def __iter__(self):
        return iter(self._group)

    def __repr__(self) -> str:
        return self.info_str()

    def __str__(self) -> str:
        return self.apply("TABLE", "COLUMN")


class RelativeModuleFunction(Protocol):
    def __call__(
        self,
        modules: List[str] | None = None,
        path: str = "app",
        node_index: int | str = "%",
    ) -> SqlOp:
        """
        Provides a SqlOp path for a module such as pNode[%].<...> or vNode[%].<...> where the
        base node is provided by the implementer.
        """
        ...
