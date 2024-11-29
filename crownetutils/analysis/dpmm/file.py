from dataclasses import dataclass
from os.path import exists
from typing import Protocol

from crownetutils.analysis.dpmm import MapType


class CfgProvider(Protocol):
    def path(self, *args) -> str:
        ...

    @property
    def map_type(self) -> MapType:
        ...


@dataclass
class MapTypedFile:
    file_name: str
    is_typed: bool

    def __call__(self, t: MapType):
        if self.is_typed:
            return f"{t.value}_{self.file_name}"
        else:
            return self.file_name

    def __post_init__(self):
        self._cfg: CfgProvider | None = None

    @property
    def name(self) -> str:
        if self._cfg is None:
            raise ValueError(
                "MapTypeFile not fully initialized. Should be done by DpmmCfg post_init"
            )
        return self(self._cfg.map_type)

    @property
    def path(self) -> str:
        if self._cfg is None:
            raise ValueError(
                "MapTypeFile not fully initialized. Should be done by DpmmCfg post_init"
            )
        return self._cfg.path(self.name)

    def as_dict(self) -> dict:
        return dict(file_name=self.file_name, is_typed=self.is_typed)

    def exists(self) -> bool:
        """True if underling file exists."""
        return exists(self.path)


@dataclass
class MapTypedGroupedFile(MapTypedFile):
    group: str

    def as_dict(self) -> dict:
        _obj = super().as_dict()
        _obj["group"] = self.group
        return _obj
