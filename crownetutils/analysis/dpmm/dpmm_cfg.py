from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections.abc import Callable
from dataclasses import InitVar, asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, TextIO

from crownetutils.omnetpp.sql import SqlOp
from crownetutils.utils.misc import Project


class MapType(Enum):
    DENSITY = "density"
    ENTROPY = "entropy"

    def __str__(self) -> str:
        return self.value


class DpmmCfgEncoder(json.JSONEncoder):
    @classmethod
    def new(cls, keep_base_dir: bool):
        """Provide pre configured callable to crete json encoder"""

        def func(*args, **kwargs) -> DpmmCfgEncoder:
            return DpmmCfgEncoder(keep_base_dir, *args, **kwargs)

        return func

    def __init__(
        self,
        keep_base_dir: bool,
        *,
        skipkeys: bool = False,
        ensure_ascii: bool = True,
        check_circular: bool = True,
        allow_nan: bool = True,
        sort_keys: bool = False,
        indent: int | str | None = None,
        separators: tuple[str, str] | None = None,
        default: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            skipkeys=skipkeys,
            ensure_ascii=ensure_ascii,
            check_circular=check_circular,
            allow_nan=allow_nan,
            sort_keys=sort_keys,
            indent=indent,
            separators=separators,
            default=default,
        )
        self.keep_base_dir = keep_base_dir

    def default(self, o: Any) -> Any:
        if isinstance(o, DpmmCfg):
            __o = asdict(o)
            if not self.keep_base_dir:
                __o["base_dir"] = None
            return __o
        elif isinstance(o, MapType):
            return o.value
        else:
            return super().default(o)


class DpmmCfgDecoder(json.JSONDecoder):
    @classmethod
    def new(cls, base_dir=None) -> DpmmCfgDecoder:
        def func(*args, **kwargs):
            return DpmmCfgDecoder(base_dir, *args, **kwargs)

        return func

    def __init__(
        self,
        base_dir=None,
        *,
        parse_float: Callable[[str], Any] | None = None,
        parse_int: Callable[[str], Any] | None = None,
        parse_constant: Callable[[str], Any] | None = None,
        strict: bool = True,
        object_pairs_hook: Callable[[list[tuple[str, Any]]], Any] | None = None,
    ) -> None:
        super().__init__(
            object_hook=self.object_hook,
            parse_float=parse_float,
            parse_int=parse_int,
            parse_constant=parse_constant,
            strict=strict,
            object_pairs_hook=object_pairs_hook,
        )
        self.base_dir = base_dir

    def object_hook(self, obj):
        if obj["base_dir"] is None and self.base_dir is None:
            raise ValueError(
                "Provided json object does not base_dir nor does the JSONDecoder. obj: {obj} "
            )
        if self.base_dir is not None:
            obj["base_dir"] = self.base_dir
        try:
            obj["map_type"] = MapType(obj["map_type"])
        except:
            ValueError(f"Did not found TypeMap {obj['map_type']}")
        return DpmmCfg(**obj)


@dataclass
class DpmmCfg:
    base_dir: str
    hdf_file: str = "data.h5"
    vec_name: str = "vars_rep_0.vec"
    sca_name: str = "vars_rep_0.sca"
    network_name: str = "World"

    map_type: MapType = MapType.DENSITY
    global_map_ini_path: str = "World.globalDensityMap"
    global_map_csv_name: str = "global.csv"
    node_map_csv_glob: str = "dcdMap_*.csv"
    node_map_csv_id_regex: str = r"dcdMap_(?P<node>\d+)\.csv"
    epsg_base: Project = Project.UTM_32N

    module_vectors: List[str] = ("misc", "pNode", "vNode")

    beacon_app_path: str | Dict[str, str] | None = "app[0]"
    map_app_path: str | Dict[str, str] | None = "app[1]"

    def path(self, *paths) -> str:
        return os.path.join(self.base_dir, *paths)

    def output_path(self, *paths) -> str:
        _p = self.map_type.value
        return os.path.join(self.base_dir, _p, *paths)

    def makedirs(self, *paths, mode=0o777, exist_ok=False) -> str:
        _p = self.path(*paths)
        os.makedirs(_p, mode=mode, exist_ok=exist_ok)
        return _p

    def makedirs_output(self, *paths, mode=0o777, exist_ok=False) -> str:
        _p = self.output_path(*paths)
        os.makedirs(_p, mode=mode, exist_ok=exist_ok)
        return _p

    def is_count_map(self):
        return self.map_type == MapType.DENSITY

    def is_entropy_map(self):
        return self.map_type == MapType.ENTROPY

    def __post_init__(self):
        self.module_vectors = list(self.module_vectors)

    def get_csv_id_regex_pattern(self) -> re.Pattern:
        return re.compile(self.node_map_csv_id_regex)

    def _create_sql_op(
        self, app: str | Dict[str, str], modules: List[str], node_index: int, path: str
    ):
        _net = self.network_name
        path = path if path.startswith(".") else f".{path}"
        if isinstance(app, str):
            _or = [f"{_net}.{m}[{node_index}].{app}{path}" for m in modules]
        elif isinstance(app, dict):
            _or = []
            for m in modules:
                app_str = app[m]
                _or.append(f"{_net}.{m}[{node_index}].{app_str}{path}")
        return SqlOp.OR(_or)

    def map_paths(self) -> str:
        return glob.glob(os.path.join(self.base_dir, self.node_map_csv_glob))

    def hdf_path(self, file_name: str | None = None) -> str:
        if file_name is None:
            return os.path.join(self.base_dir, self.hdf_file)
        else:
            return os.path.join(self.base_dir, f"{self.map_type.value}_{file_name}")

    def vec_path(self) -> str:
        return os.path.join(self.base_dir, self.vec_name)

    def sca_path(self) -> str:
        return os.path.join(self.base_dir, self.sca_name)

    def m_beacon(
        self,
        modules: List[str] | None = None,
        path: str = "app",
        node_index: int | str = "%",
    ) -> SqlOp:
        if self.beacon_app_path is None:
            raise ValueError("Current config does not have a beacon application")
        return self._create_sql_op(
            self.beacon_app_path,
            modules=self.module_vectors if modules is None else modules,
            node_index=node_index,
            path=path,
        )

    def m_map(
        self,
        modules: List[str] | None = None,
        path: str = "app",
        node_index: int | str = "%",
    ) -> SqlOp:
        if self.map_app_path is None:
            raise ValueError("Current config does not have a map application")
        return self._create_sql_op(
            self.map_app_path,
            modules=self.module_vectors if modules is None else modules,
            node_index=node_index,
            path=path,
        )

    def as_dict(self):
        d = asdict(self)
        d["map_type"] = self.map_type.value
        return d

    def to_json(self, fd: TextIO | None = None, dump_base_dir: bool = True):
        if fd is None:
            return json.dumps(
                obj=self,
                ensure_ascii=True,
                allow_nan=True,
                indent=2,
                sort_keys=True,
                cls=DpmmCfgEncoder.new(dump_base_dir),
            )
        else:
            json.dump(
                obj=self,
                fp=fd,
                ensure_ascii=True,
                allow_nan=True,
                indent=2,
                sort_keys=True,
                cls=DpmmCfgEncoder.new(dump_base_dir),
            )

    def save_cfg(self, path_fd: str | TextIO, dump_base_path: bool = True):
        if isinstance(path_fd, str):
            with open(path_fd, "w", encoding="utf-8") as fd:
                self.to_json(fd, dump_base_dir=dump_base_path)
        else:
            self.to_json(path_fd, dump_base_dir=dump_base_path)

    def copy(self, base_dir) -> DpmmCfg:
        """Get new instance with different base directory"""
        o = asdict(self)
        o["base_dir"] = base_dir
        return DpmmCfg(**o)

    @classmethod
    def from_json(cls, str_fd: str | TextIO, base_dir=None) -> DpmmCfg:
        if isinstance(str_fd, str):
            o = json.loads(str_fd, cls=DpmmCfgDecoder.new(base_dir))
        else:
            o = json.load(str_fd, cls=DpmmCfgDecoder.new(base_dir))
        return o

    @classmethod
    def load(cls, str_fd: str | TextIO, base_dir=None) -> DpmmCfg:
        if isinstance(str_fd, str):
            with open(str_fd, "r", encoding="utf-8") as fd:
                return cls.from_json(fd, base_dir=base_dir)
        else:
            return cls.from_json(str_fd, base_dir=base_dir)

    @classmethod
    def default_density_beacon_map_cfg(cls, base_dir):
        """Default configuration with beacon as app[0], map as app[1]. These are the default settings before
        the DpmmCfg class was introduced for backwards compatibility
        """
        return cls(
            base_dir=base_dir,
            map_type=MapType.DENSITY,
            beacon_app_path="app[0]",
            map_app_path="app[1]",
        )
