from __future__ import annotations

import abc
import dataclasses
import glob
import json
import os
import re
import sys
from collections.abc import Callable
from dataclasses import InitVar, asdict, dataclass, field
from io import StringIO
from typing import Any, Dict, List, TextIO

from crownetutils.analysis.dpmm import MapType
from crownetutils.analysis.dpmm.file import MapTypedFile, MapTypedGroupedFile
from crownetutils.omnetpp.sql import SqlOp
from crownetutils.utils.misc import Project


class DpmmCfgEncoder(json.JSONEncoder):
    @classmethod
    def new(cls, keep_base_dir: bool, _clazz):
        """Provide pre configured callable to crete json encoder"""

        def func(*args, **kwargs) -> DpmmCfgEncoder:
            return DpmmCfgEncoder(keep_base_dir, _clazz, *args, **kwargs)

        return func

    def __init__(
        self,
        keep_base_dir: bool,
        _clazz,
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
        self._clazz = _clazz

    def default(self, o: Any) -> Any:
        if isinstance(o, self._clazz):
            __o = asdict(o)
            if not self.keep_base_dir:
                __o["base_dir"] = None
            return __o
        elif isinstance(o, MapType):
            return o.value
        elif isinstance(o, MapTypedFile):
            return o.as_dict()
        else:
            return super().default(o)


class DpmmCfgDecoder(json.JSONDecoder):
    @classmethod
    def new(cls, _clazz, base_dir=None) -> DpmmCfgDecoder:
        def func(*args, **kwargs):
            return DpmmCfgDecoder(_clazz, base_dir, *args, **kwargs)

        return func

    def __init__(
        self,
        _clazz,
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
        self._clazz = _clazz

    def object_hook(self, obj):
        if len(obj) == 2 and "file_name" in obj and "is_typed" in obj:
            return MapTypedFile(**obj)
        if len(obj) == 3 and "file_name" in obj and "is_typed" and "group" in obj:
            return MapTypedGroupedFile(**obj)
        if "base_dir" in obj:
            if obj["base_dir"] is None and self.base_dir is None:
                raise ValueError(
                    "Provided json object does not base_dir nor does the JSONDecoder. obj: {obj} "
                )
            if self.base_dir is not None:
                obj["base_dir"] = self.base_dir

        if "map_type" in obj:
            try:
                obj["map_type"] = MapType(obj["map_type"])
            except:
                ValueError(f"Did not found TypeMap {obj['map_type']}")

        return self._clazz(**obj)


@dataclass
class DpmmCfg(abc.ABC):
    base_dir: str
    hdf_file: str = "data.h5"
    vec_name: str = "vars_rep_0.vec"
    sca_name: str = "vars_rep_0.sca"
    network_name: str = "World"

    #
    position: MapTypedFile = field(
        default_factory=lambda: MapTypedFile("position.h5", False)
    )
    node_tx: MapTypedFile = field(
        default_factory=lambda: MapTypedFile("node_tx_data.h5", False)
    )
    node_rx: MapTypedFile = field(
        default_factory=lambda: MapTypedFile("node_rx_data.h5", False)
    )
    map_count_error: MapTypedFile = field(
        default_factory=lambda: MapTypedFile("map_count_error.h5", True)
    )
    cell_count_error: MapTypedFile = field(
        default_factory=lambda: MapTypedFile("cell_count_error.h5", True)
    )
    map_size_and_age_over_distance: MapTypedFile = field(
        default_factory=lambda: MapTypedFile("map_size_and_age_over_distance.h5", True)
    )
    map_size: MapTypedFile = field(
        default_factory=lambda: MapTypedFile("map_size.csv", True)
    )
    map_size_global: MapTypedFile = field(
        default_factory=lambda: MapTypedFile("map_size_global.csv", True)
    )
    map_measurements_age_over_distance: MapTypedFile = field(
        default_factory=lambda: MapTypedFile(
            "map_measurements_age_over_distance.h5", True
        )
    )
    enb_rb: MapTypedGroupedFile = field(
        default_factory=lambda: MapTypedGroupedFile(
            "enb_rb.h5", is_typed=False, group="enb_rb"
        )
    )

    # default_output_names

    map_type: MapType = MapType.DENSITY
    global_map_ini_path: str = "World.globalDensityMap"
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
        for f in dataclasses.fields(self):
            _obj = getattr(self, f.name)
            if isinstance(_obj, MapTypedFile):
                _obj._cfg = self

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

    def as_dict(self) -> dict:
        d = asdict(self)
        d["map_type"] = self.map_type.value
        return d

    @abc.abstractmethod
    def to_json(
        self, fd: TextIO | None = None, dump_base_dir: bool = True
    ) -> None | str:
        pass

    @abc.abstractclassmethod
    def load(cls, str_fd: str | TextIO, base_dir=None) -> DpmmCfgCsv:
        pass

    @abc.abstractclassmethod
    def from_json(cls, str_fd: str | TextIO, base_dir=None) -> DpmmCfgDb:
        pass

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
        return self.__class__(**o)


@dataclass
class DpmmCfgCsv(DpmmCfg):
    global_map_csv_name: str = "global.csv"
    node_map_csv_glob: str = "dcdMap_*.csv"
    node_map_csv_id_regex: str = r"dcdMap_(?P<node>\d+)\.csv"

    def get_csv_id_regex_pattern(self) -> re.Pattern:
        return re.compile(self.node_map_csv_id_regex)

    def map_paths(self) -> str:
        return glob.glob(os.path.join(self.base_dir, self.node_map_csv_glob))

    def to_json(
        self, fd: TextIO | None = None, dump_base_dir: bool = True
    ) -> None | str:
        if fd is None:
            return json.dumps(
                obj=self,
                ensure_ascii=True,
                allow_nan=True,
                indent=2,
                sort_keys=True,
                cls=DpmmCfgEncoder.new(dump_base_dir, _clazz=DpmmCfgCsv),
            )
        else:
            json.dump(
                obj=self,
                fp=fd,
                ensure_ascii=True,
                allow_nan=True,
                indent=2,
                sort_keys=True,
                cls=DpmmCfgEncoder.new(dump_base_dir, _clazz=DpmmCfgCsv),
            )

    @classmethod
    def load(cls, str_fd: str | TextIO, base_dir=None) -> DpmmCfgCsv:
        if isinstance(str_fd, str):
            with open(str_fd, "r", encoding="utf-8") as fd:
                return cls.from_json(fd, base_dir=base_dir)
        else:
            return cls.from_json(str_fd, base_dir=base_dir)

    @classmethod
    def from_json(cls, str_fd: str | TextIO, base_dir=None) -> DpmmCfgCsv:
        if isinstance(str_fd, str):
            o = json.loads(
                str_fd, cls=DpmmCfgDecoder.new(_clazz=DpmmCfgCsv, base_dir=base_dir)
            )
        else:
            o = json.load(
                str_fd, cls=DpmmCfgDecoder.new(_clazz=DpmmCfgCsv, base_dir=base_dir)
            )
        return o

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


@dataclass
class DpmmCfgDb(DpmmCfg):
    map_db_name: str = "global_densityMap.db"
    tbl_metadata: str = "metadata"
    tbl_map: str = "dcd_map"
    tbl_map_glb: str = "dcd_map_glb"
    tbl_alg_mapping: str = "alg_mapping"
    tbl_glb_node_id_mapping: str = "glb_node_id_mapping"

    def __post_init__(self):
        return super().__post_init__()

    def to_json(
        self, fd: TextIO | None = None, dump_base_dir: bool = True
    ) -> None | str:
        if fd is None:
            return json.dumps(
                obj=self,
                ensure_ascii=True,
                allow_nan=True,
                indent=2,
                sort_keys=True,
                cls=DpmmCfgEncoder.new(dump_base_dir, _clazz=DpmmCfgDb),
            )
        else:
            json.dump(
                obj=self,
                fp=fd,
                ensure_ascii=True,
                allow_nan=True,
                indent=2,
                sort_keys=True,
                cls=DpmmCfgEncoder.new(dump_base_dir, _clazz=DpmmCfgDb),
            )

    @classmethod
    def load(cls, str_fd: str | TextIO, base_dir=None) -> DpmmCfgDb:
        if isinstance(str_fd, str):
            with open(str_fd, "r", encoding="utf-8") as fd:
                return cls.from_json(fd, base_dir=base_dir)
        else:
            return cls.from_json(str_fd, base_dir=base_dir)

    @classmethod
    def from_json(cls, str_fd: str | TextIO, base_dir=None) -> DpmmCfgDb:
        if isinstance(str_fd, str):
            o = json.loads(
                str_fd, cls=DpmmCfgDecoder.new(_clazz=DpmmCfgDb, base_dir=base_dir)
            )
        else:
            o = json.load(
                str_fd, cls=DpmmCfgDecoder.new(_clazz=DpmmCfgDb, base_dir=base_dir)
            )
        return o


class DpmmCfgBuilder:
    @classmethod
    def load_density_cfg(cls, path) -> DpmmCfg:
        b = cls()
        return b.load_cfg_from_base_dir(base_dir=path, map_type=MapType.DENSITY)

    @classmethod
    def load_entropy_cfg(cls, path) -> DpmmCfg:
        b = cls()
        return b.load_cfg_from_base_dir(base_dir=path, map_type=MapType.ENTROPY)

    @classmethod
    def load_cfg(cls, path, error_on_nan: bool = False):
        cfg = None
        try:
            cfg = cls.load_density_cfg(path)
        except FileNotFoundError:
            try:
                cfg = cls.load_entropy_cfg(path)
            except FileNotFoundError:
                pass

        if cfg is None and error_on_nan:
            raise FileNotFoundError(
                f"Cannot find config file {MapType.DENSITY.name}.cfg or {MapType.ENTROPY.name}.cfg at {path}"
            )

        return cfg

    def __init__(self) -> None:
        pass

    def _load_json(self, base_dir):
        with open(base_dir, mode="r", encoding="utf-8") as fd:
            return json.load(fd)

    def save_in_root(self, cfg: DpmmCfg):
        """Save configuration in base path directory with map type name. Remove base path to allow base path
        to be moved. Base path can be reconstructed from saved file path

        Args:
            cfg (DpmmCfg): _description_
        """
        map_type = cfg.map_type.name
        cfg.save_cfg(path_fd=cfg.path(f"{map_type}.cfg"), dump_base_path=False)

    def load_from_json(self, obj: dict, base_dir) -> DpmmCfg:
        if "map_db_name" in obj.keys():
            fd = StringIO(json.dumps(obj))
            cfg = DpmmCfgDb.load(fd, base_dir=base_dir)
        elif "global_map_csv_name" in obj.keys():
            fd = StringIO(json.dumps(obj))
            cfg = DpmmCfgCsv.load(fd, base_dir=base_dir)
        else:
            raise ValueError(
                "Provided json does not fit DpmmCfgDb or DpmmCfgCsv. got {j}"
            )

        return cfg

    def load_cfg_from_base_dir(self, base_dir, map_type: MapType) -> DpmmCfg:
        j = self._load_json(base_dir=f"{base_dir}/{map_type.name}.cfg")
        return self.load_from_json(j, base_dir=base_dir)

    def load_db_cfg_from_base_dir(self, base_dir, map_type: MapType) -> DpmmCfgDb:
        cfg = self.load_cfg_from_base_dir(base_dir, map_type)
        if isinstance(cfg, DpmmCfgDb):
            return cfg

        raise ValueError("got cfg type {type(cfg)} instead of DpmmCfgDb")

    def load_csv_cfg_from_base_dir(self, base_dir, map_type: MapType) -> DpmmCfgCsv:
        cfg = self.load_cfg_from_base_dir(base_dir, map_type)
        if isinstance(cfg, DpmmCfgCsv):
            return cfg

        raise ValueError("got cfg type {type(cfg)} instead of DpmmCfgCsv")
