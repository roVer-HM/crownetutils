from __future__ import annotations

import glob
import os
import re
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Dict, List

from crownetutils.omnetpp.sql import SqlOp
from crownetutils.utils.misc import Project


class MapType(Enum):
    DENSITY = "density"
    ENTROPY = "entropy"


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

    def hdf_path(self) -> str:
        return os.path.join(self.base_dir, self.hdf_file)

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
