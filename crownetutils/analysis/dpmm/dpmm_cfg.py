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

    beacon_app_sql_op: str | Dict[str, str] | None = "app[0].app"
    map_app_sql_op: str | Dict[str, str] | None = "app[1].app"

    def is_count_map(self):
        return self.map_type == MapType.DENSITY

    def is_entropy_map(self):
        return self.map_type == MapType.ENTROPY

    def __post_init__(self):
        self.module_vectors = list(self.module_vectors)

    def get_csv_id_regex_pattern(self) -> re.Pattern:
        return re.compile(self.node_map_csv_id_regex)

    def _create_sql_op(self, app: str | Dict[str, str], modules: List[str], node_index):
        _net = self.network_name
        if isinstance(app, str):
            _or = [f"{_net}.{m}[{node_index}].{app}" for m in modules]
        elif isinstance(app, dict):
            _or = []
            for m in modules:
                app_str = app[m]
                _or.append(f"{_net}.{m}[{node_index}].{app_str}")
        return SqlOp.OR(_or)

    def map_paths(self) -> str:
        return glob.glob(os.path.join(self.base_dir, self.node_map_csv_glob))

    def hdf_path(self) -> str:
        return os.path.join(self.base_dir, self.hdf_file)

    def vec_path(self) -> str:
        return os.path.join(self.base_dir, self.vec_name)

    def sca_path(self) -> str:
        return os.path.join(self.base_dir, self.sca_name)

    def get_beacon_app_sql_op(
        self, modules: List[str] | None = None, node_index: int | str = "%"
    ) -> SqlOp:
        if self.beacon_app_sql_op is None:
            raise ValueError("Current config does not have a beacon application")
        return self._create_sql_op(
            self.beacon_app_sql_op,
            self.module_vectors if modules is None else modules,
            node_index,
        )

    def get_map_app_sql_op(
        self, modules: List[str] | None = None, node_index: int | str = "%"
    ) -> SqlOp:
        if self.map_app_sql_op is None:
            raise ValueError("Current config does not have a map application")
        return self._create_sql_op(
            self.map_app_sql_op,
            self.module_vectors if modules is None else modules,
            node_index,
        )

    @classmethod
    def default_density_beacon_map_cfg(cls, base_dir):
        """Default configuration with beacon as app[0], map as app[1]. These are the default settings before
        the DpmmCfg class was introduced for backwards compatibility
        """
        return cls(
            base_dir=base_dir,
            map_type=MapType.DENSITY,
            beacon_app_sql_op="app[0].app",
            map_app_sql_op="app[1].app",
        )
