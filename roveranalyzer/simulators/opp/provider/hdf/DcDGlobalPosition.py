from abc import ABC
from typing import Dict, List

import pandas as pd

from roveranalyzer.simulators.crownet.dcd.util import DcdMetaData, read_csv
from roveranalyzer.simulators.opp.provider.hdf.HdfGroups import HdfGroups
from roveranalyzer.simulators.opp.provider.hdf.IHdfProvider import IHdfProvider


class DcdGlobalMapKey:
    SIMTIME = "simtime"
    X = "x"
    Y = "y"
    SOURCE = "source"
    COUNT = "count"
    MEASURE_TIME = "measured_t"
    NODE_ID = "node_id"

    types_global_raw_csv_index = {
        SIMTIME: float,
        X: float,
        Y: float,
    }
    types_global_raw_csv_col = {
        COUNT: float,
        NODE_ID: str,  # contains comma separated list will later be parsed to int list
    }

    types_global_pos = {SIMTIME: float, X: float, Y: float, NODE_ID: int}

    @classmethod
    def global_pos_columns(cls):
        return {i[0]: i[1] for i in enumerate(list(cls.types_global_pos.keys()))}

    types_global_density = {SIMTIME: float, X: float, Y: float, COUNT: int}

    @classmethod
    def global_density_columns(cls):
        return {i[0]: i[1] for i in enumerate(list(cls.types_global_density.keys()))}


def pos_density_from_csv(csv_path: str, hdf_path: str):
    pos = DcdGlobalPosition(hdf_path)
    density = DcdGlobalDensity(hdf_path)
    global_df, meta = read_csv(
        csv_path=csv_path,
        _index_types=DcdGlobalMapKey.types_global_raw_csv_index,
        _col_types=DcdGlobalMapKey.types_global_raw_csv_col,
        real_coords=True,
    )
    position_df, global_df = build_position_df(global_df)
    position_df.set_index(
        keys=list(pos.index_order().values()), inplace=True, verify_integrity=True
    )
    pos.write_dataframe(position_df)
    density.write_dataframe(global_df)

    pos.set_attribute("cell_size", meta.cell_size)
    pos.set_attribute("cell_count", meta.cell_count)
    pos.set_attribute("cell_bound", meta.bound)

    density.set_attribute("cell_size", meta.cell_size)
    density.set_attribute("cell_count", meta.cell_count)
    density.set_attribute("cell_bound", meta.bound)

    return pos, density


def build_position_df(glb_df):
    # global position map for all node_ids
    glb_loc_df = glb_df["node_id"].copy().reset_index()
    glb_loc_df = glb_loc_df.assign(
        node_id=glb_loc_df["node_id"].str.split(r",\s*")
    ).explode("node_id")
    glb_loc_df["node_id"] = pd.to_numeric(glb_loc_df["node_id"], downcast="integer")
    # remove node_id column from global
    glb_df = glb_df.drop(labels=["node_id"], axis="columns")
    return glb_loc_df, glb_df


class DcdGlobalPosition(IHdfProvider):
    def __init__(self, hdf_path):
        super().__init__(hdf_path)

    def group_key(self) -> str:
        return HdfGroups.DCD_GLOBAL_POS

    def index_order(self) -> Dict:
        return {0: DcdGlobalMapKey.SIMTIME, 1: DcdGlobalMapKey.NODE_ID}

    def columns(self) -> List[str]:
        return [DcdGlobalMapKey.X, DcdGlobalMapKey.Y]

    def default_index_key(self) -> str:
        return self.index_order()[0]

    def get_meta_object(self) -> DcdMetaData:
        cell_size = self.get_attribute("cell_size")
        cell_count = self.get_attribute("cell_count")
        cell_bound = self.get_attribute("cell_bound")
        return DcdMetaData(cell_size, cell_count, cell_bound, "global")


class DcdGlobalDensity(IHdfProvider):
    def __init__(self, hdf_path):
        super().__init__(hdf_path)

    def group_key(self) -> str:
        return HdfGroups.DCD_GLOBAL_DENSITY

    def index_order(self) -> Dict:
        return {0: DcdGlobalMapKey.SIMTIME, 1: DcdGlobalMapKey.X, 2: DcdGlobalMapKey.Y}

    def columns(self) -> List[str]:
        return [DcdGlobalMapKey.COUNT]

    def default_index_key(self) -> str:
        return self.index_order()[0]

    def get_meta_object(self) -> DcdMetaData:
        cell_size = self.get_attribute("cell_size")
        cell_count = self.get_attribute("cell_count")
        cell_bound = self.get_attribute("cell_bound")
        return DcdMetaData(cell_size, cell_count, cell_bound, "global")
