from __future__ import annotations

from typing import Dict, List, Tuple, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from roveranalyzer.simulators.crownet.common import DcdMetaData
from roveranalyzer.simulators.crownet.common.dcd_util import read_csv
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


def build_position_df(glb_df):
    # global df map for all node_ids
    glb_loc_df = glb_df["node_id"].copy().reset_index()
    glb_loc_df = glb_loc_df.assign(
        node_id=glb_loc_df["node_id"].str.split(r",\s*")
    ).explode("node_id")
    glb_loc_df["node_id"] = pd.to_numeric(glb_loc_df["node_id"], downcast="integer")
    glb_loc_df = glb_loc_df.dropna()
    # remove node_id column from global
    glb_df = glb_df.drop(labels=["node_id"], axis="columns")
    return glb_loc_df, glb_df


class DcdGlobalPosition(IHdfProvider):
    def __init__(self, hdf_path, version: str | None = None):
        super().__init__(hdf_path, version)

    def group_key(self) -> str:
        return HdfGroups.DCD_GLOBAL_POS

    def index_order(self) -> Dict:
        return {0: DcdGlobalMapKey.SIMTIME, 1: DcdGlobalMapKey.NODE_ID}

    def columns(self) -> List[str]:
        return [DcdGlobalMapKey.X, DcdGlobalMapKey.Y]

    def default_index_key(self) -> str:
        return self.index_order()[0]

    @staticmethod
    def as_geo(
        df: Union[DcdGlobalPosition, pd.DataFrame, gpd.GeoDataFrame],
        crs: str,
        slice_: slice = slice(None),
    ) -> gpd.GeoDataFrame:
        if type(df) == DcdGlobalPosition:
            df = df.geo(crs)[slice_]
        elif type(df) == pd.DataFrame:
            geo = [Point(x, y) for x, y in zip(df["x"], df["y"])]
            df = gpd.GeoDataFrame(df, geometry=geo, crs=crs)
        else:
            pass

        return df

    def get_meta_object(self) -> DcdMetaData:
        cell_size = self.get_attribute("cell_size")
        cell_count = self.get_attribute("cell_count")
        cell_bound = self.get_attribute("cell_bound")
        epsg = self.get_attribute("epsg")
        offset = self.get_attribute("offset")
        map_extend_x = self.get_attribute("map_extend_x")
        map_extend_y = self.get_attribute("map_extend_y")

        return DcdMetaData(
            cell_size,
            cell_count,
            cell_bound,
            "global",
            offset=offset,
            epsg=str(epsg),
            map_extend_x=map_extend_x,
            map_extend_y=map_extend_y,
        )

    def _to_geo(
        self, df: pd.DataFrame, to_crs: Union[str, None] = None
    ) -> gpd.GeoDataFrame:
        offset = self.get_attribute("offset")
        epsg_code = self.get_attribute("epsg")
        cell_size_half = 0.5 * self.get_attribute("cell_size")

        df["x"] = df["x"] - offset[0]
        df["y"] = df["y"] - offset[1]

        g = [
            Point(x + cell_size_half, y + cell_size_half)
            for x, y in zip(df["x"], df["y"])
        ]
        gdf = gpd.GeoDataFrame(df, geometry=g, crs=str(epsg_code))
        if to_crs is not None:
            gdf = gdf.to_crs(epsg=to_crs.replace("EPSG:", ""))
        return gdf

    def geo_bound(self):
        bound = self.get_attribute("cell_bound")


class DcdGlobalDensity(IHdfProvider):
    def __init__(self, hdf_path, version: str | None = None):
        super().__init__(hdf_path, version)

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
        epsg = self.get_attribute("epsg")
        offset = self.get_attribute("offset")
        map_extend_x = self.get_attribute("map_extend_x")
        map_extend_y = self.get_attribute("map_extend_y")

        return DcdMetaData(
            cell_size,
            cell_count,
            cell_bound,
            "global",
            offset=offset,
            epsg=str(epsg),
            map_extend_x=map_extend_x,
            map_extend_y=map_extend_y,
        )

    def _to_geo(
        self, df: pd.DataFrame, to_crs: Union[str, None] = None
    ) -> gpd.GeoDataFrame:
        offset = self.get_attribute("offset")
        epsg_code = self.get_attribute("epsg")
        cell_size = self.get_attribute("cell_size")

        _index = df.index.to_frame().reset_index(drop=True)

        _index["x"] = _index["x"] - offset[0]
        _index["y"] = _index["y"] - offset[1]
        df.index = pd.MultiIndex.from_frame(_index)

        g = [
            box(x, y, x + cell_size, y + cell_size)
            for x, y in zip(_index["x"], _index["y"])
        ]
        gdf = gpd.GeoDataFrame(df, geometry=g, crs=str(epsg_code))
        if to_crs is not None:
            gdf = gdf.to_crs(epsg=to_crs.replace("EPSG:", ""))
        return gdf


def pos_density_from_csv(
    csv_path: str,
    hdf_path: str,
) -> Tuple[DcdGlobalPosition, DcdGlobalDensity, DcdMetaData]:
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

    return pos, density, meta
