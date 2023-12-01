from __future__ import annotations

from typing import Dict, List, Tuple, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from crownetutils.analysis.dpmm.csv_loader import read_csv
from crownetutils.analysis.dpmm.dpmm_cfg import DpmmCfgDb
from crownetutils.analysis.dpmm.dpmm_sql import DpmmSql
from crownetutils.analysis.dpmm.metadata import DpmmMetaData
from crownetutils.analysis.hdf.groups import HdfGroups
from crownetutils.analysis.hdf.provider import IHdfProvider
from crownetutils.utils.logging import logger, timing


class DpmMapKey:
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


class DpmmGlobalPosition(IHdfProvider):
    def __init__(self, hdf_path, version: str | None = None):
        super().__init__(hdf_path, version)

    def group_key(self) -> str:
        return HdfGroups.DCD_GLOBAL_POS

    def index_order(self) -> Dict:
        return {0: DpmMapKey.SIMTIME, 1: DpmMapKey.NODE_ID}

    def columns(self) -> List[str]:
        return [DpmMapKey.X, DpmMapKey.Y]

    def default_index_key(self) -> str:
        return self.index_order()[0]

    @staticmethod
    def as_geo(
        df: Union[DpmmGlobalPosition, pd.DataFrame, gpd.GeoDataFrame],
        crs: str,
        slice_: slice = slice(None),
    ) -> gpd.GeoDataFrame:
        if type(df) == DpmmGlobalPosition:
            df = df.geo(crs)[slice_]
        elif type(df) == pd.DataFrame:
            geo = [Point(x, y) for x, y in zip(df["x"], df["y"])]
            df = gpd.GeoDataFrame(df, geometry=geo, crs=crs)
        else:
            pass

        return df

    def get_meta_object(self) -> DpmmMetaData:
        cell_size = self.get_attribute("cell_size")
        cell_count = self.get_attribute("cell_count")
        cell_bound = self.get_attribute("cell_bound")
        epsg = self.get_attribute("epsg")
        offset = self.get_attribute("offset")
        map_extend_x = self.get_attribute("map_extend_x")
        map_extend_y = self.get_attribute("map_extend_y")

        return DpmmMetaData(
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
        sim_bound = self.get_sim_bound()
        epsg_code = self.get_attribute("epsg")
        cell_size_half = 0.5 * self.get_attribute("cell_size")

        df["x"] = df["x"] - sim_bound.offset[0] - sim_bound.sim_offset[0]
        df["y"] = df["y"] - sim_bound.offset[1] - sim_bound.sim_offset[1]

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


class DpmmGlobal(IHdfProvider):
    def __init__(self, hdf_path, version: str | None = None):
        super().__init__(hdf_path, version)

    def group_key(self) -> str:
        return HdfGroups.DCD_GLOBAL_DENSITY

    def index_order(self) -> Dict:
        return {0: DpmMapKey.SIMTIME, 1: DpmMapKey.X, 2: DpmMapKey.Y}

    def columns(self) -> List[str]:
        return [DpmMapKey.COUNT]

    def default_index_key(self) -> str:
        return self.index_order()[0]

    def get_meta_object(self) -> DpmmMetaData:
        cell_size = self.get_attribute("cell_size")
        cell_count = self.get_attribute("cell_count")
        cell_bound = self.get_attribute("cell_bound")
        epsg = self.get_attribute("epsg")
        offset = self.get_attribute("offset")
        map_extend_x = self.get_attribute("map_extend_x")
        map_extend_y = self.get_attribute("map_extend_y")

        return DpmmMetaData(
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
        sim_bound = self.get_sim_bound()

        epsg_code = self.get_attribute("epsg")
        cell_size = self.get_attribute("cell_size")

        _index = df.index.to_frame().reset_index(drop=True)

        _index["x"] = _index["x"] - sim_bound.offset[0] - sim_bound.sim_offset[0]
        _index["y"] = _index["y"] - sim_bound.offset[1] - sim_bound.sim_offset[1]
        df.index = pd.MultiIndex.from_frame(_index)

        g = [
            box(x, y, x + cell_size, y + cell_size)
            for x, y in zip(_index["x"], _index["y"])
        ]
        gdf = gpd.GeoDataFrame(df, geometry=g, crs=str(epsg_code))
        if to_crs is not None:
            gdf = gdf.to_crs(epsg=to_crs.replace("EPSG:", ""))
        return gdf


@timing
def create_and_save_position_and_global_db(cfg: DpmmCfgDb, hdf_path: str):
    sql: DpmmSql = DpmmSql(cfg)

    pos = DpmmGlobalPosition(hdf_path)
    density = DpmmGlobal(hdf_path)
    meta: DpmmMetaData = sql.glb_metadata()
    col_types = DpmMapKey.types_global_raw_csv_col
    del col_types[DpmMapKey.NODE_ID]  # node ids are in an extra table load seperatly
    global_df = sql.read_glb_map(
        index_types=DpmMapKey.types_global_raw_csv_index,
        col_types_dict=col_types,
        real_coords=True,
    )
    position_df = sql.read_global_position(
        index_types=DpmMapKey.types_global_raw_csv_index,
        real_coords=True,
    )

    position_df.set_index(
        keys=list(pos.index_order().values()), inplace=True, verify_integrity=True
    )
    logger.info(
        f"position_df: {position_df.memory_usage(index=True).sum()/1e6} MB with shape {position_df.shape}"
    )
    logger.info(
        f"global: {global_df.memory_usage(index=True).sum()/1e6} MB with shape {global_df.shape}"
    )
    pos.write_dataframe(position_df)
    density.write_dataframe(global_df)
    return pos, density, meta


@timing
def create_and_save_position_and_global(
    csv_path: str,
    hdf_path: str,
) -> Tuple[DpmmGlobalPosition, DpmmGlobal, DpmmMetaData]:
    pos = DpmmGlobalPosition(hdf_path)
    density = DpmmGlobal(hdf_path)
    global_df, meta = read_csv(
        csv_path=csv_path,
        _index_types=DpmMapKey.types_global_raw_csv_index,
        _col_types=DpmMapKey.types_global_raw_csv_col,
        real_coords=True,
    )
    position_df, global_df = build_position_df(global_df)
    position_df.set_index(
        keys=list(pos.index_order().values()), inplace=True, verify_integrity=True
    )
    logger.info(
        f"position_df: {position_df.memory_usage(index=True).sum()/1e6} MB with shape {position_df.shape}"
    )
    logger.info(
        f"global: {global_df.memory_usage(index=True).sum()/1e6} MB with shape {global_df.shape}"
    )
    pos.write_dataframe(position_df)
    density.write_dataframe(global_df)
    # density.repack_hdf(keep_old_file=False)
    return pos, density, meta
