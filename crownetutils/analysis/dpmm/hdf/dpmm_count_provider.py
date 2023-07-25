from __future__ import annotations

from distutils.version import Version
from typing import Dict, List, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from crownetutils.analysis.hdf.groups import HdfGroups
from crownetutils.analysis.hdf.operator import Operation
from crownetutils.analysis.hdf.provider import (
    IHdfProvider,
    ProviderVersion,
    VersionDict,
)


class DpmmCountKey:
    ID = "ID"
    SIMTIME = "simtime"
    X = "x"
    Y = "y"
    COUNT = "count"
    ERR = "err"
    OWNER_DIST = "owner_dist"
    SQERR = "sqerr"
    MISSING_VAL = "missing_value"
    # v 0.4
    RSD_ID = "rsd_id"

    columns = VersionDict(
        {
            ProviderVersion.V0_1: [
                COUNT,
                ERR,
                OWNER_DIST,
                SQERR,
            ],
            ProviderVersion.V0_3: [COUNT, ERR, OWNER_DIST, SQERR, MISSING_VAL],
            ProviderVersion.V0_4: [COUNT, ERR, OWNER_DIST, SQERR, MISSING_VAL, RSD_ID],
        }
    )

    index_order = VersionDict(
        {
            ProviderVersion.V0_1: {
                0: SIMTIME,
                1: X,
                2: Y,
                3: ID,
            }
        }
    )


class DpmmCount(IHdfProvider):
    def __init__(self, hdf_path, version: str | None = None):
        super().__init__(hdf_path, version)

    def group_key(self) -> str:
        return HdfGroups.COUNT_MAP

    def index_order(self) -> Dict:
        return DpmmCountKey.index_order[self.version]

    def columns(self) -> List[str]:
        return DpmmCountKey.columns[self.version]

    def default_index_key(self) -> str:
        return DpmmCountKey.SIMTIME

    #########################
    # Exact value functions #
    #########################
    def select_id_exact(
        self, value: int, operation: str = Operation.EQ
    ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(
            key=DpmmCountKey.ID, value=value, operation=operation
        )
        return self._select_where(condition=condition)

    def select_simtime_exact(
        self, value: int, operation: str = Operation.EQ
    ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(
            key=DpmmCountKey.SIMTIME, value=value, operation=operation
        )
        return self._select_where(condition=condition)

    def select_x_exact(
        self, value: float, operation: str = Operation.EQ
    ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(
            key=DpmmCountKey.X, value=value, operation=operation
        )
        return self._select_where(condition=condition)

    def select_y_exact(
        self, value: float, operation: str = Operation.EQ
    ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(
            key=DpmmCountKey.Y, value=value, operation=operation
        )
        return self._select_where(condition=condition)  # p[I[None,None,5,None]]

    def select_count_exact(
        self, value: float, operation: str = Operation.EQ
    ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(
            key=DpmmCountKey.COUNT, value=value, operation=operation
        )
        return self._select_where(condition=condition)

    def select_err_exact(
        self, value: float, operation: str = Operation.EQ
    ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(
            key=DpmmCountKey.ERR, value=value, operation=operation
        )
        return self._select_where(condition=condition)

    def select_owner_dist_exact(
        self, value: float, operation: str = Operation.EQ
    ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(
            key=DpmmCountKey.OWNER_DIST, value=value, operation=operation
        )
        return self._select_where(condition=condition)

    def select_sqerr_exact(
        self, value: float, operation: str = Operation.EQ
    ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(
            key=DpmmCountKey.SQERR, value=value, operation=operation
        )
        return self._select_where(condition=condition)

    def select_simtime_and_node_id_exact(
        self, simtime: int, node_id: int, operation: str = Operation.EQ
    ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(
            key=DpmmCountKey.SIMTIME, value=simtime, operation=operation
        ) + self._build_exact_condition(
            key=DpmmCountKey.ID, value=node_id, operation=operation
        )
        return self._select_where(condition=condition)

    #########################
    # Range value functions #
    #########################
    def select_id_range(self, _min: int, _max: int) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(
            key=DpmmCountKey.ID, _min=_min, _max=_max
        )
        return self._select_where(condition=condition)

    def select_simtime_range(self, _min: int, _max: int) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(
            key=DpmmCountKey.SIMTIME, _min=_min, _max=_max
        )
        return self._select_where(condition=condition)

    def select_x_range(self, _min: float, _max: float) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(
            key=DpmmCountKey.X, _min=_min, _max=_max
        )
        return self._select_where(condition=condition)

    def select_y_range(self, _min: float, _max: float) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(
            key=DpmmCountKey.Y, _min=_min, _max=_max
        )
        return self._select_where(condition=condition)

    def select_count_range(self, _min: float, _max: float) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(
            key=DpmmCountKey.COUNT, _min=_min, _max=_max
        )
        return self._select_where(condition=condition)

    def select_err_range(self, _min: float, _max: float) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(
            key=DpmmCountKey.ERR, _min=_min, _max=_max
        )
        return self._select_where(condition=condition)

    def select_owner_dist_range(self, _min: float, _max: float) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(
            key=DpmmCountKey.OWNER_DIST, _min=_min, _max=_max
        )
        return self._select_where(condition=condition)

    def select_sqerr_range(self, _min: float, _max: float) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(
            key=DpmmCountKey.SQERR, _min=_min, _max=_max
        )
        return self._select_where(condition=condition)

    def _to_geo(
        self, df: pd.DataFrame, to_crs: Union[str, None] = None
    ) -> gpd.GeoDataFrame:
        sim_bound = self.get_sim_bound()
        epsg_code = self.get_attribute("epsg")
        cell_size = self.get_attribute("cell_size")

        _index = df.index.to_frame().reset_index(drop=True)
        # keep cell (ids)
        df = df.reset_index(drop=True)
        df["cell_x"] = _index["x"]
        df["cell_y"] = _index["y"]

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
