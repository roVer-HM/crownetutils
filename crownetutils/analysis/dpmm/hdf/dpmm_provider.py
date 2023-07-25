from __future__ import annotations

import os
import re
from typing import Dict, List, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.core.indexing import IndexSlice
from pandas.errors import EmptyDataError
from shapely.geometry.geo import box

from crownetutils.analysis.dpmm.csv_loader import (
    delay_feature,
    owner_dist_feature,
    read_csv,
)
from crownetutils.analysis.dpmm.metadata import DpmmMetaData
from crownetutils.analysis.hdf.groups import HdfGroups
from crownetutils.analysis.hdf.provider import (
    IHdfProvider,
    ProviderVersion,
    VersionDict,
)
from crownetutils.utils.dataframe import FrameConsumer, LazyDataFrame
from crownetutils.utils.logging import logger
from crownetutils.utils.misc import ProgressCmd


class DpmmKey:
    # index
    SIMTIME = "simtime"
    X = "x"
    Y = "y"
    SOURCE = "source"
    NODE = "ID"
    # columns csv
    COUNT = "count"
    MEASURE_TIME = "measured_t"
    RECEIVED_TIME = "received_t"
    SELECTION = "selection"
    OWN_CELL = "own_cell"
    # v 0.2
    SOURCE_HOST = "sourceHost"
    SOURCE_ENTRY = "sourceEntry"
    HOST_ENTRY = "hostEntry"
    # v 0.4
    RSD_ID = "rsd_id"
    # feature
    X_OWNER = "x_owner"
    Y_OWNER = "y_owner"
    OWNER_DIST = "owner_dist"
    DELAY = "delay"
    MEASURE_AGE = "measurement_age"
    UPDATE_AGE = "update_age"
    SELECTION_RANK = "selectionRank"

    types_csv_index = VersionDict(
        {
            ProviderVersion.V0_1: {
                SIMTIME: float,
                X: float,
                Y: float,
                SOURCE: int,
                # node id is added later
            },
            ProviderVersion.V0_2: {
                SIMTIME: float,
                X: float,
                Y: float,
                SOURCE: int,
                # node id is added later
            },
        }
    )

    types_csv_columns = VersionDict(
        {
            ProviderVersion.V0_1: {
                COUNT: float,
                MEASURE_TIME: float,
                RECEIVED_TIME: float,
                SELECTION: str,
                OWN_CELL: int,
            },
            ProviderVersion.V0_2: {
                COUNT: float,
                MEASURE_TIME: float,
                RECEIVED_TIME: float,
                SELECTION: str,
                OWN_CELL: int,
                SOURCE_HOST: float,
                SOURCE_ENTRY: float,
                HOST_ENTRY: float,
                SELECTION_RANK: float,
            },
            ProviderVersion.V0_4: {
                COUNT: float,
                MEASURE_TIME: float,
                RECEIVED_TIME: float,
                SELECTION: str,
                OWN_CELL: int,
                SOURCE_HOST: float,
                SOURCE_ENTRY: float,
                HOST_ENTRY: float,
                SELECTION_RANK: float,
                RSD_ID: float,
            },
        }
    )

    count_map_creation_cols = {
        ProviderVersion.V0_1: [COUNT, X_OWNER, Y_OWNER],
        ProviderVersion.V0_4: [COUNT, X_OWNER, Y_OWNER, RSD_ID],
    }

    types_features = {
        X_OWNER: float,
        Y_OWNER: float,
        OWNER_DIST: float,
        DELAY: float,
        MEASURE_AGE: float,
        UPDATE_AGE: float,
    }

    @classmethod
    def columns(cls, version: ProviderVersion = ProviderVersion.current()):
        v = ProviderVersion.current() if version is None else version
        return {**cls.types_csv_columns[v], **cls.types_features}

    @classmethod
    def col_list(cls, version: str | None = None):
        return list(cls.columns(version).keys())


class DpmmProvider(IHdfProvider):
    def __init__(self, hdf_path, version: str | None = None):
        super().__init__(hdf_path, version)
        self.selection_mapping = {
            "NaN": 0,
            "ymf": 1,
            "invSourceDist": 2,
            "mean": 3,
            "median": 4,
            "ymfPlusDist": 5,
            "ymfPlusDistStep": 6,
        }
        self.used_selection = set()
        self.node_regex = re.compile(r"dcdMap_(?P<node>\d+)\.csv")
        # some filter callbacks to apply to parsed csv before any further processing
        self.csv_filters = []

    def group_key(self) -> str:
        return HdfGroups.DCD_MAP

    def index_order(self) -> Dict:
        return {
            0: DpmmKey.SIMTIME,
            1: DpmmKey.X,
            2: DpmmKey.Y,
            3: DpmmKey.SOURCE,
            4: DpmmKey.NODE,
        }

    def columns(self) -> List[str]:
        return DpmmKey.col_list(self.version)

    def default_index_key(self) -> str:
        return DpmmKey.SIMTIME

    def create_from_csv(
        self, csv_paths: List[str], frame_consumer: List[FrameConsumer] = [], **kwargs
    ) -> None:
        progress = ProgressCmd(prefix="read csv: ", cycle_count=len(csv_paths))
        for file_path in csv_paths:
            progress.incr()
            # build data frame from csv
            try:
                dcd_df = self.build_dcd_dataframe(file_path, **kwargs)
            except EmptyDataError as e:
                logger.warning(f"Empty DPMM file. Skip {file_path}")
                continue

            # append to table but do not index (will be done at the end)
            with self.ctx() as store:
                store.append(
                    key=self.group,
                    value=dcd_df,
                    index=True,
                    format="table",
                    data_columns=True,
                )
            # send data frame to frame_consumers
            for consumer in frame_consumer:
                consumer(dcd_df)

        self.set_selection_mapping_attribute()
        self.set_used_selection_attribute()

    def parse_node_id(self, path: str) -> int:
        grps = [m.groupdict() for m in self.node_regex.finditer(path)]
        if not grps:
            raise ValueError(f"No node id found in: {path}")
        node_id = int(grps.pop()["node"])
        return node_id

    def build_dcd_dataframe(self, path: str, **kwargs) -> pd.DataFrame:
        _df = LazyDataFrame.from_path(path)
        meta = _df.read_meta_data()
        meta = DpmmMetaData.from_dict(meta)
        if meta.version != self.version:
            logger.warn(f"version missmatch {meta.version}!={self.version} in {path}")

        df, meta = read_csv(
            csv_path=path,
            _index_types=DpmmKey.types_csv_index[meta.version],
            _col_types=DpmmKey.types_csv_columns[meta.version],
            real_coords=True,
            df_filter=self.csv_filters,
        )
        # add own node id
        df[DpmmKey.NODE] = self.parse_node_id(path)
        # set index
        df = df.reset_index()
        index = list(self.index_order().values())
        df = df.set_index(keys=index, verify_integrity=True, drop=True)
        # cleanup string based column
        # ensure all keys in df are mapped to integer. Add new ones if needed.
        self.update_selection_map(df[DpmmKey.SELECTION].unique().tolist())

        df[DpmmKey.SELECTION] = df[DpmmKey.SELECTION].fillna(
            self.selection_mapping["NaN"]
        )
        df[DpmmKey.SELECTION] = df[DpmmKey.SELECTION].replace(self.selection_mapping)

        # #####
        # apply features
        # #####

        num_rows = df.shape[0]

        # apply owner_dist_feature
        df = owner_dist_feature(df, meta, **kwargs)
        if df.shape[0] != num_rows:
            raise RuntimeError(
                "Inconsistency detected in owner_dist_feature. "
                f"Number of rows were affected. actual: {df.shape[0]} expected: {num_rows} "
            )  # shape = df.shape
        # apply delay_feature
        df = delay_feature(df, **kwargs)
        if df.shape[0] != num_rows:
            raise RuntimeError(
                "Inconsistency detected in delay_feature. "
                f"Number of rows were affected. actual: {df.shape[0]} expected: {num_rows} "
            )

        return df

    def update_selection_map(self, keys):
        next_idx = max(self.selection_mapping.values()) + 1
        for k in keys:
            if isinstance(k, float) and np.isnan(k):
                # ignore nan
                continue
            if k not in self.selection_mapping:
                self.selection_mapping[k] = next_idx
                print(f"found new selection algorithm. Map {k} -> {next_idx}")
                next_idx += 1
            if self.selection_mapping[k] not in self.used_selection:
                self.used_selection.add(self.selection_mapping[k])

    def get_dcd_file_paths(self, base_path: str) -> List[str]:
        dcd_files = []
        for root, dirs, files in os.walk(base_path):
            for name in files:
                if self.node_regex.search(name):
                    dcd_files.append(os.path.join(root, name))
        return dcd_files

    def get_selection_mapping_attribute(self):
        return self.get_attribute("selection_mapping")

    def set_selection_mapping_attribute(self):
        self.set_attribute("selection_mapping", self.selection_mapping)

    def set_used_selection_attribute(self):
        self.set_attribute("used_selection", self.used_selection)

    def _to_geo(
        self, df: pd.DataFrame, to_crs: Union[str, None] = None
    ) -> gpd.GeoDataFrame:
        sim_bound = self.get_sim_bound()

        epsg_code = self.get_attribute("epsg")
        cell_size = self.get_attribute("cell_size")

        _index = df.index.to_frame().reset_index(drop=True)
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
