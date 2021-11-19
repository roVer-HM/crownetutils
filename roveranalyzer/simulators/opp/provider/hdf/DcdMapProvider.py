import os
import re
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from roveranalyzer.simulators.crownet.dcd.util import (
    delay_feature,
    owner_dist_feature,
    read_csv,
)
from roveranalyzer.simulators.opp.provider.hdf.HdfGroups import HdfGroups
from roveranalyzer.simulators.opp.provider.hdf.IHdfProvider import (
    FrameConsumer,
    IHdfProvider,
)
from roveranalyzer.utils.misc import ProgressCmd


class DcdMapKey:
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
    # feature
    X_OWNER = "x_owner"
    Y_OWNER = "y_owner"
    OWNER_DIST = "owner_dist"
    DELAY = "delay"
    MEASURE_AGE = "measurement_age"
    UPDATE_AGE = "update_age"

    types_csv_index = {
        SIMTIME: float,
        X: float,
        Y: float,
        SOURCE: int,
        # node id is added later
    }

    types_csv_columns = {
        COUNT: float,
        MEASURE_TIME: float,
        RECEIVED_TIME: float,
        SELECTION: str,
        OWN_CELL: int,
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
    def columns(cls):
        return {**cls.types_csv_columns, **cls.types_features}

    @classmethod
    def col_list(cls):
        return list(cls.columns().keys())


class DcdMapProvider(IHdfProvider):
    def __init__(self, hdf_path):
        super().__init__(hdf_path)
        self.selection_mapping = {
            "NaN": 0,
            "ymf": 1,
            "invSourceDist": 2,
            "mean": 3,
            "median": 4,
        }
        self.node_regex = re.compile(r"dcdMap_(?P<node>\d+)\.csv")
        # some filter callbacks to apply to parsed csv before any further processing
        self.csv_filters = []

    def group_key(self) -> str:
        return HdfGroups.DCD_MAP

    def index_order(self) -> Dict:
        return {
            0: DcdMapKey.SIMTIME,
            1: DcdMapKey.X,
            2: DcdMapKey.Y,
            3: DcdMapKey.SOURCE,
            4: DcdMapKey.NODE,
        }

    def columns(self) -> List[str]:
        return DcdMapKey.col_list()

    def default_index_key(self) -> str:
        return DcdMapKey.SIMTIME

    def create_from_csv(
        self, csv_paths: List[str], frame_consumer: List[FrameConsumer] = [], **kwargs
    ) -> None:
        progress = ProgressCmd(prefix="read csv: ", cycle_count=len(csv_paths))
        for file_path in csv_paths:
            progress.incr()
            # build data frame from csv
            dcd_df = self.build_dcd_dataframe(file_path, **kwargs)

            # append to table but do not index (will be done at the end)
            with self.ctx() as store:
                store.append(
                    key=self.group,
                    value=dcd_df,
                    index=False,
                    format="table",
                    data_columns=True,
                )
            # send data frame to frame_consumers
            for consumer in frame_consumer:
                consumer.consume(dcd_df)

        # create index
        with self.ctx() as store:
            store.create_table_index(
                key=self.group,
                columns=list(self.index_order().values()),
                optlevel=9,
                kind="full",
            )
        self.set_selection_mapping_attribute()

    def parse_node_id(self, path: str) -> int:
        grps = [m.groupdict() for m in self.node_regex.finditer(path)]
        if not grps:
            raise ValueError(f"No node id found in: {path}")
        node_id = int(grps.pop()["node"])
        return node_id

    def build_dcd_dataframe(self, path: str, **kwargs) -> pd.DataFrame:
        df, meta = read_csv(
            csv_path=path,
            _index_types=DcdMapKey.types_csv_index,
            _col_types=DcdMapKey.types_csv_columns,
            real_coords=True,
            df_filter=self.csv_filters,
        )
        # add own node id
        df[DcdMapKey.NODE] = self.parse_node_id(path)
        # set index
        df = df.reset_index()
        index = list(self.index_order().values())
        df = df.set_index(keys=index, verify_integrity=True, drop=True)
        # cleanup string based column
        # ensure all keys in df are mapped to integer. Add new ones if needed.
        self.update_selection_map(df[DcdMapKey.SELECTION].unique().tolist())

        df[DcdMapKey.SELECTION] = df[DcdMapKey.SELECTION].fillna(
            self.selection_mapping["NaN"]
        )
        df[DcdMapKey.SELECTION] = df[DcdMapKey.SELECTION].replace(
            self.selection_mapping
        )

        # #####
        # apply features
        # #####

        num_rows = df.shape[0]

        # apply owner_dist_feature
        df = owner_dist_feature(df, **kwargs)
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
