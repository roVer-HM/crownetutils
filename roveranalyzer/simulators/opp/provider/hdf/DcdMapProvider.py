import os
import re
from typing import List

import pandas as pd

from roveranalyzer.simulators.opp.provider.hdf.HdfGroups import HdfGroups
from roveranalyzer.simulators.opp.provider.hdf.IHdfProvider import IHdfProvider


class DcdMapKey:
    SIMTIME = "simtime"
    X = "x"
    Y = "y"
    SOURCE = "source"
    NODE = "node"
    COUNT = "count"
    MEASURE_TIME = "measure_t"
    RECEIVED_TIME = "received_t"
    SELECTION = "selection"
    OWN_CELL = "own_cell"


class DcdMapProvider(IHdfProvider):
    def __init__(self, hdf_path):
        super().__init__(hdf_path)
        self.selection_mapping = {"NaN": 0, "ymf": 1}
        self.node_regex = re.compile(r"dcdMap_(?P<node>\d+)\.csv")

    def group_key(self) -> str:
        return HdfGroups.DCD_MAP

    def index_order(self) -> {}:
        return {
            0: DcdMapKey.SIMTIME,
            1: DcdMapKey.X,
            2: DcdMapKey.Y,
            3: DcdMapKey.SOURCE,
            4: DcdMapKey.NODE,
        }

    def columns(self) -> List[str]:
        return [
            DcdMapKey.COUNT,
            DcdMapKey.MEASURE_TIME,
            DcdMapKey.RECEIVED_TIME,
            DcdMapKey.SELECTION,
            DcdMapKey.OWN_CELL,
        ]

    def default_index_key(self) -> str:
        return DcdMapKey.SIMTIME

    def create_from_csv(self, csv_paths: List[str]) -> None:
        for file_path in csv_paths:
            dcd_df = self.build_dcd_dataframe(file_path)
            with self.ctx() as store:
                store.append(
                    key=self.group, value=dcd_df, format="table", data_columns=True
                )
        self.set_selection_mapping_attribute()

    def parse_node_id(self, path: str) -> int:
        grps = [m.groupdict() for m in self.node_regex.finditer(path)]
        if not grps:
            raise ValueError(f"No node id found in: {path}")
        node_id = int(grps.pop()["node"])
        return node_id

    def build_dcd_dataframe(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(filepath_or_buffer=path, sep=";", header=1)
        df["node"] = self.parse_node_id(path)
        index = [i[1] for i in self.index_order().items()]
        df.set_index(keys=index, inplace=True)
        df[DcdMapKey.SELECTION] = df[DcdMapKey.SELECTION].fillna(
            self.selection_mapping["NaN"]
        )
        df[DcdMapKey.SELECTION] = df[DcdMapKey.SELECTION].replace(
            self.selection_mapping
        )
        return df

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
