from __future__ import annotations

import glob
import json
import multiprocessing
import os
import pickle
from functools import partial
from typing import Union

import numpy as np
import pandas as pd
from pandas import IndexSlice as Idx

import crownetutils.analysis.dpmm.csv_loader as DcdUtil
from crownetutils.analysis.dpmm.dpmm import DpmMap, DpmMapMulti, MapType
from crownetutils.analysis.dpmm.hdf.dpmm_count_provider import DpmmCount
from crownetutils.analysis.dpmm.hdf.dpmm_global_positon_provider import (
    DpmmGlobal,
    DpmmGlobalPosition,
    pos_density_from_csv,
)
from crownetutils.analysis.dpmm.hdf.dpmm_provider import DpmmProvider
from crownetutils.analysis.dpmm.metadata import DpmmMetaData
from crownetutils.analysis.hdf.provider import ProviderVersion
from crownetutils.utils.dataframe import (
    ArbitraryValueImputation,
    FrameConsumer,
    MissingValueImputationStrategy,
)
from crownetutils.utils.logging import logging
from crownetutils.vadere.plot.topgraphy_plotter import VadereTopographyPlotter


def _hdf_job(args):
    input = args[0:-2]
    override_existing = args[-1]
    _filter = args[-2]
    _builder = DpmmHdfBuilder.get(*input)
    _builder.single_df_filters.extend(_filter)
    if override_existing or not _builder.hdf_exist:
        _builder.remove_hdf()
        # append filters before processing
        _builder.map_p.csv_filters.extend(_builder.single_df_filters)
        _builder.create_hdf_fast()
    else:
        print(f"{_builder.hdf_path} already exist and override_existing is false")


class DpmmProviders:
    def __init__(
        self,
        metadata,
        global_p,
        position_p,
        map_p,
        count_p,
        time_slice,
        id_slice,
        x_slice,
        y_slice,
    ):
        self.metadata: DpmmMetaData = metadata
        self.global_p: DpmmGlobal = global_p
        self.position_p: DpmmGlobalPosition = position_p
        self.map_p: DpmmProvider = map_p
        self.count_p: DpmmCount = count_p

        self.id_slice: slice = id_slice
        self.x_slice: slice = x_slice
        self.y_slice: slice = y_slice
        self.time_slice: slice = time_slice
        self.count_slice: Idx = Idx[time_slice, x_slice, y_slice, id_slice]
        self.map_slice: Idx = Idx[time_slice, x_slice, y_slice, :, id_slice]
        self.global_slice: Idx = Idx[time_slice, x_slice, y_slice]
        self.postion_slice: Idx = Idx[time_slice, id_slice]

    @property
    def global_df(self):
        return self.global_p[self.global_slice]

    @property
    def postion_df(self):
        return self.position_p[self.postion_slice]


class DpmmHdfBuilder(FrameConsumer):
    F_selected_only = DcdUtil.remove_not_selected_cells

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.create_count_map(df)

    @classmethod
    def get(
        cls, hdf_path, source_path, map_glob="dcdMap_*.csv", global_name="global.csv"
    ):
        if not os.path.isabs(hdf_path):
            hdf_path = os.path.join(source_path, hdf_path)
        return cls(
            hdf_path=hdf_path,
            map_paths=glob.glob(os.path.join(source_path, map_glob)),
            global_path=os.path.join(source_path, global_name),
        )

    @classmethod
    def create(
        cls,
        job_list,
        n_jobs: Union[int, float] = 0.6,
        override_existing=False,
        _filter=None,
    ):
        """
        job_list:  [[hdf_name, source_path, map_glob, global_name], ..., []]
        n_jobs:    number of parallel jobs or percentage of number of cpus to use
        override_existing: if true delete hdf_name and recreate it.
        """
        if isinstance(n_jobs, int):
            if n_jobs <= 0:
                n_jobs = multiprocessing.cpu_count()
            else:
                n_jobs = min(n_jobs, multiprocessing.cpu_count())
        elif isinstance(n_jobs, float):
            n_jobs = min(
                int(multiprocessing.cpu_count() * n_jobs), multiprocessing.cpu_count()
            )
        else:
            n_jobs = 1

        pool = multiprocessing.Pool(processes=n_jobs)
        if _filter is None:
            _filter = []
        job_list = [[*i, _filter, override_existing] for i in job_list]
        pool.map(_hdf_job, job_list)

    def __init__(self, hdf_path, map_paths, global_path, epsg=""):
        super().__init__()
        # paths
        self.hdf_path = hdf_path
        self.map_paths = map_paths
        self.global_path = global_path
        # providers
        self.count_p = DpmmCount(self.hdf_path)
        self.map_p = DpmmProvider(self.hdf_path)
        self.position_p = DpmmGlobalPosition(self.hdf_path)
        self.global_p = DpmmGlobal(self.hdf_path)
        # options:
        # filters used during csv processing
        self.single_df_filters = []
        self._only_selected_cells = True
        self._epsg = epsg
        self._imputation_function = ArbitraryValueImputation(0.0)
        self._map_type: MapType = MapType.DENSITY

        # set later on
        self.global_df = None
        self.position_df = None
        self._all_times = None

    def get_selected_alg(self):
        sel = self.build().map_p.get_attribute("used_selection")
        if len(sel) > 1:
            print(f"multiple selections found: {sel}")
        return list(sel)[0]

    def epsg(self, epsg):
        self._epsg = epsg
        return self

    def only_selected_cells(self, val=True):
        self._only_selected_cells = val
        return self

    def set_imputation_strategy(self, s: MissingValueImputationStrategy):
        self._imputation_function = s

    def set_map_type(self, t: MapType):
        self._map_type = t

    @property
    def map_type(self) -> MapType:
        return self._map_type

    def build(
        self,
        time_slice: slice = slice(None),
        id_slice: slice = slice(None),
        x_slice: slice = slice(None),
        y_slice: slice = slice(None),
        override_hdf=False,
    ) -> DpmmProviders:
        if not self.hdf_exist or override_hdf:
            try:
                os.remove(self.hdf_path)
            except FileNotFoundError:
                pass
            print(f"create HDF {self.hdf_path}")
            # append filters before processing
            self.map_p.csv_filters.extend(self.single_df_filters)
            self.create_hdf_fast()

        metadata = DpmmMetaData(
            cell_size=self.position_p.get_attribute("cell_size"),
            cell_count=self.position_p.get_attribute("cell_count"),
            bound=self.position_p.get_attribute("cell_bound"),
            offset=self.position_p.get_attribute("offset"),
            epsg=self.position_p.get_attribute("epsg"),
            version=ProviderVersion.from_str(
                self.position_p.get_attribute(
                    "version", default=ProviderVersion.current().value
                )
            ),
            node_id=0,
        )
        return DpmmProviders(
            metadata=metadata,
            global_p=self.global_p,
            position_p=self.position_p,
            map_p=self.map_p,
            count_p=self.count_p,
            time_slice=time_slice,
            id_slice=id_slice,
            x_slice=x_slice,
            y_slice=y_slice,
        )

    def build_dcdMap(
        self,
        time_slice: slice = slice(None),
        id_slice: slice = slice(None),
        x_slice: slice = slice(None),
        y_slice: slice = slice(None),
        selection: str | int = "ymf",
    ):
        if self._only_selected_cells:
            self.add_df_filter(DcdUtil.remove_not_selected_cells)
        providers = self.build(time_slice, id_slice, x_slice, y_slice)

        if isinstance(selection, str):
            m = providers.map_p.get_selection_mapping_attribute()
            if selection not in m:
                logging.logging.error(
                    f"selection string '{selection}'not found expected one of"
                    f"[{','.join(m.keys())}]"
                )
            selection = m[selection]

        return DpmMap(
            metadata=providers.metadata,
            global_df=providers.global_df,
            map_df=None,  # lazy loading with provider
            position_df=providers.postion_df,
            count_p=providers.count_p,
            count_slice=providers.count_slice,
            map_p=providers.map_p.add_filter(
                selection=selection
            ),  # ensure only selected cells are load
            map_slice=providers.map_slice,
        )

    def build_dcdMapMulti(
        self,
        time_slice: slice = slice(None),
        id_slice: slice = slice(None),
        x_slice: slice = slice(None),
        y_slice: slice = slice(None),
    ):
        raise NotImplemented("")

    def add_df_filter(self, filter):
        self.single_df_filters.append(filter)

    @property
    def hdf_exist(self):
        return os.path.exists(self.hdf_path)

    def remove_hdf(self):
        if self.hdf_exist:
            os.remove(self.hdf_path)

    def create_hdf_fast(self):
        t = DcdUtil.Timer.create_and_start("create_hdf", label="")
        # 1) parse global.csv in position and global provider
        self.position_p, self.global_p, meta = pos_density_from_csv(
            self.global_path, self.hdf_path
        )
        # 2) access global_df and setup helpers for parsing map_*.csv to create
        #    map and count provider together
        self.global_df = self.global_p.get_dataframe()
        self.position_df = self.position_p.get_dataframe()
        self._all_times = (
            self.global_df.index.get_level_values("simtime")
            .unique()
            .sort_values()
            .to_numpy()
        )
        # add self as frame_consumer to build count_map iteratively
        self.map_p.create_from_csv(
            self.map_paths,
            frame_consumer=[
                partial(self.create_count_map, imputation_f=self._imputation_function)
            ],
            global_position=self.position_df,
            global_metadata=meta,
        )
        # 3) append global count to count provider
        self.append_global_count()
        # 4) create index on count_map_provider
        with self.count_p.ctx() as store:
            store.create_table_index(
                key=self.count_p.group,
                columns=list(self.count_p.index_order().values()),
                optlevel=9,
                kind="full",
            )

        # 5) set attributes
        for p in [self.position_p, self.global_p, self.map_p, self.count_p]:
            p.set_attribute("cell_size", meta.cell_size)
            p.set_attribute("cell_count", meta.cell_count)
            p.set_attribute("cell_bound", meta.bound)
            p.set_attribute("offset", meta.offset)
            p.set_attribute("epsg", self._epsg)
            p.set_attribute("version", meta.version.as_string())
            p.set_attribute("data_type", meta.data_type)
            p.set_attribute(
                "time_interval", [np.min(self._all_times), np.max(self._all_times)]
            )
            # _cell_bound is the whole simulation area. Map_extends only gives the area of the density map
            p.set_attribute(
                "map_extend_x",
                [
                    self.global_df.index.get_level_values("x").min(),
                    self.global_df.index.get_level_values("x").max(),
                ],
            )
            p.set_attribute(
                "map_extend_y",
                [
                    self.global_df.index.get_level_values("y").min(),
                    self.global_df.index.get_level_values("y").max(),
                ],
            )
            p.set_attribute(
                "id_interval",
                [
                    self.position_df.index.get_level_values("node_id").min(),
                    self.position_df.index.get_level_values("node_id").max(),
                ],
            )

        t.stop()
        return {
            "glb": self.global_p,
            "pos": self.position_p,
            "map": self.map_p,
            "count": self.count_p,
        }

    def create_hdf(self):
        t = DcdUtil.Timer.create_and_start("create_hdf", label="")
        print("build global")
        self.position_p, self.global_p = pos_density_from_csv(
            self.global_path, self.hdf_path
        )
        print("build dcd map")
        self.map_p.create_from_csv(self.map_paths)
        print("build count map")
        count_df = DcdUtil.create_error_df(
            self.map_p.get_dataframe(), self.global_p.get_dataframe()
        )
        self.count_p.write_dataframe(count_df)
        t.stop()
        print("done")
        return {
            "glb": self.global_p,
            "pos": self.position_p,
            "map": self.map_p,
            "count": self.count_p,
        }

    def create_count_map(
        self,
        df: pd.DataFrame,
        imputation_f: MissingValueImputationStrategy = ArbitraryValueImputation(),
    ):
        """
        Creates dataframe of the form:
          index: [simtime, x, y, ID]
          columns: [count, err, sqerr, owner_dist]
        df: Input data frame of one node containing count values seen by this node. It is possible
            that the node didn't see all occupied cells. To fill the gaps the data frame is concatednated
            with the ground truth to fill the missing cell values with zero (i.e. maximal error!)
        """
        if df.empty:
            # ignore empty frames (nodes which do not have a density map)
            # (mostly artifacts at end of simulation. Node created at end and
            # simulation finished before the item is logged)
            return
        # only use selected values
        _df = df[df["selection"] != 0].copy(deep=True)
        # extract id, times and positions from data frame
        id = _df.index.get_level_values("ID").unique()[0]
        present_at_times = (
            _df.index.get_level_values("simtime").unique().sort_values().to_numpy()
        )
        not_present_at_times = np.setdiff1d(self._all_times, present_at_times)
        positions = _df.loc[:, ["x_owner", "y_owner"]].droplevel(
            ["x", "y", "ID", "source"]
        )
        positions = positions[np.invert(positions.index.duplicated(keep="first"))]

        # get count and node position
        _df = _df.loc[:, ["count", "x_owner", "y_owner"]].droplevel(["source", "ID"])
        # merge with global, rename columns and fill glb_count nan with '0'
        # fill only global. The index where this happens are values where
        # the global map does not have any values -> thus count=0
        _df = pd.concat([self.global_df, _df], axis=1)
        _df.columns = ["glb_count", "count", "x_owner", "y_owner"]
        # add marker column for which data imputation is used.
        missing_value_idx = _df[_df["count"].isna().values].index
        _df["missing_value"] = False
        _df.loc[missing_value_idx, ["missing_value"]] = True
        # use arbitrary value imputation with value=0
        # For the default scenario (counting pedestrians via beacons) a count of
        # zero (i.e. value=0) is a reasonable assumption because in the case of
        # perfect reception and zero package loss no information from a given cell
        # translates to no pedestrian in this cell, thus a count of zero. For other
        # measurements this assumption is not automatically right and other imputation
        # methods or deletion might be better.
        _df = imputation_f(_df, "count")
        # _df = _df.fillna(0)

        # remove times where node is not part of simulation
        _df = _df.loc[Idx[present_at_times, :, :], :]

        # fill missing owner position values
        for _time in present_at_times:
            _df.loc[Idx[_time, :, :], "x_owner"] = positions.loc[_time, ["x_owner"]][0]
            _df.loc[Idx[_time, :, :], "y_owner"] = positions.loc[_time, ["y_owner"]][0]

        _x_idx = _df.index.get_level_values("x")
        _y_idx = _df.index.get_level_values("y")
        _df["err"] = _df["count"] - _df["glb_count"]
        _df["sqerr"] = _df["err"] ** 2
        _df["owner_dist"] = np.sqrt(
            (_df["x_owner"] - _x_idx) ** 2 + (_df["y_owner"] - _y_idx) ** 2
        )
        _df = _df.drop(columns=["glb_count", "x_owner", "y_owner"])
        _df["ID"] = id
        _df = _df.set_index(["ID"], drop=True, append=True)
        # _df["count"] = _df["count"].astype(int)
        # _df["err"] = _df["err"].astype(int)
        self.append_to_provider(self.count_p, _df)

    def append_global_count(self):
        _df = self.global_df.copy()
        _df["ID"] = 0  # global id set to 0
        _df["ID"].convert_dtypes(int)
        _df["err"] = 0.0
        _df["sqerr"] = 0.0
        _df["owner_dist"] = 0.0
        _df["missing_value"] = False
        _df = _df.set_index(["ID"], drop=True, append=True)
        self.append_to_provider(self.count_p, _df)

    @staticmethod
    def append_to_provider(provider, df: pd.DataFrame):
        with provider.ctx() as store:
            store.append(key=provider.group, value=df, index=False, data_columns=True)
