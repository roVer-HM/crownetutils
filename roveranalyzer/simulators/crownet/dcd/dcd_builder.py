from __future__ import annotations

import copy
import glob
import json
import multiprocessing
import os
import pickle
from typing import Union

import numpy as np
import pandas as pd
from pandas import IndexSlice as Idx

import roveranalyzer.simulators.crownet.common.dcd_util as DcdUtil
from roveranalyzer.simulators.crownet.common import DcdMetaData
from roveranalyzer.simulators.crownet.dcd.dcd_map import DcdMap2D, DcdMap2DMulti
from roveranalyzer.simulators.opp.provider.hdf.DcDGlobalPosition import (
    DcdGlobalDensity,
    DcdGlobalPosition,
    pos_density_from_csv,
)
from roveranalyzer.simulators.opp.provider.hdf.DcdMapCountProvider import DcdMapCount
from roveranalyzer.simulators.opp.provider.hdf.DcdMapProvider import DcdMapProvider
from roveranalyzer.simulators.opp.provider.hdf.IHdfProvider import FrameConsumer
from roveranalyzer.simulators.vadere.plots.scenario import VaderScenarioPlotHelper
from roveranalyzer.utils import logging


def _hdf_job(args):
    input = args[0:-2]
    override_existing = args[-1]
    _filter = args[-2]
    _builder = DcdHdfBuilder.get(*input)
    _builder.single_df_filters.extend(_filter)
    if override_existing or not _builder.hdf_exist:
        _builder.remove_hdf()
        # append filters before processing
        _builder.map_p.csv_filters.extend(_builder.single_df_filters)
        _builder.create_hdf_fast()
    else:
        print(f"{_builder.hdf_path} already exist and override_existing is false")


class DcdProviders:
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
        self.metadata: DcdMetaData = metadata
        self.global_p: DcdGlobalDensity = global_p
        self.position_p: DcdGlobalPosition = position_p
        self.map_p: DcdMapProvider = map_p
        self.count_p: DcdMapCount = count_p

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


class DcdHdfBuilder(FrameConsumer):

    F_selected_only = DcdUtil.remove_not_selected_cells

    def consume(self, df: pd.DataFrame):
        self.create_count_map(df)

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
        self._epsg = epsg
        # providers
        self.count_p = DcdMapCount(self.hdf_path)
        self.map_p = DcdMapProvider(self.hdf_path)
        self.position_p = DcdGlobalPosition(self.hdf_path)
        self.global_p = DcdGlobalDensity(self.hdf_path)
        # filters used during csv processing
        self.single_df_filters = []

        # set later on
        self.global_df = None
        self.position_df = None
        self._all_times = None

    def epsg(self, epsg):
        self._epsg = epsg
        return self

    def build(
        self,
        time_slice: slice = slice(None),
        id_slice: slice = slice(None),
        x_slice: slice = slice(None),
        y_slice: slice = slice(None),
        override_hdf=False,
    ) -> DcdProviders:
        if not self.hdf_exist or override_hdf:
            try:
                os.remove(self.hdf_path)
            except FileNotFoundError:
                pass
            print(f"create HDF {self.hdf_path}")
            # append filters before processing
            self.map_p.csv_filters.extend(self.single_df_filters)
            self.create_hdf_fast()

        metadata = DcdMetaData(
            cell_size=self.position_p.get_attribute("cell_size"),
            cell_count=self.position_p.get_attribute("cell_count"),
            bound=self.position_p.get_attribute("cell_bound"),
            offset=self.position_p.get_attribute("offset"),
            epsg=self.position_p.get_attribute("epsg"),
            node_id=0,
        )
        return DcdProviders(
            metadata=copy.deepcopy(metadata),
            global_p=copy.deepcopy(self.global_p),
            position_p=copy.deepcopy(self.position_p),
            map_p=copy.deepcopy(self.map_p),
            count_p=copy.deepcopy(self.count_p),
            time_slice=copy.deepcopy(time_slice),
            id_slice=copy.deepcopy(id_slice),
            x_slice=copy.deepcopy(x_slice),
            y_slice=copy.deepcopy(y_slice),
        )

    def build_dcdMap(
        self,
        time_slice: slice = slice(None),
        id_slice: slice = slice(None),
        x_slice: slice = slice(None),
        y_slice: slice = slice(None),
        selection: str | int = "ymf",
    ):
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

        return DcdMap2D(
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
        self.map_p.create_from_csv(
            self.map_paths, [self], global_position=self.position_df
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

    def create_count_map(self, df: pd.DataFrame):
        """
        Creates dataframe of the form:
          index: [simtime, x, y, ID]
          columns: [count, err, sqerr, owner_dist]
        df: Input data frame of one node containing count values seen by this node. It is possible
            that the node didn't see all occupied cells. To fill the gaps the data frame is concatednated
            with the ground truth to fill the missing cell values with zero (i.e. maximal error!)
        """
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
        # _df["glb_count"] = _df["glb_count"].fillna(0)
        _df = _df.fillna(0)

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
        _df = _df.set_index(["ID"], drop=True, append=True)
        self.append_to_provider(self.count_p, _df)

    @staticmethod
    def append_to_provider(provider, df: pd.DataFrame):
        with provider.ctx() as store:
            store.append(key=provider.group, value=df, index=False, data_columns=True)


class PickleState:
    DEACTIVATED = 0  # do not use pickle
    CSV_ONLY = 1  # only dataframe of given csv files (no position_df, no id mapping, no merging, no features)
    MERGED = 2  # merged frame, position_df, id maping done (no features)
    FULL = 3  # all data needed to build DcDMap


class DcdBuilder:
    VIEW_MAP = 1
    FULL_MAP = 2

    tsc_id_idx_name = "ID"
    tsc_time_idx_name = "simtime"
    tsc_x_idx_name = "x"
    tsc_y_idx_name = "y"
    tsc_source_name = "source"

    VIEW_IDX = [
        tsc_id_idx_name,
        tsc_time_idx_name,
        tsc_x_idx_name,
        tsc_y_idx_name,
    ]

    FULL_IDX = [
        tsc_id_idx_name,
        tsc_time_idx_name,
        tsc_x_idx_name,
        tsc_y_idx_name,
        tsc_source_name,
    ]

    GLOBAL_IDX = [
        tsc_time_idx_name,
        tsc_x_idx_name,
        tsc_y_idx_name,
    ]

    GLOBAL_COLS = {
        "count": int,
        "node_id": np.str,
    }

    VIEW_COLS = {
        "count": int,
        "measured_t": float,
        "received_t": float,
        "source": np.str,
        "own_cell": int,
    }

    FULL_COLS = {
        "count": int,
        "measured_t": float,
        "received_t": float,
        "source": np.str,
        "own_cell": int,
        "selection": np.str,  # needed to filter out view form other data
    }

    def __init__(self):
        self._map_type = self.VIEW_MAP
        self._map_idx = self.VIEW_IDX
        self._glb_idx = self.GLOBAL_IDX
        self._glb_cols = self.GLOBAL_COLS
        self._map_cols = self.VIEW_COLS
        self._real_coords = True
        self._single_df_filters = []
        self._scenario_plotter = None
        self._features = [DcdUtil.delay_feature, DcdUtil.owner_dist_feature]
        self._global_csv_path = None
        self._node_csv_paths = []
        self._data_base_path = None
        self._root_pickle = None
        self._pickle_state = PickleState.DEACTIVATED

    def view(self):
        self._map_type = self.VIEW_MAP
        self._map_idx = self.VIEW_IDX
        return self

    def all(self):
        self._map_type = self.FULL_MAP
        self._map_idx = self.FULL_IDX
        self._map_cols = self.FULL_COLS
        return self

    def csv_paths(self, global_csv_path, node_csv_paths):
        self._global_csv_path = global_csv_path
        self._node_csv_paths = node_csv_paths
        return self

    def data_base_path(self, path):
        self._data_base_path = path
        return self

    def pickle_name(self, pickle_name):
        if os.path.isabs(pickle_name):
            self._root_pickle = pickle_name
        else:
            self._root_pickle = os.path.join(self._data_base_path, pickle_name)
        return self

    def pickle_as(self, state: PickleState):
        self._pickle_state = state
        return self

    def data_idx(self, *kw):
        self._map_idx = kw
        return self

    def glb_idx(self, *kw):
        self._glb_idx = kw
        return self

    def glb_cols(self, **kwargs):
        self._glb_cols = kwargs
        return self

    def map_cols(self, **kwargs):
        self._map_cols = kwargs
        return self

    def add_single_filter(self, *kwarg):
        self._single_df_filters.extend(*kwarg)
        return self

    def clear_single_filter(self):
        self._single_df_filters.clear()
        return self

    def clear_features(self):
        self._features.clear()
        return self

    def add_feature(self, *kw):
        self._features.extend(*kw)
        return self

    def use_real_coords(self, val=True):
        self._real_coords = val
        return self

    def plotter(self, plotter):
        if type(plotter) == str:
            self._scenario_plotter = VaderScenarioPlotHelper(plotter)
        else:
            self._scenario_plotter = plotter
        return self

    def build(self):
        if self._pickle_state == PickleState.DEACTIVATED or not os.path.exists(
            self._root_pickle
        ):
            # no pickle path provided build from csv data (may take some time!)
            return self.build_from_csv()

        # read pickle and extract data. Recreate DcdBuilder object used to create the data and check that the pickle is
        # usable for the current DcdBuilder configuration
        _pickle = self.read_pickle()
        _builder = self.from_json(_pickle["builder_json"])
        _state = _builder._pickle_state
        _data = _pickle["data"]
        if not self._compare_builder(_builder):
            raise RuntimeError("pickle not compatible with current builder")

        # Apply missing build steps based on pickle state (if any)
        print(f"found pickle state {_state} build object ...")
        if (
            _state == PickleState.CSV_ONLY
        ):  # no location table, features only csv, (delay gets calulated afterwards)
            # position_df, no id mapping, no merging, no features
            _data = self.do_merge(**_data)
            _data = self.do_feature(**_data)
        elif _state == PickleState.MERGED:
            _data = self.do_feature(**_data)
            _data = self.do_extract_view(**_data)
        elif _state == PickleState.FULL:
            pass  # do nothing
        else:
            raise RuntimeError("unknown PickleState")

        # Create Dcd object (finally...)
        return self._build(
            _data,
            self._map_type,
            plotter=self._scenario_plotter,
            data_base_dir=self._data_base_path,
        )

    def _compare_builder(self, other):
        # other only contains static content! class/method references were striped due to pickle
        return self._pickle_state == other._pickle_state

    def build_from_csv(self):
        data = self.dcd_data_from_csv(self._global_csv_path, self._node_csv_paths)
        return self._build(
            data,
            self._map_type,
            plotter=self._scenario_plotter,
            data_base_dir=self._data_base_path,
        )

    @classmethod
    def _build(cls, data, map_type, **kwargs):
        if map_type == cls.FULL_MAP:
            return DcdMap2DMulti(**data, **kwargs)
        else:
            return DcdMap2D(**data, **kwargs)

    def write_pickle(self, data: dict, **kwargs):
        """
        write given data to pickle file. Add DcdBuilder json and any additional data (kwargs)
        to the pickle. Write json representation of DcdBuilder object with striped references
        to ensure pickles will work with later version of the analysis software. Only pickle
        stable data (DataFrames, Series, Json strings, serializable content)
        """
        if self._root_pickle is None:
            raise RuntimeError("Path not set")
        p = {
            "builder_json": self.to_json(clear_references=True),
            "data": data,
        }
        p.update(kwargs)
        with open(self._root_pickle, "wb") as fd:
            pickle.dump(p, fd)

    def read_pickle(self):
        if self._root_pickle is None:
            raise RuntimeError("Path not set")
        with open(self._root_pickle, "rb") as fd:
            data = pickle.load(fd)
        return data

    def _pickle_if_needed(self, data, state):
        if (
            self._pickle_state != PickleState.DEACTIVATED
            and self._pickle_state == state
        ):
            self.write_pickle(data)

    def dcd_data_from_csv(self, global_data_path, node_data_paths):
        _ret = self.do_csv(global_data_path, node_data_paths)
        self._pickle_if_needed(_ret, PickleState.CSV_ONLY)
        _ret = self.do_merge(**_ret)
        self._pickle_if_needed(_ret, PickleState.MERGED)
        _ret = self.do_feature(**_ret)
        _ret = self.do_extract_view(**_ret)
        self._pickle_if_needed(_ret, PickleState.FULL)
        return _ret

    def do_csv(self, global_data_path, node_data_paths):
        df_global = DcdUtil.build_density_map(
            csv_path=global_data_path,
            index=self._glb_idx,
            column_types=self._glb_cols,
            real_coords=self._real_coords,
            df_filter=None,
        )

        # load map for each node *.csv -> list of DataFrames
        njobs = int(multiprocessing.cpu_count() * 0.60)
        with multiprocessing.Pool(processes=njobs) as pool:
            job_args = [
                {
                    "csv_path": p,
                    "index": self.VIEW_IDX[
                        1:
                    ],  # ID will be set later (use always VIEW_IDX here. IN full map source will be set later)
                    "column_types": self._map_cols,
                    "real_coords": self._real_coords,
                    "df_filter": self._single_df_filters,
                }
                for p in node_data_paths
            ]
            df_data = DcdUtil.run_pool(pool, DcdUtil.build_density_map, job_args)
        return {"global_data": df_global, "node_data": df_data}

    def do_merge(self, global_data, node_data):
        meta_data, global_df = global_data

        # create location df [x, y] -> nodeId (long string)
        print("create location df [x, y] -> nodeId")
        position_df, global_df = self.build_position_df(global_df)

        # merge node frames and set index
        print("merge dataframes")
        map_df = self.merge_frames(node_data, self._map_idx)

        _ret = {
            "metadata": meta_data,
            "global_df": global_df,
            "map_df": map_df,  # may be full map based on self._type!
            "position_df": position_df,
        }
        return _ret

    def do_extract_view(self, **data):
        if self._map_type == self.FULL_MAP:
            data.setdefault("map_all_df", data["map_df"])
            _map_df = data["map_df"]
            _map_df = (
                _map_df[_map_df["selection"].notnull()]
                .drop(columns=["selection"])
                .reset_index(self._map_idx[-1], drop=False)
            )
            if not _map_df.index.is_unique:
                raise ValueError("map_view_df must have unique index")
            data["map_df"] = _map_df

        return data

    def do_feature(self, **data):
        print("calculate features")
        for _feature in self._features:
            data["map_df"] = _feature(data["map_df"], **data)

        return data

    @staticmethod
    def build_position_df(glb_df):
        # global position map for all node_ids
        glb_loc_df = glb_df["node_id"].copy().reset_index()
        glb_loc_df = glb_loc_df.assign(
            node_id=glb_loc_df["node_id"].str.split(r",\s*")
        ).explode("node_id")
        # remove node_id column from global
        glb_df = glb_df.drop(labels=["node_id"], axis="columns")
        return glb_loc_df, glb_df

    @staticmethod
    def merge_frames(input_df, index):
        node_dfs = []
        for meta, _df in input_df:
            # add ID as column
            _df["ID"] = meta.node_id
            # remove index an collect in list for later pd.concat
            _df = _df.reset_index()
            node_dfs.append(_df)

        _df_ret = pd.concat(node_dfs, levels=index, axis=0)
        _df_ret = _df_ret.set_index(index)
        _df_ret = _df_ret.sort_index()
        return _df_ret

    def to_json(self, clear_references=True):
        _obj = dict(self.__dict__)
        if clear_references:
            replace_default = {
                "_single_df_filters": [],
                "_scenario_plotter": None,
                "_features": [],
            }
            for key, value in replace_default.items():
                _obj[key] = value
        return json.dumps(_obj, default=lambda o: str(o), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, json_str):
        _json = json.loads(json_str)
        _obj = cls()
        for key, val in _json.items():
            setattr(_obj, key, val)

        return _obj
