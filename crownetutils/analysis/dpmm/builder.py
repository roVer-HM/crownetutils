from __future__ import annotations

import glob
import json
import multiprocessing
import os
import pickle
import re
from functools import partial
from typing import Union

import numpy as np
import pandas as pd
from pandas import IndexSlice as Idx

import crownetutils.analysis.dpmm.csv_loader as DcdUtil
from crownetutils.analysis.dpmm.dpmm import DpmMap
from crownetutils.analysis.dpmm.dpmm_cfg import DpmmCfg, MapType
from crownetutils.analysis.dpmm.hdf.dpmm_count_provider import DpmmCount
from crownetutils.analysis.dpmm.hdf.dpmm_global_positon_provider import (
    DpmmGlobal,
    DpmmGlobalPosition,
    create_and_save_position_and_global,
)
from crownetutils.analysis.dpmm.hdf.dpmm_provider import DpmmKey, DpmmProvider
from crownetutils.analysis.dpmm.imputation import (
    ArbitraryValueImputation,
    MissingValueImputationStrategy,
)
from crownetutils.analysis.dpmm.metadata import DpmmMetaData
from crownetutils.analysis.hdf.provider import BaseHdfProvider, ProviderVersion
from crownetutils.omnetpp.scave import CrownetSql
from crownetutils.utils.dataframe import FrameConsumer
from crownetutils.utils.logging import logger, logging, timing
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


def parse_node_id(path: str, regex: re.Pattern) -> int:
    grps = [m.groupdict() for m in regex.finditer(path)]
    if not grps:
        raise ValueError(f"No node id found in: '{path}' using pattern: '{regex}'")
    node_id = int(grps.pop()["node"])
    return node_id


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
        cls,
        cfg: DpmmCfg,
        override_hdf: bool = False,
    ):
        if override_hdf and os.path.exists(cfg.hdf_path()):
            logger.info(
                f"found hdf file but override hdf is active. remove {cfg.hdf_path()}"
            )
            os.remove(cfg.hdf_path())
        return cls(cfg=cfg)

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

    def __init__(self, cfg: DpmmCfg):  # hdf_path, map_paths, global_path, epsg=""):
        super().__init__()
        self.cfg = cfg
        self.sql = CrownetSql.from_dpmm_cfg(cfg)
        # paths
        self.hdf_path = self.cfg.hdf_path()
        self.map_paths = glob.glob(os.path.join(cfg.base_dir, cfg.node_map_csv_glob))
        self.global_path = os.path.join(cfg.base_dir, cfg.global_map_csv_name)

        # providers
        # self.count_p = DpmmCount(self.cfg.hdf_path("count.h5"))
        self.count_p = DpmmCount(self.hdf_path)
        self.map_p = DpmmProvider(self.hdf_path)
        self.position_p = DpmmGlobalPosition(self.hdf_path)
        self.global_p = DpmmGlobal(self.hdf_path)
        # options:
        # filters used during csv processing
        self.single_df_filters = []
        self._only_selected_cells = True
        self._epsg = cfg.epsg_base
        self._imputation_function = ArbitraryValueImputation(0.0)
        self._map_type: MapType = cfg.map_type

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
                pass
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
        self.position_p, self.global_p, meta = create_and_save_position_and_global(
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
        id_extractor = partial(
            parse_node_id, regex=re.compile(self.cfg.node_map_csv_id_regex)
        )
        self.map_p.create_from_csv(
            self.map_paths,
            id_extractor=id_extractor,
            frame_consumer=[
                partial(
                    self.create_count_map,
                    csv_version=self.map_p.version,
                    imputation_f=self._imputation_function,
                )
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

        if self.map_p.version >= ProviderVersion.V0_4:
            rsd = self.sql.get_resource_sharing_domains(ids_only=True)
            with self.map_p.ctx() as store:
                store.append(
                    key=DpmmKey.RSD_ID,
                    value=rsd,
                    index=False,
                    format="table",
                    data_columns=True,
                )

        # 6) set attributes
        for p in [self.position_p, self.global_p, self.map_p, self.count_p]:
            p.set_attribute("cell_size", meta.cell_size)
            p.set_attribute("cell_count", meta.cell_count)
            p.set_attribute("cell_bound", meta.bound)
            p.set_attribute("sim_bbox", meta.sim_bbox)
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
        self.position_p, self.global_p = create_and_save_position_and_global(
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

    @timing
    def create_count_map(
        self,
        df: pd.DataFrame,
        csv_version: ProviderVersion = ProviderVersion.current(),
        imputation_f: MissingValueImputationStrategy = ArbitraryValueImputation(),
    ):
        """
        Creates dataframe of the form (index)[column]:
          (simtime, x, y, ID)[count, err, sqerr, owner_dist, missing_value]

          The frame contains only time steps for which the node is also present in the simulation. Missing values,
          for timestamped cells (simtime, x, y) the node does not have any information, are append using ground truth
          data as well as the imputation strategy provide as an argument to the method.

          The returned frame does not contain any NAN values in any column. If NAN values are present after the imputation
          a ValueError is raised.

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
        # extract node id and times from data frame
        id = _df.index.get_level_values("ID").unique()[0]

        # performance: use boolean index to filter time interval instead of frame.loc[xxx] (~18 times faster)
        #              the filter assumes that a node is always present between start and end time. This is
        #              always valid
        t_min = _df.index.get_level_values("simtime").min()
        t_max = _df.index.get_level_values("simtime").max()
        # select global data for time interval the current node is in the simulation
        _m = (self.global_df.index.get_level_values(0) >= t_min) & (
            self.global_df.index.get_level_values(0) <= t_max
        )
        # glb = self.global_df.loc[present_at_times].set_axis(["glb_count"], axis=1)
        glb = self.global_df[_m].set_axis(["glb_count"], axis=1)

        # get count and node position
        selected_columns = DpmmKey.count_map_creation_cols[csv_version]
        _df = _df.loc[:, selected_columns].droplevel(["source", "ID"])
        # merge with global and mark 'missing values' based on 'count' column
        # NAN values in 'glb_count' are not missing values. These are cases where
        # the node measured something but there was nothing in the ground truth.
        # If and how this kind of error is dealt with is determined by the imputation strategy
        _df = pd.concat([glb, _df], axis=1)
        # safe missing count values as discriminator
        _df["missing_value"] = _df["count"].isna()

        # See MissingValueImputationStrategy  configuration of builder for more information
        # imputation will sort by index [simtime, x, y]
        _df = imputation_f.with_csv_id(id=id).apply(_df)

        # no NAN values after this point.
        if _df.isna().any(axis=0).any():
            raise ValueError(
                f"Count map processing for node {id} contains NAN values after imputation processing: NAN found in columns: {_df.isna().any(axis=0).to_dict()}"
            )

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
        _df = _df.astype(
            {
                k: v
                for k, v in DpmmKey.types_csv_columns[csv_version].items()
                if k in _df.columns
            }
        )
        self.append_to_provider(self.count_p, _df)

    def append_global_count(self):
        _df = self.global_df.copy()
        _df["ID"] = 0  # global id set to 0
        _df["ID"].convert_dtypes(int)
        _df["err"] = 0.0
        _df["sqerr"] = 0.0
        _df["owner_dist"] = 0.0
        _df["missing_value"] = False
        if self.global_p.version >= ProviderVersion.V0_4:
            _df[DpmmKey.RSD_ID] = -1
            _df[DpmmKey.RSD_ID_OWNER] = -1
        _df = _df.set_index(["ID"], drop=True, append=True)
        self.append_to_provider(self.count_p, _df)

    @staticmethod
    @timing
    def append_to_provider(provider: BaseHdfProvider, df: pd.DataFrame):
        logger.debug(f"save frame with shape {df.shape} to group {provider.group}")
        with provider.ctx() as store:
            store.append(
                key=provider.group,
                value=df,
                format="table",
                index=False,
                data_columns=True,
                complevel=9,
                complib="blosc",
            )
