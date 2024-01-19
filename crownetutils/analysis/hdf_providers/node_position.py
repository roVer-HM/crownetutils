from __future__ import annotations

import os
from collections.abc import Iterator
from enum import Enum
from functools import partial

import numpy as np
import pandas as pd
from matplotlib.colors import Colormap

from crownetutils.analysis.common import Simulation
from crownetutils.analysis.hdf.provider import (
    BaseHdfProvider,
    HdfGroupDataFactory,
    HdfGroupFactory,
)
from crownetutils.omnetpp.scave import CrownetSql
from crownetutils.utils.colors import get_color_map
from crownetutils.utils.dataframe import (
    get_index_name_or_names,
    index_or_col,
    merge_on_interval,
)
from crownetutils.utils.logging import logger
from crownetutils.utils.misc import Project


class NodePositionHdf(BaseHdfProvider):
    @classmethod
    def get(cls, hdf_path: str, sql: CrownetSql, **kwargs) -> NodePositionHdf:
        obj = cls(hdf_path=hdf_path, group="trajectories", allow_lazy_loading=True)
        if len(kwargs) > 0:
            obj.add_group_factory(
                HdfGroupFactory(obj.group, partial(sql.node_position, **kwargs))
            )
        else:
            obj.add_group_factory(HdfGroupFactory(obj.group, sql.node_position))
        return obj

    @classmethod
    def from_sim(cls, sim: Simulation, **kwargs) -> NodePositionHdf:
        return cls.get(sim.path("position.h5"), sim.sql, **kwargs)


class EnbPositionHdf(BaseHdfProvider):
    @classmethod
    def get(cls, hdf_path: str, sql: CrownetSql, **kwargs) -> EnbPositionHdf:
        obj = cls(hdf_path=hdf_path, group="enb", allow_lazy_loading=True)
        if len(kwargs) > 0:
            obj.add_group_factory(
                HdfGroupFactory(obj.group, partial(sql.enb_position, **kwargs))
            )
        else:
            obj.add_group_factory(HdfGroupFactory(obj.group, sql.enb_position))
        return obj

    @classmethod
    def from_sim(cls, sim: Simulation, **kwargs) -> EnbPositionHdf:
        return cls.get(sim.path("position.h5"), sim.sql, **kwargs)


class CoordinateType(Enum):
    """Named coorinate pairs for different offsets and coordiante systems.
    The enum contains a tuple of column names representing 'x/lon' and 'y/lat'
    coordinates. The 'x'/'y' property provides the corespinging column string name.

    Use deref '*' to access the columns interator or the cols property for a list.
    """

    lonlat = ("lonlat", ("lon", "lat"), False)
    xy = ("xy", ("x", "y"), True)
    xy_no_geo_offset = ("_xy", ("_x", "_y"), True)
    xy_cell = (
        "cxy",
        ("_x_cell", "_y_cell"),
        True,
    )  # workaround due to density map cell origin offset and
    # gis offset. cxy origin aligns with cell (0, 0)

    @property
    def is_cartesian(self) -> bool:
        return self.value[2]

    @property
    def cols(self):
        return list(self.value[1])

    @property
    def x(self):
        return self.value[1][0]

    @property
    def y(self):
        return self.value[1][1]

    def __iter__(self):
        return iter(self.cols)

    @classmethod
    def from_str(cls, val: str):
        for i in cls:
            if i.value[0] == val or i.name == val:
                return i
        raise ValueError(f"{cls.__name__} has no matching value")


class NodePositionWithRsdHdf:
    """ """

    _ue_idx_cols = [
        ("hostId", "time", "host", "vecIdx"),
        "x",
        "y",
        "lon",
        "lat",
        "_x",
        "_y",
    ]
    _host_time_enb_interval = []

    group_ue_position = "ue_position"
    group_interval = "host_time_enb_interval"
    group_enb_position = "enb_position"

    def __init__(
        self,
        sim: Simulation,
        hdf_path: str,
        base_epsg: str = "EPSG:3857",
        static_nodes: str = "misc",
    ) -> None:
        self.sim: Simulation = sim
        self.base_epsg = base_epsg
        self.hdf_path = hdf_path
        self.static_nodes = static_nodes
        self._ue = None
        self._enb = None
        self._host_time_interval = None
        self.groups = [
            self.group_ue_position,
            self.group_enb_position,
            self.group_interval,
        ]

    @classmethod
    def get_or_create(
        cls,
        sim,
        hdf_path,
        base_epsg: str = "EPSG:3857",
        static_nods: str = "misc",
        override_existing: bool = True,
    ) -> NodePositionWithRsdHdf:
        obj: NodePositionWithRsdHdf = cls(sim, hdf_path, base_epsg, static_nods)
        same = True
        if os.path.exists(hdf_path):
            # check if file is consitent with paramters
            for g in obj.groups:
                if not obj.ue.contains_group(g):
                    same = False
                    break
                for now, in_file in [
                    (obj.static_nodes, "static_node"),
                    (obj.base_epsg, "base_epsg"),
                ]:
                    # use ue here but all groups are in same hdf file.
                    if now != obj.ue.get_attribute(
                        attr_key=in_file, group=g, default=""
                    ):
                        same = False
                        break
            if same:
                # hdf file exists and contains all data with same parameters
                logger.info(
                    f"found existing {cls.__name__} file with match parameter setup. No build required."
                )
                ret = obj
            else:
                if override_existing is False:
                    raise ValueError(
                        f"File {hdf_path} already exists but parameter settings mismatch and override is false"
                    )
                else:
                    logger.info(
                        "found existing hdf file with parameter mismatch. Remove file and rebuild."
                    )
                    os.remove(hdf_path)
                    obj._build_hdf()
                    ret = obj
        else:
            logger.info("no existing hdf file found. Build hdf...")
            obj._build_hdf()
            ret = obj

        return ret

    @classmethod
    def create_new(
        cls,
        sim,
        hdf_path,
        base_epsg: str = "EPSG:3857",
        static_nods: str = "misc",
        override_existing: bool = True,
    ):
        if os.path.exists(hdf_path):
            if override_existing is False:
                raise ValueError(
                    f"File {hdf_path} already exists but override is false"
                )
            else:
                os.remove(hdf_path)

        obj = cls(sim, hdf_path, base_epsg, static_nods)
        obj._build_hdf()
        return obj

    @property
    def ue(self) -> BaseHdfProvider:
        if self._ue is None:
            self._ue = BaseHdfProvider(
                self.hdf_path, group=self.group_ue_position, allow_lazy_loading=True
            )
            self._ue.add_group_factory(
                HdfGroupFactory(
                    group_name=self.group_ue_position,
                    factory=HdfGroupDataFactory.LAZY_LOAD_NOT_SUPPORTED,
                    meta=None,
                    post=self._start_end_to_interval,
                )
            )
        return self._ue

    @property
    def enb(self) -> BaseHdfProvider:
        if self._enb is None:
            self._enb = BaseHdfProvider(
                self.hdf_path, group=self.group_enb_position, allow_lazy_loading=False
            )
        return self._enb

    @property
    def host_time_interval(self):
        if self._host_time_interval is None:
            self._host_time_interval = BaseHdfProvider(
                self.hdf_path, group=self.group_interval, allow_lazy_loading=True
            )
            self._host_time_interval.add_group_factory(
                HdfGroupFactory(
                    group_name=self.group_interval,
                    factory=HdfGroupDataFactory.LAZY_LOAD_NOT_SUPPORTED,
                    meta=None,
                    post=self._start_end_to_interval,
                )
            )
        return self._host_time_interval

    @staticmethod
    def _start_end_to_interval(
        frame: pd.DataFrame, p: BaseHdfProvider, key=None
    ) -> pd.DataFrame:
        m = p.get_attribute("time_interval_data", group=key)
        if all(i in frame.columns for i in [m[1], m[2]]):
            idx = [
                pd.Interval(v[0], v[1], closed=m[3])
                for v in frame.loc[:, [m[1], m[2]]].values
            ]
            frame[m[0]] = pd.IntervalIndex(idx, closed=m[3])
        return frame

    def merge_rsd_id_on_host_time_interval(
        self,
        data: pd.DataFrame,
        host_id_col: str | int = "hostId",
        time_col="time",
        append_interval: bool = False,
    ) -> pd.DataFrame:
        old_index = get_index_name_or_names(data)
        if isinstance(host_id_col, int):
            interval = self.host_time_interval.select(where=f"index == {host_id_col}")
            try:
                if not data.index.name == time_col:
                    if isinstance(data.index, pd.RangeIndex):
                        data = data.set_index([time_col]).sort_index()
                    else:
                        data = data.reset_index().set_index([time_col]).sort_index()
            except Exception as e:
                raise ValueError(f"data must contain '{time_col}' index") from e
        else:
            try:
                if not data.index.names == [host_id_col, time_col]:
                    if isinstance(data.index, pd.RangeIndex):
                        data = data.set_index([host_id_col, time_col]).sort_index()
                    else:
                        data = (
                            data.reset_index()
                            .set_index([host_id_col, time_col])
                            .sort_index()
                        )
            except Exception as e:
                raise ValueError(
                    f"data must contain '{host_id_col}' and 'name' columns or index"
                ) from e

            u_host_ids = index_or_col(data, host_id_col).unique().to_list()
            if len(u_host_ids) < 30:
                interval = self.host_time_interval.select(
                    where=f"hostId in {list(u_host_ids)}"
                )
            else:
                interval = self.host_time_interval.frame()
            interval.index.name = host_id_col
        ret = merge_on_interval(
            data=data,
            df_interval=interval,
            index=time_col,
            interval_col="time_interval",
            interval_closed_at="left",
            merge=True,
            copy_data=False,
        )
        if not append_interval:
            ret = ret.drop(
                columns=["time_interval", "time_interval_start", "time_interval_end"]
            )
        if ret["servingEnb"].isna().any():
            raise ValueError("got Nan")
        ret["servingEnb"] = ret["servingEnb"].astype(int)
        if old_index is not None:
            ret = ret.set_index(old_index)
        return ret

    def _build_hdf(self):
        ue, enb = self._query_position()

        max_time_dict = ue.groupby("hostId")["time"].max().to_dict()
        max_time = ue["time"].max()
        static_hosts = ue[["hostId", "host"]].drop_duplicates().copy()
        static_hosts["host"] = static_hosts["host"].str.startswith(self.static_nodes)
        static_hosts = {i[0]: i[1] for i in static_hosts.values}

        logger.info("query serving intervals form sql data base")
        serving_enbs: pd.DataFrame = self._query_serving_intervals(
            max_time_dict=max_time_dict, static_hosts=static_hosts, max_time=max_time
        )

        logger.info("query ue data form sql data base")
        ue = ue.set_index(["hostId", "time"]).sort_index()
        ue = merge_on_interval(
            ue, serving_enbs, index="time", interval_closed_at="left"
        ).drop(columns=["interval"])
        ue = ue.rename(
            columns={"start": "time_interval_start", "end": "time_interval_end"}
        )

        logger.info("query enb data form sql data base")
        self.enb.write_frame(
            frame=enb,
            group=self.group_enb_position,
            index=True,
            index_data_columns=True,
        )

        self.ue.write_frame(
            frame=ue, group=self.group_ue_position, index=True, index_data_columns=True
        )
        self.ue.set_attribute(
            attr_key="time_interval_data",
            value=("time_interval", "time_interval_start", "time_interval_end", "left"),
            group=self.group_ue_position,
        )

        serving_enbs = (
            serving_enbs.reset_index()
            .drop(columns=["interval"])
            .rename(
                columns={"start": "time_interval_start", "end": "time_interval_end"}
            )
        )
        serving_enbs = serving_enbs[
            ["hostId", "time_interval_start", "time_interval_end", "servingEnb"]
        ]
        serving_enbs = serving_enbs.set_index(["hostId"])

        self.host_time_interval.write_frame(
            frame=serving_enbs,
            group=self.group_interval,
            index=True,
            index_data_columns=True,
        )
        self.host_time_interval.set_attribute(
            attr_key="time_interval_data",
            value=("time_interval", "time_interval_start", "time_interval_end", "left"),
            group=self.group_interval,
        )

        for g in self.groups:
            # use ue here but all groups are in same hdf file.
            self.ue.set_attribute(
                group=g, attr_key="static_node", value=self.static_nodes
            )
            self.ue.set_attribute(group=g, attr_key="base_epsg", value=self.base_epsg)

        self.ue.repack_hdf(keep_old_file=False)

    def enb_colors(self) -> dict:
        """Get color dictoinary with key is the enb vector index"""
        rsd_ids = self.enb.frame()["rsd_id"]
        cmap, normalizer, colors = get_color_map(
            N=rsd_ids.shape[0], append_not_connected=True, cmap="tab20"
        )
        rsd_id_color_map = {i: c for i, c in enumerate(colors)}
        return rsd_id_color_map

    def get_ue_traces(
        self,
        coord: CoordinateType = CoordinateType.xy,
        cols=("hostId", "time", "servingEnb"),
        cmap: str | Colormap = "tab20",
        ue_where_clause: str | None = None,
    ):
        """Create line segements for rsd color coded line segement of ue traces.

        returns: trace, line segement array, line segement color, rsd mapped to color
        """

        segment = [*coord.cols, *[f"{c}_end" for c in coord.cols]]
        ue = self.ue.select(
            columns=[*cols, *coord.cols], where=ue_where_clause
        ).sort_values(["hostId", "time"])
        ue2 = ue[["hostId", *coord.cols]].shift(-1)
        cols = [*ue.columns, *ue2.add_suffix("_end").columns]
        _id_mask = (ue["hostId"] == ue2["hostId"]).values
        traces = pd.concat([ue[_id_mask], ue2[_id_mask]], axis=1, ignore_index=True)
        traces = traces.set_axis(cols, axis=1)
        if traces.shape[0] != _id_mask[_id_mask].shape[0]:
            raise ValueError("hie")

        rsd_ids = self.enb.frame()["rsd_id"]
        cmap, normalizer, colors = get_color_map(
            N=rsd_ids.shape[0], append_not_connected=True, cmap="tab20"
        )

        rsd_id_color_map = {i: c for i, c in enumerate(colors)}
        trace_colors = colors[traces["servingEnb"].astype(int).values]
        return (
            traces,
            traces[segment].values.reshape((-1, 2, 2)),
            trace_colors,
            rsd_id_color_map,
            cmap,
            normalizer,
        )

    def _query_serving_intervals(
        self, max_time_dict: dict, static_hosts: dict, max_time: float
    ):
        def _apply(s):
            # The end time of the last interval for each node is not part of the result data.
            # This happens because nodes can leave the simulation, but the 'leave' event will not
            # trigger a position change and thus no data is recoreded. Similarly for static nodes
            # or nodes that stay in the simulation longer than the simulation time will not have a
            # end time for the last interval.
            # Here we search for the max time for each node, which is either the largest time each node
            # provides in the result data, or in case of static nodes, the overall largest time.
            # As the position is only updated every second, the end time of the last interval is increased
            # by 1.0 second to esnure that any trailing event that happens betwen the last postion update
            # and the the removal event will fall into this event.
            if np.isnan(s["end"]):
                i = int(s["hostId"])
                if i in static_hosts:
                    s["end"] = max_time + 1.0
                elif i in max_time_dict:
                    t = max_time if s["start"] > max_time_dict[i] else max_time_dict[i]
                    s["end"] = t + 1.0  # ensure last measurement is part of interval
            return s

        end_time_provider = _apply
        serving = self.sim.sql.vector_ids_to_host(
            module_name=self.sim.sql.m_phy(),
            vector_name="servingCell:vector",
            pull_data=True,
            value_name="servingEnb",
            name_columns=["hostId"],
            drop=["vectorId"],
        )
        serving = serving.loc[:, ["hostId", "time", "servingEnb"]].sort_values(
            ["hostId", "time"]
        )

        # shift time column to get end time step.
        _end = serving.groupby("hostId").shift(-1)
        serving["end"] = _end["time"]
        serving = serving.rename(columns={"time": "start"})

        # set removal time of node as last time step based on end_time_provider
        serving = serving.apply(end_time_provider, axis=1).dropna()
        serving["delta"] = serving["end"] - serving["start"]

        serving["interval"] = [
            pd.Interval(*row, closed="left")
            for _, row in serving[["start", "end"]].iterrows()
        ]
        serving = serving.loc[:, ["hostId", "interval", "start", "end", "servingEnb"]]
        serving = serving.set_index(["hostId", "interval"]).sort_index()
        return serving

    def _query_position(self):
        bound = self.sim.sql.sim_bound()
        ue = self.sim.sql.node_position(
            bottom_left_origin=True,
            epsg_code_base=self.base_epsg,
            epsg_code_to=Project.WSG84_lat_lon,
        )
        lonlat = CoordinateType.lonlat
        no_geo = CoordinateType.xy_no_geo_offset
        cell_aligned = CoordinateType.xy_cell

        ue[lonlat.x] = ue.geometry.x
        ue[lonlat.y] = ue.geometry.y
        ue[no_geo.x] = ue["x"] + bound.offset[0]
        ue[no_geo.y] = ue["y"] + bound.offset[1]
        ue[cell_aligned.x] = ue[no_geo.x] + bound.sim_offset[0]
        ue[cell_aligned.y] = ue[no_geo.y] + bound.sim_offset[1]
        ue = pd.DataFrame(ue.drop(columns="geometry"))
        ue = ue.sort_index()

        enb = self.sim.sql.enb_position(
            bottom_left_origin=True,
            epsg_code_base=self.base_epsg,
            epsg_code_to=Project.WSG84_lat_lon,
        )
        enb[lonlat.x] = enb.geometry.x
        enb[lonlat.y] = enb.geometry.y
        enb[no_geo.x] = enb["x"] + bound.offset[0]
        enb[no_geo.y] = enb["y"] + bound.offset[1]
        enb[cell_aligned.x] = enb[no_geo.x] + bound.sim_offset[0]
        enb[cell_aligned.y] = enb[no_geo.y] + bound.sim_offset[1]

        enb = pd.DataFrame(enb.drop(columns="geometry"))
        enb["hostId"] = enb["hostId"].astype(int)
        enb["enb_id"] = enb["hostId"] + 1
        enb["rsd_id"] = enb["enb_id"]
        enb = enb[
            [
                "host",
                "hostId",
                "enb_id",
                "rsd_id",
                "x",
                "y",
                *lonlat,
                *no_geo,
                *cell_aligned,
            ]
        ].copy()
        enb = enb.sort_index()
        return ue, enb
