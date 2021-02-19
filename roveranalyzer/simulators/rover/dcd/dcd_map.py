import multiprocessing
from functools import wraps
from itertools import combinations
from typing import List, Union

import matplotlib.pyplot as plt
import matplotlib.table as tbl
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import IndexSlice as I

from roveranalyzer.simulators.rover.dcd.util import (
    DcdMetaData,
    build_density_map,
    run_pool,
)
from roveranalyzer.simulators.vadere.plots.plots import t_cmap
from roveranalyzer.simulators.vadere.plots.scenario import VaderScenarioPlotHelper
from roveranalyzer.utils.misc import intersect
from roveranalyzer.utils.plot import check_ax, update_dict


def plot_decorator(method):
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        if self.plot_wrapper is not None:
            return self.plot_wrapper(method, self, *method_args, **method_kwargs)
        else:
            return method(self, *method_args, **method_kwargs)

    return _impl


class DcdMap:
    cell_features = [
        "count",
        "measured_t",
        "received_t",
        "delay",
        "m_aoi",
        "r_aoi",
        "source",
        "own_cell",
    ]
    tsc_global_id = 0

    def __init__(
        self, m_glb: DcdMetaData, id_map, glb_loc_df: pd.DataFrame, plotter=None
    ):
        plt.Axes
        self.meta = m_glb
        self.node_to_id = id_map
        self.glb_loc_df = glb_loc_df
        self.id_to_node = {v: k for k, v in id_map.items()}
        self.scenario_plotter = plotter
        self.plot_wrapper = None
        self.font_dict = {
            "title": {"fontsize": 26},
            "xlabel": {"fontsize": 20},
            "ylabel": {"fontsize": 20},
            "legend": {"size": 20},
            "tick_size": 16,
        }

    def get_location(self, simtime, node_id, cell_id=False):
        try:
            ret = self.glb_loc_df.loc[simtime, node_id]
            if ret.shape == (2,):
                ret = ret.to_numpy()
                if cell_id:
                    ret = np.floor(ret / self.meta.cell_size)
            else:
                raise TypeError()
        except (TypeError, KeyError) as e:
            ret = np.array([-1, -1])
        return ret

    def set_scenario_plotter(self, plotter):
        self.scenario_plotter = plotter

    @staticmethod
    def extract_location_df(glb_df):
        # global position map for all node_ids
        glb_loc_df = glb_df["node_id"].copy().reset_index()
        glb_loc_df = glb_loc_df.assign(
            node_id=glb_loc_df["node_id"].str.split(r",\s*")
        ).explode("node_id")
        # remove node_id column from global
        glb_df = glb_df.drop(labels=["node_id"], axis="columns")
        return glb_loc_df, glb_df

    @staticmethod
    def create_id_map(glb_loc_df, node_data):
        ids = [n[0].node_id for n in node_data]
        ids.sort()
        ids = list(enumerate(ids, 1))
        ids.insert(0, (0, "-1"))
        node_to_id = {n[1]: n[0] for n in ids}

        glb_loc_df["node_id"] = glb_loc_df["node_id"].apply(lambda x: node_to_id[x])
        glb_loc_df = glb_loc_df.set_index(["simtime", "node_id"])
        return glb_loc_df, node_to_id

    @staticmethod
    def merge_data_frames(input_df, node_to_id, index):
        node_dfs = []
        for meta, _df in input_df:
            if meta.node_id not in node_to_id:
                raise ValueError(f"{meta.node_id} not found in id_map")
            # add ID as column
            _df["ID"] = node_to_id[meta.node_id]
            # replace source string id with int based ID
            _df["source"] = _df["source"].map(node_to_id)
            # remove index an collect in list for later pd.concat
            _df = _df.reset_index()
            node_dfs.append(_df)

        _df_ret = pd.concat(node_dfs, levels=index, axis=0)
        _df_ret = _df_ret.set_index(index)
        _df_ret = _df_ret.sort_index()
        return _df_ret

    @staticmethod
    def calculate_features(_df_ret):
        index_names = _df_ret.index.names
        # calculate features (delay, AoI_NR (measured-to-now), AoI_MNr (received-to-now)
        now = _df_ret.index.get_level_values("simtime")
        _df_ret["delay"] = _df_ret["received_t"] - _df_ret["measured_t"]
        # AoI_NM == measurement_age (time since last measurement)
        _df_ret["measurement_age"] = now - _df_ret["measured_t"]
        # AoI_NR == update_age (time since last update)
        _df_ret["update_age"] = now - _df_ret["received_t"]

        # Distance to owner location.
        # get owner positions for each time step {ID/simtime}[x_owner,y_owner]
        owner_locations = (
            _df_ret.loc[_df_ret["own_cell"] == 1, []]
            .index.to_frame(index=False)
            .drop(columns=["source"])
            .drop_duplicates()
            .set_index(["ID", "simtime"], drop=True, verify_integrity=True)
            .rename(columns={"x": "x_owner", "y": "y_owner"})
        )
        # create dummy data for global view (index = 0) and set 'owner' location of ground truth at origin.
        # create index {ID/simtime} for global id (:=0) and all time steps found
        _index = pd.MultiIndex.from_product(
            [[0], _df_ret.loc[0].index.get_level_values("simtime").unique()],
            names=["ID", "simtime"],
        )
        # create df with dummy owner location for ground truth
        glb_location = (
            pd.DataFrame(columns=["x_owner", "y_owner"], index=_index)
            .fillna(0.0)
            .astype(float)
        )
        # append dummy location to owner_locations
        owner_locations = glb_location.append(owner_locations, ignore_index=False)

        # merge ower position
        _df_ret = pd.merge(
            _df_ret.reset_index(),
            owner_locations,
            on=["ID", "simtime"],
        ).reset_index(drop=True)
        _df_ret["owner_dist"] = np.sqrt(
            (_df_ret["x"] - _df_ret["x_owner"]) ** 2
            + (_df_ret["y"] - _df_ret["y_owner"]) ** 2
        )
        _df_ret = _df_ret.set_index(index_names, drop=True, verify_integrity=True)

        return _df_ret


class DcdMap2D(DcdMap):
    """
    decentralized crowed map
    """

    tsc_id_idx_name = "ID"
    tsc_time_idx_name = "simtime"
    tsc_x_idx_name = "x"
    tsc_y_idx_name = "y"

    # single map in data frame
    view_index = [
        tsc_id_idx_name,
        tsc_time_idx_name,
        tsc_x_idx_name,
        tsc_y_idx_name,
    ]

    @classmethod
    def from_paths(
        cls,
        global_data: str,
        node_data: List,
        real_coords=True,
        scenario_plotter: Union[str, VaderScenarioPlotHelper] = None,
    ):
        _df_global = build_density_map(
            csv_path=global_data,
            index=cls.view_index[1:],  # ID will be set later
            column_types={
                "count": int,
                "measured_t": float,
                "received_t": float,
                "source": np.str,
                "own_cell": int,
                "node_id": np.str,
            },
            real_coords=real_coords,
            add_missing_cells=False,
        )
        _df_data = [
            build_density_map(
                csv_path=p,
                index=cls.view_index[1:],  # ID will be set later
                column_types={
                    "count": int,
                    "measured_t": float,
                    "received_t": float,
                    "source": np.str,
                    "own_cell": int,
                },
                real_coords=real_coords,
                add_missing_cells=False,
            )
            for p in node_data
        ]
        _obj = cls.from_frames(_df_global, _df_data)
        if scenario_plotter is not None:
            if type(scenario_plotter) == str:
                _obj.set_scenario_plotter(VaderScenarioPlotHelper(scenario_plotter))
            else:
                _obj.set_scenario_plotter(scenario_plotter)
        return _obj

    @classmethod
    def from_frames(cls, global_data, node_data):
        glb_m, glb_df = global_data

        # create location df [x, y] -> nodeId (long string)
        glb_loc_df, glb_df = cls.extract_location_df(glb_df)

        # ID mapping
        glb_loc_df, node_to_id = cls.create_id_map(glb_loc_df, node_data)

        # merge node frames and set index
        input_dfs = [(glb_m, glb_df), *node_data]
        _df_ret = cls.merge_data_frames(input_dfs, node_to_id, cls.view_index)
        _df_ret = cls.calculate_features(_df_ret)

        return cls(glb_m, node_to_id, _df_ret, glb_loc_df)

    def __init__(
        self,
        glb_meta: DcdMetaData,
        node_id_map: dict,
        map_view_df: pd.DataFrame,
        location_df: pd.DataFrame,
        plotter=None,
    ):
        super().__init__(glb_meta, node_id_map, location_df, plotter)
        self._map = map_view_df

    def iter_nodes_d2d(self, first=0, last=-1):
        _i = pd.IndexSlice

        data = self._map.loc[_i[first:], :].groupby(level=self.tsc_id_idx_name)
        for i, ts_2d in data:
            yield i, ts_2d

    @property
    def glb_map(self):
        return self.map.loc[
            self.tsc_global_id,
        ]

    @property
    def map(self):
        return self._map

    def unique_level_values(self, level_name, df_slice=None):
        if df_slice is None:
            df_slice = ([self.tsc_global_id], [])

        idx = self.map.loc[df_slice].index.get_level_values(level_name).unique()
        return idx

    def age_mask(self, age_column, threshold):
        _mask = self._map[age_column] <= threshold
        return _mask

    def with_age(self, age_column, threshold):
        df2d = self._map[self.age_mask(age_column, threshold)].copy()
        return DcdMap2D(self.meta, self.node_to_id, df2d, self.scenario_plotter)

    def count_diff(self, diff_metric="diff"):
        metric = {
            "diff": lambda loc, glb: glb - loc,
            "abs_diff": lambda loc, glb: (glb - loc).abs(),
            "sqr_diff": lambda loc, glb: (glb - loc) ** 2,
        }
        _df = pd.DataFrame(self._map.groupby(["ID", "simtime"]).sum()["count"])
        _df = _df.combine(_df.loc[0], metric[diff_metric])
        _df = _df.rename(columns={"count": "count_diff"})
        return _df

    def create_2d_map(self, time_step, node_id):
        """
        create 2d matrix of density map for one instance in
        time and for one node. The returned data frame as a shape of
         (N, M) where N,M is the number of cells in X respectively Y axis
        """
        # print(time_step, node_id)
        _i = pd.IndexSlice
        data = self._map.loc[_i[node_id, time_step], ["count"]]
        full_index = self.meta.grid_index_2d(real_coords=True)
        df = pd.DataFrame(np.zeros((len(full_index), 1)), columns=["count"])
        df = df.set_index(full_index)
        df.update(data)
        df = df.unstack().T
        return df

    def update_color_mesh(self, qmesh: QuadMesh, time_step, node_id):
        df = self.create_2d_map(time_step, node_id)
        data = np.array(df)
        qmesh.set_array(data.ravel())
        return qmesh

    @staticmethod
    def clear_color_mesh(qmesh: QuadMesh, default_val=0):
        qmesh.set_array(qmesh.get_array() * default_val)

    def info_dict(self, x, y, time_step, node_id):
        _i = pd.IndexSlice
        _data_dict = {c: "---" for c in self.map.columns}
        _data_dict["count"] = 0
        _data_dict["source"] = -1
        _data_dict.setdefault("_node_id", -1)
        _data_dict.setdefault("_omnet_node_id", -1)
        try:
            _data = self.map.loc[_i[node_id, time_step, x, y], :]
            _data_dict = _data.to_dict()
            _data_dict["_node_id"] = int(node_id)
            _data_dict["_omnet_node_id"] = self.id_to_node[node_id]
        except KeyError as e:
            pass
        finally:
            _data_dict["count"] = (
                int(_data_dict["count"]) if _data_dict["count"] != np.nan else "n/a"
            )
            try:
                _data_dict["source"] = int(_data_dict["source"])
            except ValueError:
                _data_dict["source"] = "n/a"
            for k, v in _data_dict.items():
                if type(v) == float:
                    _data_dict[k] = f"{v:.6f}"
            _data_dict.setdefault("_celll_coord", f"[{x}, {y}]")
        return _data_dict

    def own_cell(self):
        own_cell_mask = self._map["own_cell"] == 1
        places = (
            self._map[own_cell_mask]  # only own cells (cells where one node is located)
            .index.to_frame()  # make the index to the dataframe
            .reset_index(["x", "y"], drop=True)  # remove not needed index
            .drop(
                columns=["ID", "simtime"]
            )  # and remove columns created by to_frame we do not need
        )
        return places

    def create_label_positions(self, df, n=5):
        directions = 7
        teta = 2 * np.pi / directions
        r = 18.5
        rot = [
            r * np.array([np.cos(i), np.sin(i)])
            for i in np.arange(0, 2 * np.pi, step=teta, dtype=float)
        ]

        places = df.copy()
        places["x_center"] = places["x"] + 0.5 * self.meta.cell_size
        places["y_center"] = places["y"] + 0.5 * self.meta.cell_size
        places["x_text"] = places["x_center"]
        places["y_text"] = places["y_center"]
        for idx, row in places.iterrows():
            row["x_text"] = row["x_center"] + rot[idx[0] % directions][0]
            row["y_text"] = row["y_center"] + rot[idx[0] % directions][1]

        pairs = list(combinations(places.index.to_list(), 2))
        intersection_found = False
        for i in range(n):
            intersection_found = False
            for n1, n2 in pairs:
                # if overlapping change check overlapping
                l1 = (
                    places.loc[n1, ["x_center", "y_center", "x_text", "y_text"]]
                    .to_numpy()
                    .reshape(-1, 2)
                )
                l2 = (
                    places.loc[n2, ["x_center", "y_center", "x_text", "y_text"]]
                    .to_numpy()
                    .reshape(-1, 2)
                )
                if intersect(l1, l2):
                    _dir = int(np.floor(np.random.random() * directions))
                    places.loc[n2, "x_text"] = places.loc[n1, "x_center"] + rot[_dir][0]
                    places.loc[n2, "y_text"] = places.loc[n1, "y_center"] + rot[_dir][1]
                    print(f"intersection found {n1}<->{n2}")
                    intersection_found = True

            if not intersection_found:
                break

        if intersection_found:
            print(f"still overlaps found in annotation arrows after {n} rounds")
        return places

    def describe_raw(self, global_only=False):
        _i = pd.IndexSlice
        if global_only:
            data = self.glb_map  # only global data
            data_str = "Global"
        else:
            data = self._map.loc[_i[1:], :]  # all *but* global
            data_str = "Local"

        desc = data.describe().T
        print("=" * 79)
        print(f"Counts with values > 0 ({data_str}):")
        print(desc.loc[["count"], ["mean", "std", "min", "max"]])
        print("-" * 79)
        print(f"Delay for each cell ({data_str}):")
        print(desc.loc[["delay"], ["mean", "std", "min", "max"]])
        print("-" * 79)
        print(f"Time since measurement was taken ({data_str}):")
        print(desc.loc[["measurement_age"], ["mean", "std", "min", "max"]])
        print("-" * 79)
        print(f"Time since last update ({data_str}):")
        print(desc.loc[["update_age"], ["mean", "std", "min", "max"]])
        print("=" * 79)

    def plot_summary(self, simtime, id, title="", **kwargs):
        kwargs.setdefault("figsize", (16, 9))
        f, ax = plt.subplots(2, 2, **kwargs)
        ax = ax.flatten()
        f.suptitle(f"Node {id}_{self.id_to_node[id]} for time {simtime} {title}")
        self.plot_density_map(simtime, id, ax=ax[0])
        self.plot_location_map(simtime, ax=ax[1])
        self.plot_count(ax=ax[2])
        ax[3].clear()
        return f, ax

    @plot_decorator
    def plot_location_map(self, time_step, *, ax=None, add_legend=True):
        places = self.own_cell()
        _i = pd.IndexSlice
        places = places.loc[_i[:, time_step], :]  # select only the needed timestep

        f, ax = check_ax(ax)

        ax.set_title(f"Node Placement at time {time_step}s")
        ax.set_aspect("equal")
        ax.set_xlim([0, self.meta.x_dim])
        ax.set_ylim([0, self.meta.y_dim])
        if self.scenario_plotter is not None:
            self.scenario_plotter.add_obstacles(ax)

        for id, df in places.groupby(level=self.tsc_id_idx_name):
            # move coordinate for node :id: to center of cell.
            df = df + 0.5 * self.meta.cell_size
            ax.scatter(df["x"], df["y"], label=f"{id}_{self.id_to_node[id]}")

        if add_legend:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        return f, ax

    @plot_decorator
    def plot_location_map_annotated(self, time_step, *, ax=None, annotate=None):
        places = self.own_cell()
        _i = pd.IndexSlice
        places = places.loc[_i[:, time_step], :]  # select only the needed timestep
        f, ax = self.plot_location_map(time_step, ax, add_legend=False)

        # places = places.droplevel("simtime")
        places = self.create_label_positions(places)
        for id, df in places.groupby(level=self.tsc_id_idx_name):
            # move coordinate for node :id: to center of cell.
            ax.annotate(
                text=id,
                xy=df.loc[(id, time_step), ["x_center", "y_center"]].to_numpy(),
                xytext=df.loc[(id, time_step), ["x_text", "y_text"]].to_numpy(),
                xycoords="data",
                arrowprops=dict(arrowstyle="->"),
            )

        return f, ax

    def update_delay_over_distance(
        self,
        time_step,
        node_id,
        delay_kind,
        remove_null=True,
        data: Line2D = None,
        bins_width=2.5,
    ):
        df = self.map
        # remove cells with count 0
        if remove_null:
            df = self.map.loc[self.map["count"] != 0]

        # average over distance (ignore node_id)
        if bins_width > 0:
            df = df.loc[I[1:, time_step], ["owner_dist", delay_kind]]
            bins = int(np.floor(df["owner_dist"].max() / bins_width))
            df = df.groupby(pd.cut(df["owner_dist"], bins)).mean().dropna()
            df.index = df.index.rename("dist_bin")
        else:
            df = df.loc[I[node_id, time_step], ["owner_dist", delay_kind]]

        df = df.sort_values(axis=0, by=["owner_dist"])
        if data is not None:
            data.set_ydata(df[delay_kind].to_numpy())
            data.set_xdata(df["owner_dist"].to_numpy())
        return df

    @plot_decorator
    def plot_delay_over_distance(
        self,
        time_step,
        node_id,
        delay_kind="measurement_age",
        remove_null=True,
        label=None,
        *,
        ax=None,
        fig_dict: dict = None,
        ax_prop: dict = None,
        **kwargs,
    ):
        """
        Plot delay_kind* over the distance between measurements location (cell) and
        the position of the map owner.

        Default data view: per node / per time / all cells

        *)delay_kind: change definition of delay using the delay_kind parameter.
          one of: ["delay", "measurement_age", "update_age"]
        """

        f, ax = check_ax(ax, **fig_dict if fig_dict is not None else {})
        df = self.update_delay_over_distance(
            time_step, node_id, delay_kind, remove_null=remove_null, data=None
        )
        if label is None:
            label = delay_kind
        ax.plot("owner_dist", delay_kind, data=df, label=label)

        if ax_prop is None:
            ax_prop = {}

        ax_prop.setdefault(
            "title",
            f"Delay({delay_kind}) over Distance",
        )
        ax_prop.setdefault("xlabel", "Cell distance (euklid) to owners location [m]")
        ax_prop.setdefault("ylabel", "Delay in [s]")
        for k, v in ax_prop.items():
            getattr(ax, f"set_{k}", None)(v, **self.font_dict[k])

        return f, ax

    @plot_decorator
    def plot_density_map(
        self,
        time_step,
        node_id,
        *,
        ax=None,
        cmap_dic: dict = None,
        pcolormesh_dic: dict = None,
        fig_dict: dict = None,
        ax_prop: dict = None,
        **kwargs,
    ):
        """
        Birds eyes view of density in a 2D color mesh with X/Y spanning the
        area under observation. Z axis (density) is shown with given color grading.

        Default data view: per node / per time / all cells
        """
        df = self.create_2d_map(time_step, node_id)
        f, ax = check_ax(ax, **fig_dict if fig_dict is not None else {})

        cell = self.get_location(time_step, node_id, cell_id=False)
        if "title" in kwargs:
            ax.set_title(kwargs["title"], **self.font_dict["title"])
        else:
            ax.set_title(
                f"Density map for node {node_id}_{self.id_to_node[node_id]} at time "
                f"{time_step} cell [{cell[0]}, {cell[1]}]",
                **self.font_dict["title"],
            )
        ax.set_aspect("equal")
        ax.tick_params(axis="x", labelsize=self.font_dict["tick_size"])
        ax.tick_params(axis="y", labelsize=self.font_dict["tick_size"])

        if self.scenario_plotter is not None:
            self.scenario_plotter.add_obstacles(ax)

        _d = update_dict(cmap_dic, cmap_name="Reds", replace_index=(0, 1, 0.0))
        cmap = t_cmap(**_d)
        _d = update_dict(pcolormesh_dic, cmap=cmap, shading="flat")

        pcm = ax.pcolormesh(self.meta.X_flat, self.meta.Y_flat, df, **_d)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.set_label("colorbar")
        cax.tick_params(axis="y", labelsize=self.font_dict["tick_size"])
        f.colorbar(pcm, cax=cax)

        ax.update(ax_prop if ax_prop is not None else {})

        return f, ax

    @plot_decorator
    def plot_count(self, *, ax=None, **kwargs):
        f, ax = check_ax(ax, **kwargs)
        ax.set_title("Total node count over time", **self.font_dict["title"])
        ax.set_xlabel("time [s]", **self.font_dict["xlabel"])
        ax.set_ylabel("total node count", **self.font_dict["ylabel"])

        for id, df in self.iter_nodes_d2d(first=1):
            df_time = df.groupby(level=self.tsc_time_idx_name).sum()
            ax.plot(
                df_time.index, df_time["count"], label=f"{id}_{self.id_to_node[id]}"
            )

        g = self.glb_map.groupby(level=self.tsc_time_idx_name).sum()
        ax.plot(
            g.index.get_level_values(self.tsc_time_idx_name),
            g["count"],
            label=f"0_{self.id_to_node[0]}",
        )
        ax.legend()
        return f, ax

    @plot_decorator
    def plot_count_diff(self, *, ax=None, **kwargs):
        f, ax = check_ax(ax, **kwargs)
        ax.set_title("Node Count over Time", **self.font_dict["title"])
        ax.set_xlabel("Time [s]", **self.font_dict["xlabel"])
        ax.set_ylabel("Pedestrian Count", **self.font_dict["ylabel"])
        _i = pd.IndexSlice
        nodes = (
            self._map.loc[_i[1:], _i["count"]]
            .groupby(level=[self.tsc_id_idx_name, self.tsc_time_idx_name])
            .sum()
            .groupby(level="simtime")
            .mean()
        )
        nodes_std = (
            self._map.loc[_i[1:], _i["count"]]
            .groupby(level=[self.tsc_id_idx_name, self.tsc_time_idx_name])
            .sum()
            .groupby(level="simtime")
            .std()
        )
        glb = self.glb_map.groupby(level=self.tsc_time_idx_name).sum()["count"]
        ax.plot(nodes.index, nodes, label="Mean count")
        ax.fill_between(
            nodes.index,
            nodes + nodes_std,
            nodes - nodes_std,
            alpha=0.35,
            interpolate=True,
            label="Count +/- 1 std",
        )
        ax.plot(glb.index, glb, label="Actual count")
        f.legend()
        return f, ax


def filter_selected_cells(df: pd.DataFrame):
    return df.loc[df["selection"].notna()].copy(deep=True)


class DcdMap2DMulti(DcdMap2D):
    tsc_source_idx_name = "source"

    # single map in data frame
    @classmethod
    def from_paths(
        cls,
        global_data: str,
        node_data: List,
        real_coords=True,
        load_all=True,
        scenario_plotter: Union[str, VaderScenarioPlotHelper] = None,
    ):
        """
         Parameters
        ----------
        global_data: Path to global map csv, First line must container meta data
                     information starting with a '#'
        node_data:   List of path for local density map
        real_coords: Translate gird ids to coordinates by multiplying with cell size found in meta data. default=True
        load_all:    Load all cell counts even the ones not chosen by the fusion algorithm. Default=True
        scenario_plotter:  Path to scenario file or VaderScenarioPlotHelper object. This is used to add scenario
                           elements to plots.
        """

        # load global map global.csv -> [metaObject, DataFrame]
        _df_global = build_density_map(
            csv_path=global_data,
            index=cls.view_index[1:],  # ID will be set later
            column_types={
                "count": np.int,
                "measured_t": np.float,
                "received_t": np.float,
                "source": np.str,
                "own_cell": np.int,
                "node_id": np.str,
            },
            real_coords=real_coords,
            add_missing_cells=False,
        )

        # load map for each node *.csv -> list of DataFrames
        njobs = int(multiprocessing.cpu_count() * 0.60)
        pool = multiprocessing.Pool(processes=njobs)
        job_args = [
            {
                "csv_path": p,
                "index": cls.view_index[1:],
                "column_types": {
                    "count": np.int,
                    "measured_t": np.float,
                    "received_t": np.float,
                    "source": np.str,
                    "own_cell": np.int,
                    "selection": np.str,  # needed to filter out view form other data
                },
                "real_coords": real_coords,
                "add_missing_cells": False,
                "df_filter": filter_selected_cells,
            }
            for p in node_data
        ]
        _df_data = run_pool(pool, build_density_map, job_args)
        # pool.starmap(build_density, jobs)

        # create DcdMap2D from frames
        _obj = cls.from_frames(_df_global, _df_data, load_all)

        # add VaderScenarioPlotHelper if not provided
        if scenario_plotter is not None:
            if type(scenario_plotter) == str:
                _obj.set_scenario_plotter(VaderScenarioPlotHelper(scenario_plotter))
            else:
                _obj.set_scenario_plotter(scenario_plotter)
        return _obj

    @classmethod
    def from_frames(cls, global_data, node_data, load_all=True):
        glb_meta, glb_df = global_data

        # add selection column to global data to ensure the ground truth is not dropped in map_view_df
        glb_df["selection"] = "global"

        # create location df [x, y] -> nodeId (long string)
        location_df, glb_df = cls.extract_location_df(glb_df)

        # ID mapping
        location_df, node_id_map = cls.create_id_map(location_df, node_data)

        source_index = [
            cls.tsc_id_idx_name,
            cls.tsc_time_idx_name,
            cls.tsc_x_idx_name,
            cls.tsc_y_idx_name,
            cls.tsc_source_idx_name,
        ]

        # merge rest of map data and set index
        map_all_df = cls.merge_data_frames(
            [(glb_meta, glb_df), *node_data], node_id_map, source_index
        )

        if not load_all:
            # remove not selected values to save space and speed up other operations if they are
            map_all_df = map_all_df[map_all_df["selection"].notnull()].copy()

        map_all_df = cls.calculate_features(map_all_df)

        map_view_df = (
            map_all_df[map_all_df["selection"].notnull()]
            .drop(columns=["selection"])
            .reset_index(source_index[-1], drop=False)
        )
        if not map_view_df.index.is_unique:
            raise ValueError("map_view_df must have unique index")

        return cls(glb_meta, node_id_map, map_view_df, location_df, map_all_df)

    def __init__(
        self,
        glb_meta: DcdMetaData,
        node_id_map: dict,
        map_view_df: pd.DataFrame,
        location_df: pd.DataFrame,
        map_all_df: pd.DataFrame,
        load_all=True,
    ):
        """
        Parameters
        ----------
        glb_meta: Meta data instance for current map. cell size, map size
        node_id_map: Mapping between node
        """
        super().__init__(glb_meta, node_id_map, map_view_df, location_df)
        self.map_all_df = map_all_df
        self.all_data = load_all

    @staticmethod
    def extract_view(node_data):
        """
        select subset of node_data which is the density map used by the simulation.
        return: two list of (metadata, df) tuples for further processing
        """
        view_data = []
        for meta, _df in node_data:
            _m = _df["selection"].notnull()
            _view = _df[_m].copy().drop(columns=["selection"])
            if _view.index.is_unique == False:
                raise ValueError(f"view index not unique for id {meta.node_id}")
            view_data.append([meta, _view])
        return view_data

    def info_dict(self, x, y, time_step, node_id):
        info_dict = super().info_dict(x, y, time_step, node_id)
        try:
            others = self.map_all_df.loc[
                pd.IndexSlice[node_id, time_step, x, y, :],
                ["count", "measured_t", "received_t"],
            ]
            _data = []
            for index, row in others.iterrows():
                row_dict: dict = row.to_dict()
                row_dict.setdefault("_node_id", index[4])
                _data.append(row_dict)

            info_dict.setdefault("y_other_values_#", len(_data))
            info_dict.setdefault("y_other_mean_count", others.mean().loc["count"])
            info_dict.setdefault("z_other_values", _data)
        except KeyError as e:
            pass

        return info_dict
