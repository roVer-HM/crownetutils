import os
from functools import wraps
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import QuadMesh
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import IndexSlice as Idx

from roveranalyzer.simulators.crownet.dcd.util import DcdMetaData, create_error_df
from roveranalyzer.simulators.opp.provider.hdf.CountMapProvider import (
    CountMapHdfProvider,
)
from roveranalyzer.utils import logger
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
    tsc_global_id = 0

    def __init__(
        self,
        m_glb: DcdMetaData,
        id_map,
        location_df: pd.DataFrame,
        plotter=None,
        data_base_dir=None,
    ):
        self.meta = m_glb
        self.node_to_id = (
            id_map  # mapping of omnetpp identifier to data frame id's [1,..., max]
        )
        self.id_to_node = {
            v: k for k, v in id_map.items()
        }  # data frame id to omnetpp node identifier
        self.location_df = location_df
        self.scenario_plotter = plotter
        self.plot_wrapper = None
        self.font_dict = {
            "title": {"fontsize": 24},
            "xlabel": {"fontsize": 20},
            "ylabel": {"fontsize": 20},
            "legend": {"size": 20},
            "tick_size": 16,
        }
        self.data_base_dir = data_base_dir

    # def _load_or_create(self, pickle_name, create_f, *create_args):
    #     """
    #     Load data from hdf or create data based on :create_f:
    #     """
    #
    #     # just load data with provided create function
    #     if not self.lazy_load_from_hdf:
    #         print("create from scratch (no hdf)")
    #         return create_f(*create_args)
    #
    #     # load from hdf if exist and create if missing
    #     hdf_path = os.path.join(self.hdf_base_path, pickle_name)
    #     if os.path.exists(hdf_path):
    #         print(f"load from hdf {hdf_path}")
    #         return self.count_map_provider.get_dataframe()
    #     else:
    #         print("create from scratch ...", end=" ")
    #         data = create_f(*create_args)
    #         print(f"write to hdf {hdf_path}")
    #         self.count_map_provider.write_dataframe(data)
    #         return data

    def get_location(self, simtime, node_id, cell_id=False):
        try:
            ret = self.location_df.loc[simtime, node_id]
            if ret.shape == (2,):
                ret = ret.to_numpy()
                if cell_id:
                    ret = np.floor(ret / self.meta.cell_size)
            else:
                raise TypeError()
        except (TypeError, KeyError):
            ret = np.array([-1, -1])
        return ret

    def set_scenario_plotter(self, plotter):
        self.scenario_plotter = plotter


class DcdMap2D(DcdMap):
    """
    decentralized crowed map
    """

    tsc_id_idx_name = "ID"
    tsc_time_idx_name = "simtime"
    tsc_x_idx_name = "x"
    tsc_y_idx_name = "y"

    def __init__(
        self,
        glb_meta: DcdMetaData,
        node_id_map: dict,
        global_df: pd.DataFrame,
        map_df: pd.DataFrame,
        location_df: pd.DataFrame,
        plotter=None,
        **kwargs,
    ):
        super().__init__(glb_meta, node_id_map, location_df, plotter, **kwargs)
        self._map = map_df
        self._global_df = global_df
        self._count_map = None
        self.hdf_file_name = "DcdMap2D.hdf"
        self._count_map_provider = None

    def iter_nodes_d2d(self, first=0):
        _i = pd.IndexSlice

        data = self._map.loc[_i[first:], :].groupby(level=self.tsc_id_idx_name)
        for i, ts_2d in data:
            yield i, ts_2d

    @property
    def glb_map(self):
        return self._global_df

    @property
    def map(self):
        return self._map

    @property
    def count_map_provider(self):
        if self._count_map_provider is None:
            self._count_map_provider = CountMapHdfProvider(
                os.path.join(self.data_base_dir, self.hdf_file_name)
            )

            if not self._count_map_provider.exists():
                logger.info(f"create {self.hdf_file_name} from scratch ...")
                data = create_error_df(self.map, self.glb_map)
                logger.info(f"write to hdf {self._count_map_provider.hdf_path}")
                self.count_map_provider.write_dataframe(data)

        return self._count_map_provider

    @property
    def count_map(self):
        # lazy load data if needed
        if self._count_map is None:
            logger.warning(
                "complete count_map placed in memory. Use count_map_proivder"
            )
            self._count_map = self.count_map_provider.get_dataframe()
        return self._count_map

    def all_ids(self, with_ground_truth=True):
        ids = np.array(list(self.id_to_node.keys()))
        if not with_ground_truth:
            ids = ids[ids != 0]
        return ids

    def valid_times(self, _from=-1, _to=-1):
        time_values = self.map.index.get_level_values("simtime").unique().to_numpy()
        np.append(
            time_values,
            self.glb_map.index.get_level_values("simtime").unique().to_numpy(),
        )
        time_values = np.sort(np.unique(time_values))
        if _from >= 0:
            time_values = time_values[time_values >= _from]
        if _to >= 0:
            time_values = time_values[time_values < _to]
        return time_values

    def unique_level_values(self, level_name, df_slice=None):
        if df_slice is None:
            df_slice = ([self.tsc_global_id], [])

        idx = self.map.loc[df_slice].index.get_level_values(level_name).unique()
        return idx

    def age_mask(self, age_column, threshold):
        _mask = self._map[age_column] <= threshold
        return _mask

    def count_diff(self, diff_metric="diff"):
        metric = {
            "diff": lambda loc, glb: glb - loc,
            "abs_diff": lambda loc, glb: (glb - loc).abs(),
            "sqr_diff": lambda loc, glb: (glb - loc) ** 2,
        }
        _df = pd.DataFrame(self.map.groupby(["ID", "simtime"]).sum()["count"])
        _df = _df.combine(self.glb_map, metric[diff_metric])
        _df = _df.rename(columns={"count": "count_diff"})
        return _df

    def create_2d_map(self, time_step, node_id, value_name):
        """
        create 2d matrix of density map for one instance in
        time and for one node. The returned data frame as a shape of
         (N, M) where N,M is the number of cells in X respectively Y axis
        """
        data = pd.DataFrame(
            self.count_map_provider.select_simtime_and_node_id_exact(
                time_step, node_id
            )[value_name]
        )
        data = data.set_index(data.index.droplevel([0, 3]))  # (x,y) as new index
        full_index = self.meta.grid_index_2d(real_coords=True)
        df = pd.DataFrame(np.zeros((len(full_index), 1)), columns=[value_name])
        df = df.set_index(full_index)
        df.update(data)
        df = df.unstack().T
        return df

    def update_color_mesh(self, qmesh: QuadMesh, time_step, node_id, value_name):
        df = self.create_2d_map(time_step, node_id, value_name)
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
        except KeyError:
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
            data = self._map
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

    def plot_summary(self, simtime, node_id, title="", **kwargs):
        kwargs.setdefault("figsize", (16, 9))
        f, ax = plt.subplots(2, 2, **kwargs)
        ax = ax.flatten()
        f.suptitle(
            f"Node {node_id}_{self.id_to_node[node_id]} for time {simtime} {title}"
        )
        self.area_plot(simtime, node_id, ax=ax[0])
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

        for _id, df in places.groupby(level=self.tsc_id_idx_name):
            # move coordinate for node :id: to center of cell.
            df = df + 0.5 * self.meta.cell_size
            ax.scatter(df["x"], df["y"], label=f"{_id}_{self.id_to_node[_id]}")

        if add_legend:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        return f, ax

    @plot_decorator
    def plot_location_map_annotated(self, time_step, *, ax=None):
        places = self.own_cell()
        _i = pd.IndexSlice
        places = places.loc[_i[:, time_step], :]  # select only the needed timestep
        f, ax = self.plot_location_map(time_step, ax, add_legend=False)

        # places = places.droplevel("simtime")
        places = self.create_label_positions(places)
        for _id, df in places.groupby(level=self.tsc_id_idx_name):
            # move coordinate for node :id: to center of cell.
            ax.annotate(
                text=_id,
                xy=df.loc[(_id, time_step), ["x_center", "y_center"]].to_numpy(),
                xytext=df.loc[(_id, time_step), ["x_text", "y_text"]].to_numpy(),
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
        line: Line2D = None,
        bins_width=2.5,
    ):
        df = self.map
        # remove cells with count 0
        if remove_null:
            df = self.map.loc[self.map["count"] != 0]

        # average over distance (ignore node_id)
        if bins_width > 0:
            df = df.loc[Idx[:, time_step], ["owner_dist", delay_kind]]
            bins = int(np.floor(df["owner_dist"].max() / bins_width))
            df = df.groupby(pd.cut(df["owner_dist"], bins)).mean().dropna()
            df.index = df.index.rename("dist_bin")
        else:
            df = df.loc[Idx[node_id, time_step], ["owner_dist", delay_kind]]

        df = df.sort_values(axis=0, by=["owner_dist"])
        if line is not None:
            line.set_ydata(df[delay_kind].to_numpy())
            line.set_xdata(df["owner_dist"].to_numpy())
        return df

    def apply_ax_props(self, ax: plt.Axes, ax_prop: dict):
        for k, v in ax_prop.items():
            getattr(ax, f"set_{k}", None)(v, **self.font_dict[k])

    def update_error_over_distance(
        self,
        time_step,
        node_id,
        value,
        line: Line2D = None,
        bins_width=2.5,
    ):

        df = self.count_map_provider.select_simtime_and_node_id_exact(
            time_step, node_id
        )[["owner_dist", value]]
        df = df.reset_index(drop=True)

        # average over distance (ignore node_id)
        if bins_width > 0:
            bins = int(np.floor(df["owner_dist"].max() / bins_width))
            df = df.groupby(pd.cut(df["owner_dist"], bins)).mean().dropna()
            df.index = df.index.rename("dist_bin")

        df = df.sort_values(axis=0, by=["owner_dist"])
        if line is not None:
            line.set_ydata(df[value].to_numpy())
            line.set_xdata(df["owner_dist"].to_numpy())
        return df

    @plot_decorator
    def plot_error_over_distance(
        self,
        time_step,
        node_id,
        value,
        label=None,
        *,
        ax=None,
        fig_dict: dict = None,
        ax_prop: dict = None,
        **kwargs,
    ):

        f, ax = check_ax(ax, **fig_dict if fig_dict is not None else {})
        df = self.update_error_over_distance(
            time_step, node_id, value, line=None, **kwargs
        )

        if label is None:
            label = value
        ax.plot("owner_dist", value, data=df, label=label)

        ax_prop = {} if ax_prop is None else ax_prop
        ax_prop.setdefault(
            "title",
            f"Error({value}) over Distance",
        )
        ax_prop.setdefault("xlabel", "Cell distance (euklid) to owners location [m]")
        ax_prop.setdefault("ylabel", f"{value}")

        ax.lines[0].set_linestyle("None")
        ax.lines[0].set_marker("o")

        self.apply_ax_props(ax, ax_prop)

        return f, ax

    @plot_decorator
    def plot_delay_over_distance(
        self,
        time_step,
        node_id,
        value,
        remove_null=True,
        label=None,
        *,
        ax: plt.Axes = None,
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
            time_step, node_id, value, remove_null=remove_null, line=None, **kwargs
        )

        if label is None:
            label = value
        ax.plot("owner_dist", value, data=df, label=label)

        ax_prop = {} if ax_prop is None else ax_prop
        ax_prop.setdefault(
            "title",
            f"Delay({value}) over Distance",
        )
        ax_prop.setdefault("xlabel", "Cell distance (euklid) to owners location [m]")
        ax_prop.setdefault("ylabel", "Delay in [s]")

        ax.lines[0].set_linestyle("None")
        ax.lines[0].set_marker("o")

        self.apply_ax_props(ax, ax_prop)

        return f, ax

    @plot_decorator
    def area_plot(
        self,
        time_step,
        node_id,
        value,
        *,
        ax=None,
        pcolormesh_dict: dict = None,
        fig_dict: dict = None,
        ax_prop: dict = None,
        **kwargs,
    ):
        """
        Birds eyes view of density in a 2D color mesh with X/Y spanning the
        area under observation. Z axis (density) is shown with given color grading.

        Default data view: per node / per time / all cells
        """
        df = self.create_2d_map(time_step, node_id, value)
        f, ax = check_ax(ax, **fig_dict if fig_dict is not None else {})

        cell = self.get_location(time_step, node_id, cell_id=False)
        if "title" in kwargs:
            ax.set_title(kwargs["title"], **self.font_dict["title"])
        else:
            ax.set_title(
                f"Area plot of '{value}'. node: {node_id}/{self.id_to_node[node_id]} time: "
                f"{time_step} cell [{cell[0]}, {cell[1]}]",
                **self.font_dict["title"],
            )
        ax.set_aspect("equal")
        ax.tick_params(axis="x", labelsize=self.font_dict["tick_size"])
        ax.tick_params(axis="y", labelsize=self.font_dict["tick_size"])

        if self.scenario_plotter is not None:
            self.scenario_plotter.add_obstacles(ax)

        _d = update_dict(pcolormesh_dict, shading="flat")

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

        for _id, df in self.iter_nodes_d2d(first=1):
            df_time = df.groupby(level=self.tsc_time_idx_name).sum()
            ax.plot(
                df_time.index, df_time["count"], label=f"{_id}_{self.id_to_node[_id]}"
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
            self._map.loc[_i[:], _i["count"]]
            .groupby(level=[self.tsc_id_idx_name, self.tsc_time_idx_name])
            .sum()
            .groupby(level="simtime")
            .mean()
        )
        nodes_std = (
            self._map.loc[_i[:], _i["count"]]
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


class DcdMap2DMulti(DcdMap2D):
    tsc_source_idx_name = "source"

    def __init__(
        self,
        glb_meta: DcdMetaData,
        node_id_map: dict,
        global_df: pd.DataFrame,
        map_df: pd.DataFrame,
        location_df: pd.DataFrame,
        map_all_df: pd.DataFrame,
        **kwargs,
    ):
        """
        Parameters
        ----------
        glb_meta: Meta data instance for current map. cell size, map size
        node_id_map: Mapping between node
        """
        super().__init__(
            glb_meta, node_id_map, global_df, map_df, location_df, **kwargs
        )
        self.map_all_df = map_all_df

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
        except KeyError:
            pass

        return info_dict
