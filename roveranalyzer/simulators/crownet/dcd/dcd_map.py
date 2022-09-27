from __future__ import annotations

import os
from itertools import combinations
from typing import Callable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import IndexSlice as Idx

import roveranalyzer.utils.dataframe as FrameUtl
import roveranalyzer.utils.plot as _Plot
from roveranalyzer.simulators.crownet.common.dcd_util import DcdMetaData
from roveranalyzer.simulators.opp.provider.hdf.DcdMapCountProvider import DcdMapCount
from roveranalyzer.simulators.opp.provider.hdf.DcdMapProvider import DcdMapProvider
from roveranalyzer.utils import logger
from roveranalyzer.utils.misc import intersect
from roveranalyzer.utils.plot import Style, check_ax, update_dict

PlotUtil = _Plot.PlotUtil


class DcdMap:
    tsc_global_id = 0

    def __init__(
        self,
        metadata: DcdMetaData,
        position_df: pd.DataFrame,
        plotter=None,
        data_base_dir=None,
    ):
        self.metadata = metadata
        self.position_df = position_df
        self.scenario_plotter = plotter
        self.plot_wrapper = None
        self.style = Style()
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
    #         return self.count_p.get_dataframe()
    #     else:
    #         print("create from scratch ...", end=" ")
    #         data = create_f(*create_args)
    #         print(f"write to hdf {hdf_path}")
    #         self.count_p.write_dataframe(data)
    #         return data

    def get_location(self, simtime, node_id, cell_id=False):
        try:
            ret = self.position_df.loc[simtime, node_id]
            if ret.shape == (2,):
                ret = ret.to_numpy()
                if cell_id:
                    ret = np.floor(ret / self.metadata.cell_size)
            else:
                raise TypeError()
        except (TypeError, KeyError):
            ret = np.array([-1, -1])
        return ret

    def set_scenario_plotter(self, plotter):
        self.scenario_plotter = plotter


def percentile(n):
    def _percentil(x):
        return np.percentile(x, float(n) * 100.0)

    _percentil.__name__ = f"p_{n*100:2.0f}"
    return _percentil


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
        metadata: DcdMetaData,
        global_df: pd.DataFrame,
        map_df: Union[pd.DataFrame, None],
        position_df: pd.DataFrame,
        count_p: DcdMapCount = None,
        count_slice: pd.IndexSlice = None,
        map_p: DcdMapProvider = None,
        map_slice: pd.IndexSlice = None,
        plotter=None,
        **kwargs,
    ):
        super().__init__(metadata, position_df, plotter, **kwargs)
        self._map = map_df
        self._global_df = global_df
        self._count_map = None
        self._count_p = count_p
        self._count_slice = count_slice

        self._map_p: DcdMapProvider = map_p
        self._map_slice = map_slice

    def iter_nodes_d2d(self, first_node_id=0):
        # index order: [time, x, y, source, node]
        _i = pd.IndexSlice

        data = self.map.loc[_i[:, :, :, :, first_node_id:], :].groupby(
            level=self.tsc_id_idx_name
        )
        for i, ts_2d in data:
            yield i, ts_2d

    @property
    def glb_map(self):
        return self._global_df

    @property
    def map(self):
        if self._map is None:
            logger.info("load map")
            self._map = self._map_p[self._map_slice, :]
            self._map = self._map.sort_index()
        return self._map

    @property
    def count_p(self):
        if self._count_p is None:
            raise ValueError("count map is not setup")
        return self._count_p

    @property
    def count_map(self):
        # lazy load data if needed
        if self._count_map is None:
            logger.info("load count map from HDF")
            self._count_map = self._count_p[self._count_slice, :]
        return self._count_map

    def all_ids(self, with_ground_truth=True):
        ids = self.position_df.index.get_level_values("node_id").unique().to_numpy()
        ids.sort()
        # ids = np.array(list(self.id_to_node.keys()))
        if with_ground_truth:
            np.insert(ids, 0, 0)
            # ids = ids[ids != 0]
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
        _mask = self.map[age_column] <= threshold
        return _mask

    def update_area(self, time_step, node_id, value_name):
        """
        create 2d matrix of density map for one instance in
        time and for one node. The returned data frame as a shape of
         (N, M) where N,M is the number of cells in X respectively Y axis
        """
        data = pd.DataFrame(
            self.count_p.select_simtime_and_node_id_exact(time_step, node_id)[
                value_name
            ]
        )
        data = data.set_index(data.index.droplevel([0, 3]))  # (x,y) as new index
        df = self.metadata.update_missing(data, real_coords=True)
        df.update(data)
        df = df.unstack().T
        return df

    def update_color_mesh(self, qmesh: QuadMesh, time_step, node_id, value_name):
        df = self.update_area(time_step, node_id, value_name)
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
        own_cell_mask = self.map["own_cell"] == 1
        places = (
            self.map[own_cell_mask]  # only own cells (cells where one node is located)
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
        places["x_center"] = places["x"] + 0.5 * self.metadata.cell_size
        places["y_center"] = places["y"] + 0.5 * self.metadata.cell_size
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
            data = self.map
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

    @PlotUtil.savefigure
    @PlotUtil.plot_decorator
    def plot_summary(self, simtime, node_id, title="", **kwargs):
        kwargs.setdefault("figsize", (16, 9))
        f, ax = plt.subplots(2, 2, **kwargs)
        ax = ax.flatten()
        f.suptitle(f"Node {node_id} for time {simtime} {title}")
        self.plot_area(simtime, node_id, ax=ax[0])
        self.plot_location_map(simtime, ax=ax[1])
        self.plot_count(ax=ax[2])
        ax[3].clear()
        return f, ax

    @PlotUtil.savefigure
    @PlotUtil.with_axis
    @PlotUtil.plot_decorator
    def plot_location_map(self, time_step, *, ax: plt.Axes = None, add_legend=True):
        places = self.own_cell()
        _i = pd.IndexSlice
        places = places.loc[_i[:, time_step], :]  # select only the needed timestep

        ax.set_title(f"Node Placement at time {time_step}s")
        ax.set_aspect("equal")
        ax.set_xlim([0, self.metadata.x_dim])
        ax.set_ylim([0, self.metadata.y_dim])
        if self.scenario_plotter is not None:
            self.scenario_plotter.add_obstacles(ax)

        for _id, df in places.groupby(level=self.tsc_id_idx_name):
            # move coordinate for node :id: to center of cell.
            df = df + 0.5 * self.metadata.cell_size
            ax.scatter(df["x"], df["y"], label=f"{_id}")

        if add_legend:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        return ax.get_figure(), ax

    @PlotUtil.savefigure
    @PlotUtil.with_axis
    @PlotUtil.plot_decorator
    def plot_location_map_annotated(self, time_step, *, ax: plt.Axes = None):
        places = self.own_cell()
        _i = pd.IndexSlice
        places = places.loc[_i[:, time_step], :]  # select only the needed timestep
        f, ax = self.plot_location_map(time_step, ax=ax, add_legend=False)

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
            getattr(ax, f"set_{k}", None)(v, **self.style.font_dict[k])

    def update_cell_error(
        self,
        time_slice: slice,
        value: str = "err",
        agg_func: Union[Callable, str, list, dict] = "mean",
        drop_index: bool = False,
        name: Union[str, None] = None,
        time_bin: Union[float, None] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame():
        """
        Aggregated cell error over all nodes over a given time.
        """
        # select time slice. Do not select ground truth (ID = 0)
        df = self.count_p[pd.IndexSlice[time_slice, :, :, 1:], value]

        df = df.groupby(by=["x", "y"]).aggregate(func=agg_func, *args, **kwargs)
        if name is not None:
            df = df.rename(columns={value: name})
        if drop_index:
            df = df.reset_index(drop=True)
        return df

    def update_error_over_distance(
        self,
        time_step,
        node_id,
        value,
        line: Line2D = None,
        bins_width=2.5,
    ):
        df = self.count_p.select_simtime_and_node_id_exact(time_step, node_id)[
            ["owner_dist", value]
        ]
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

    def error_hist(
        self,
        time_slice: slice = slice(None),
        value="err",
        agg_func="mean",
    ):

        if time_slice == slice(None):
            _ts = self.count_p.get_time_interval()
            time_slice = slice(_ts[0], _ts[1])
        data = self.update_cell_error(time_slice, value, agg_func, drop_index=True)
        return data, time_slice

    @PlotUtil.savefigure
    @PlotUtil.with_axis
    @PlotUtil.plot_decorator
    def plot_error_histogram(
        self,
        time_slice: slice = slice(None),
        value="err",
        agg_func="mean",
        data_source=None,
        *,
        stat: str = "count",  # "percent"
        fill: bool = True,
        ax: plt.Axes = None,
        **hist_kwargs,
    ):
        if data_source is not None:
            data, time_slice = data_source(time_slice, value, agg_func)
        else:
            data, time_slice = self.error_hist(time_slice, value, agg_func)

        _t = f"Cell count Error ('{value}') for Time {time_slice.start}"
        if time_slice.stop is not None:
            _t += f" - {time_slice.stop}"
        ax.set_title(_t)
        ax.set_xlabel(f"{value}")

        ax = sns.histplot(data=data, stat=stat, fill=fill, ax=ax, **hist_kwargs)
        return ax.get_figure(), ax

    @PlotUtil.savefigure
    @PlotUtil.with_axis
    @PlotUtil.plot_decorator
    def plot_error_quantil_histogram(
        self,
        value="err",
        agg_func="mean",
        *,
        stat: str = "count",  # "percent"
        fill: bool = False,
        ax: plt.Axes = None,
        **hist_kwargs,
    ):
        tmin, tmax = self.count_p.get_time_interval()
        time = (tmax - tmin) / 4
        intervals = {
            f"Time Quantil {i+1}": slice(time * i, time * i + time) for i in range(4)
        }

        data = self.update_cell_error(
            slice(None), value, agg_func, name="All", drop_index=True
        )
        quant = [
            self.update_cell_error(v, value, agg_func, name=k, drop_index=True)
            for k, v in intervals.items()
        ]
        data = pd.concat([data, *quant], axis=1)

        if isinstance(ax, plt.Axes):
            ax.set_title("Cell Error Histogram")
            ax.set_xlabel(value)

            sns.histplot(
                data=data,
                stat=stat,
                common_norm=False,
                fill=fill,
                legend=True,
                ax=ax,
                **hist_kwargs,
            )
            return ax.get_figure(), ax

        elif isinstance(ax, np.ndarray) and len(ax) == 5:
            for idx in range(len(ax)):
                _ax = ax[idx]
                _ax.set_title(f"Cell Error Histogram {data.iloc[:, idx].name}")
                _ax.set_xlabel(value)
                sns.histplot(
                    data=data.iloc[:, idx],
                    stat=stat,
                    common_norm=False,
                    fill=fill,
                    legend=True,
                    ax=_ax,
                    **hist_kwargs,
                )
            return ax[0].get_figure(), ax
        else:
            raise ValueError(f"Expected ax or array of 5 axes but got {type(ax)}")

    @PlotUtil.savefigure
    @PlotUtil.plot_decorator
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

    @PlotUtil.savefigure
    @PlotUtil.plot_decorator
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

    @PlotUtil.savefigure
    @PlotUtil.plot_decorator
    def plot_area(
        self,
        time_step: float,
        node_id: int,
        value: str,
        *,
        ax=None,
        pcolormesh_dict: dict = None,
        fig_dict: dict = None,
        ax_prop: dict = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Birds eyes view of density in a 2D color mesh with X/Y spanning the
        area under observation. Z axis (density) is shown with given color grading.

        Default data view: per node / per time / all cells
        """
        df = self.update_area(time_step, node_id, value)
        f, ax = check_ax(ax, **fig_dict if fig_dict is not None else {})

        cell = self.get_location(time_step, node_id, cell_id=False)
        if "title" in kwargs:
            ax.set_title(kwargs["title"], **self.style.font_dict["title"])
        else:
            ax.set_title(
                f"Area plot of '{value}'. node: {node_id} time: "
                f"{time_step} cell [{cell[0]}, {cell[1]}]",
                **self.style.font_dict["title"],
            )
        ax.set_aspect("equal")
        ax.tick_params(axis="x", labelsize=self.style.font_dict["tick_size"])
        ax.tick_params(axis="y", labelsize=self.style.font_dict["tick_size"])

        # if self.scenario_plotter is not None:
        #     self.scenario_plotter.add_obstacles(ax)

        _d = update_dict(pcolormesh_dict, shading="flat")

        pcm = ax.pcolormesh(self.metadata.X_flat, self.metadata.Y_flat, df, **_d)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.set_label("colorbar")
        cax.tick_params(axis="y", labelsize=self.style.font_dict["tick_size"])
        f.colorbar(pcm, cax=cax)

        ax.update(ax_prop if ax_prop is not None else {})

        return f, ax

    @PlotUtil.savefigure
    @PlotUtil.with_axis
    @PlotUtil.plot_decorator
    def plot_count(self, *, ax=None, **kwargs) -> Tuple[Figure, Axes]:
        ax.set_title("Total node count over time", **self.font_dict["title"])
        ax.set_xlabel("time [s]", **self.font_dict["xlabel"])
        ax.set_ylabel("total node count", **self.font_dict["ylabel"])

        for _id, df in self.iter_nodes_d2d(first_node_id=1):
            df_time = df.groupby(level=self.tsc_time_idx_name).sum()
            ax.plot(df_time.index, df_time["count"], label=f"{_id}")

        g = self.glb_map.groupby(level=self.tsc_time_idx_name).sum()
        ax.plot(
            g.index.get_level_values(self.tsc_time_idx_name),
            g["count"],
            label=f"0",
        )
        ax.legend()
        return ax.get_figure(), ax

    def map_count_measure(self, load_cached_version: bool = True) -> pd.DataFrame:
        """create map based error measure over time to indicate **total area count correctness**

        Get map count measure that shows how good the number of agents are
        represented by the density map irrespective of there positions. Meaning
        this error measure only shows if no agents are left out or are 'produced' by
        the density map. There positional information is lost in this measure.

        Returns:
            pd.DataFrame: [index](columns): [simtime](
                glb_map_count, mean_map_count, median_map_count,
                map_mean_err, map_mean_sqrerr, map_median_err, map_median_sqerr
                )
        """
        if self._map_p.contains_group("map_measure") and load_cached_version:
            return self._map_p.get_dataframe(group="map_measure")

        _i = pd.IndexSlice
        nodes: pd.DataFrame = (
            self.count_p[_i[:, :, :, 1:], ["count"]]  # all put ground truth
            .groupby(level=[self.tsc_id_idx_name, self.tsc_time_idx_name])  # ID|time
            .sum()
            .groupby(level="simtime")
            .agg(
                [
                    "mean",
                    percentile(0.5),
                    percentile(0.25),
                    percentile(0.75),
                    "min",
                    "max",
                ]
            )
        )
        nodes = nodes.rename(
            columns={
                "p_50": "map_median_count",
                "mean": "map_mean_count",
                "p_25": "map_count_p25",
                "p_75": "map_count_p75",
                "min": "map_count_min",
                "max": "map_count_max",
            }
        )
        nodes.columns = nodes.columns.droplevel(0)
        glb = (
            self.count_p[_i[:, :, :, 0], _i["count"]]  # only ground truth
            .groupby(level=[self.tsc_time_idx_name])
            .sum()
        )
        glb.columns = ["map_glb_count"]

        df = pd.concat([glb, nodes], axis=1)
        df["map_mean_err"] = df["map_mean_count"] - df["map_glb_count"]
        df["map_mean_sqrerr"] = np.power(df["map_mean_err"], 2)
        df["map_median_err"] = df["map_median_count"] - df["map_glb_count"]
        df["map_median_sqerr"] = np.power(df["map_median_err"], 2)

        return df

    def cell_count_measure(
        self,
        load_cached_version: bool = True,
        index_slice: slice | Tuple(slice) = slice(None),
        xy_slice: Tuple(slice) | pd.MultiIndex = (slice(None), slice(None)),
        columns: slice | List[str] = slice(None),
    ) -> pd.DataFrame:
        """create cell based error measures over time to indicate **positional correctness**

        cell_slice: defines the cells used for the calculation. If None use all available cells
        from the raw data. This will affect the denominator 1/N in the calculations below.

        count_p contains count, err, sqerr values at the (time, id, x, y) level.
        In other words the table contains the count err, squerr values for
        each node (id) for a given cell (x, y) for a given time (time).
        The table count_p only communicated cells as well as over and underestimation
        errors.

        Assume cell x_i was occupied until t=10 . Then this cell is reported for each
        time step and each node. Either with err = 0 if the node sees the occupant or
        err = -1 for nodes where the occupant is not seen and err >= 1 in the case some
        nodes see more than one node. Note that negative values (underestimation) is bound
        by the real number of occupants in the cell. On the other hand overestimation is
        not bound!

        Assume now t > 100 and x_i is not occupied anymore and any TTL is reached, thus
        no node should have any values for the cell x_i.
        Assume now that from a total of N=10 nodes one node is faulty and has a
        count for cell x_i. This count will be part of the count_p table and
        marked with an error count of 1. The correct value of count=0 for all
        other nodes is not stored in count_p but are implied. Thus to calculate the
        mean error of cell x_i is:

                mean_abs_err = (|1| + 9*|0|)/(N=10) = 0.1

        For this reason a simple count_p.groupby([...]).mean() will not work because
        the used number of observations will be wrong because the implied zero-error
        values are not saved in count_p. This function will therefore calculate the mean
        errors manually by utilizing the total number of nodes N for each time `t`. The
        numerator will be the same because only zero-counts / zero-erros are implied. Any
        non-zero count or error will be saved explicitly in count_p.

            N:= set of cells (x, y) with index i
            M:= set of agents/measuring agents (ID) with index j
            Y^_i := (Y-Hat) ground truth for cell i. This is identical for each agents thus
                    Y^_ij - Y^_i(j+1) for all i and j.

        Returns:
            pd.DataFrame: _description_
        """
        if self._map_p.contains_group("cell_measures") and load_cached_version:
            return self._map_p.get_dataframe(group="cell_measure").loc[
                index_slice, columns
            ]

        _i = pd.IndexSlice
        # total number of nodes at each time
        if isinstance(xy_slice, pd.MultiIndex):
            glb = self.count_p[_i[:, :, :, 0], _i["count"]]  # only ground truth
            glb = FrameUtl.partial_index_match(glb, xy_slice)
        else:
            glb = self.count_p[
                _i[:, xy_slice[0], xy_slice[1], 0], _i["count"]
            ]  # only ground truth
        glb = glb.droplevel("ID")
        glb.columns = ["glb_count"]
        glb_map_sum = glb.groupby("simtime").sum()  # [simtime](count) aka. M
        glb_map_sum.columns = ["num_Agents"]

        # all (time, x, y, id) based count, err, squerr cell values
        # without ground truth (see slice last slice `1:`)
        # The measurements are summed over all nodes (this will drop the id index )
        if isinstance(xy_slice, pd.MultiIndex):
            nodes: pd.DataFrame = self.count_p[
                _i[:, :, :, 1:], _i["count", "err", "sqerr"]
            ]  # all but ground truth
            nodes = FrameUtl.partial_index_match(nodes, xy_slice)
        else:
            nodes: pd.DataFrame = self.count_p[
                _i[:, xy_slice[0], xy_slice[1], 1:], _i["count", "err", "sqerr"]
            ]  # all but ground truth
        nodes["abserr"] = np.abs(nodes["err"])

        # metric III 1/N sum^N_i[ 1/M sum^M_j (Y_ij - Y^_i)^2 ]
        # create sum: sum^M_j[*]  with [*] is nodes["sqerr"] = (Y_ij - Y^_i)^2 and nodes["count"] = (Y_ij)
        cell_base: pd.DateFrame = nodes.groupby(
            level=[self.tsc_time_idx_name, self.tsc_x_idx_name, self.tsc_y_idx_name]
        ).agg(
            ["sum"]
        )  # [time, x, y](...data-columns...)

        cell_base.columns = [f"{a}_{b}" for a, b in cell_base.columns]
        # join total number of agents (aka. M) with cell based measures. See function description
        cell_base: pd.DataFrame = cell_base.join(glb_map_sum, on="simtime")
        cell_base: pd.DataFrame = cell_base.join(glb, on=["simtime", "x", "y"])
        cell_base["glb_count"] = cell_base["glb_count"].fillna(value=0)
        if cell_base.isna().any().any():
            raise ValueError(
                f"Found coulmns with nan values: {cell_base.isna().any(axis=0)}"
            )

        # divide by total number of nodes at each time to create mean measruements
        cell_base["cell_mean_count_est"] = (
            cell_base["count_sum"] / cell_base["num_Agents"]
        )  # 1/M sum^M_j (Y_ij) -> neee for metric II
        cell_base["cell_mean_est_sqerr"] = np.power(
            cell_base["cell_mean_count_est"] - cell_base["glb_count"], 2
        )  # (1/M sum^M_j (Y_ij) - Y^_i)^2 -> needed for metric II

        cell_base["cell_mse"] = (
            cell_base["sqerr_sum"] / cell_base["num_Agents"]
        )  # 1/M sum^M_j (Y_ij - Y^_i)^2 -> needed for metric III
        cell_base["cell_mean_err"] = (
            cell_base["err_sum"] / cell_base["num_Agents"]
        )  # 1/M sum^M_j (Y_ij - Y^_i) -> optional
        cell_base["cell_mean_abserr"] = (
            cell_base["abserr_sum"] / cell_base["num_Agents"]
        )  # 1/M sum^M_j |Y_ij - Y^_i| -> optional

        # remove uncessary columns
        # cell_base = cell_base.drop(columns=["err_sum", "count_sum", "sqerr_sum", "abserr_sum"])
        return cell_base.loc[index_slice, columns]

    def count_diff(
        self, val: set = {}, agg: set = {}, id_slice: slice | int = slice(1, None, None)
    ) -> pd.DataFrame:

        if len(agg) == 0:
            agg = {
                "count",
                "mean",
                "std",
                "min",
                percentile(0.25),
                percentile(0.50),
                percentile(0.75),
                "max",
            }
        if len(val) == 0:
            val = {"count", "err", "sqerr"}

        if self._map_p.contains_group("count_diff"):
            nodes = self._map_p.get_dataframe(group="count_diff")
        else:
            _i = pd.IndexSlice
            df = []
            if "abs_err" in val:
                val.remove(abs_err)
                abs_err = np.abs(self.count_p[_i[:, :, :, id_slice], _i["err"]])
                abs_err.columns = ["abs_err"]
                abs_err = (
                    abs_err.groupby(
                        level=[self.tsc_id_idx_name, self.tsc_time_idx_name]
                    )
                    .sum()
                    .groupby(level="simtime")
                    .agg(list(agg))
                )
                abs_err.columns = [f"{c}_{stat}" for c, stat in abs_err.columns]

            # stat_list = list(val)
            stat_list = ["count"]  # todo err values wrong after sum()
            # todo err values wrong
            nodes = (
                self.count_p[_i[:, :, :, id_slice], stat_list]  # all put ground truth
                .groupby(level=[self.tsc_id_idx_name, self.tsc_time_idx_name])
                .sum()
                .groupby(level="simtime")
                .agg(list(agg))
            )
            nodes.columns = [f"{c}_{stat}" for c, stat in nodes.columns]
            df.insert(0, nodes)

            glb = (
                self.count_p[_i[:, :, :, 0], _i["count"]]  # only ground truth
                .groupby(level=[self.tsc_time_idx_name])
                .sum()
            )
            glb.columns = ["glb_count"]
            df.insert(0, glb)  # at front!

            # glb = self.glb_map.groupby(level=self.tsc_time_idx_name).sum()["count"]
            nodes = pd.concat(df, axis=1)

        return nodes

    @PlotUtil.savefigure
    @PlotUtil.with_axis
    @PlotUtil.plot_decorator
    def plot_map_count_diff(self, *, ax=None, **kwargs) -> Tuple[Figure, Axes]:
        if "data_source" in kwargs:
            nodes = kwargs["data_source"]()
        else:
            nodes = self.map_count_measure()

        font_dict = self.style.font_dict
        ax.set_title("Node Count over Time", **font_dict["title"])
        ax.set_xlabel("Time [s]", **font_dict["xlabel"])
        ax.set_ylabel("Pedestrian Count", **font_dict["ylabel"])
        n = (
            nodes.loc[:, ["map_median_count", "map_count_p25", "map_count_p75"]]
            .dropna()
            .reset_index()
        )
        ax.plot("simtime", "map_median_count", data=n, label="Median count")

        ax.fill_between(
            n["simtime"],
            n["map_count_p25"],
            n["map_count_p75"],
            alpha=0.35,
            interpolate=True,
            label="[Q1;Q3]",
        )
        glb = nodes.loc[:, ["map_glb_count"]].dropna().reset_index()
        ax.plot("simtime", "map_glb_count", data=glb, label="Actual count")
        if self.style.create_legend:
            ax.get_figure().legend()
        return ax.get_figure(), ax

    def err_box_over_time(self, bin_width=10.0):
        i = pd.IndexSlice
        data = (
            self._count_p[i[:, :, :, 1:], ["err"]]
            .groupby(level=[self.tsc_id_idx_name, self.tsc_time_idx_name])
            .sum()
            .groupby(level="simtime")
            .mean()
        )
        bins = int(np.floor(data.index.max() / bin_width))
        _cut = pd.cut(data.index, bins)
        return data, _cut

    @PlotUtil.savefigure
    @PlotUtil.plot_decorator
    def plot_err_box_over_time(
        self, bin_width=10.0, xtick_sep=5, *, ax=None, **kwargs
    ) -> Tuple[Figure, Axes]:

        if "data_source" in kwargs:
            data, _cut = kwargs["data_source"](bin_width)
        else:
            data, _cut = self.err_box_over_time(bin_width)

        f, ax = check_ax(ax, **kwargs)
        font_dict = self.style.font_dict
        ax.set_title("Node Count over Time", **font_dict["title"])
        ax.set_xlabel("Time [s]", **font_dict["xlabel"])
        ax.set_ylabel("Count Err", **font_dict["ylabel"])

        flierprops = kwargs.get(
            "flierprops",
            dict(
                marker=".",
                color="black",
                markerfacecolor="black",
                markersize=3,
                linestyle="none",
            ),
        )
        boxprops = kwargs.get("boxprops", dict(color="black"))
        whiskerprops = kwargs.get("whiskerprops", dict(color="black"))

        ax = data.groupby(_cut).boxplot(
            subplots=False,
            sharey=True,
            grid=False,
            ax=ax,
            zorder=100,
            flierprops=flierprops,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
        )
        xlabels = data.reset_index().groupby(_cut)["simtime"].max().to_list()
        ax.set_xticklabels(xlabels)
        for idx, lbl in enumerate(ax.get_xticklabels()):
            if (
                idx == 0
                or idx == (len(ax.get_xticklabels()) - 1)
                or (idx % (xtick_sep - 1) == 0)
            ):
                lbl.set_visible(True)
            else:
                lbl.set_visible(False)
        ax.grid(alpha=0.2, zorder=10)
        ax.axhline(y=0.0, color="red", alpha=0.3, zorder=20)
        return f, ax


class DcdMap2DMulti(DcdMap2D):
    tsc_source_idx_name = "source"

    def __init__(
        self,
        metadata: DcdMetaData,
        global_df: pd.DataFrame,
        map_df: pd.DataFrame,
        position_df: pd.DataFrame,
        map_all_df: pd.DataFrame,
        **kwargs,
    ):
        """
        Parameters
        ----------
        metadata: Meta data instance for current map. cell size, map size
        node_id_map: Mapping between node
        """
        super().__init__(metadata, global_df, map_df, position_df, **kwargs)
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
