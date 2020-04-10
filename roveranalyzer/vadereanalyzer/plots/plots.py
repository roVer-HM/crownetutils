import datetime
import itertools
import os
import sys
from enum import Enum
from multiprocessing import Pool, TimeoutError

import matplotlib.animation as animation
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

import trimesh
from uitls import Timer
from uitls.mesh import SimpleMesh
from uitls.plot_helper import PlotHelper
from vadereanalyzer.plots.custom_tripcolor import tripcolor_costum

sys.path.append(
    os.path.abspath("")
)  # in case tutorial is called from the root directory
sys.path.append(os.path.abspath(".."))  # in tutorial directly


class PlotOptions(Enum):
    COUNT = (1, "counts")
    DENSITY = (2, "density")
    DENSITY_SMOOTH = (3, "density_smooth")


def mono_cmap(
    replace_with=(0.0, 0.0, 0.0, 0.0),
    replace_index=(0, 1),
    base_color=0,
    cspace=(0.0, 1.0),
    n=256,
):
    start, stop = replace_index
    map = np.array([(0.0, 0.0, 0.0, 1.0) for i in np.arange(n)])
    map[:, base_color] = np.linspace(cspace[0], cspace[1], n)
    map[start:stop] = replace_with
    return ListedColormap(map)


class DensityPlots:
    @classmethod
    def from_path(cls, mesh_file_path, df_counts: pd.DataFrame):
        return cls(SimpleMesh.from_path(mesh_file_path), df_counts)

    def __init__(
        self,
        mesh: SimpleMesh,
        df_data: pd.DataFrame,
        df_cmap: dict = None,
        time_resolution=0.4,
        slow_motion=None,
    ):
        self._mesh: SimpleMesh = mesh
        # data frame with count data.
        self.time_resolution = time_resolution
        self.df_data: pd.DataFrame = df_data.copy()

        _d = np.array(df_data.index.get_level_values("timeStep"))
        self.df_data["time"] = self.time_resolution * _d
        self.df_data = self.df_data.set_index("time", append=True)
        self.slow_motion_intervals = None
        t = Timer.create_and_start(
            "add slow down frames", label="__init__.DensityPlots"
        )
        if slow_motion is not None:
            self.slow_motion_intervals = self.__apply_slow_motion(slow_motion)
        t.stop()
        self.cmap_dict: dict = df_cmap if df_cmap is not None else {}

    def __apply_slow_motion(self, slow_motion_areas):
        """
        slow_motion_areas = [(t_start, t_stop, frame_multiplier), (...), ...]
        """
        self.df_data["subframe"] = np.full(
            (self.df_data.shape[0],), fill_value=0, dtype=int
        )
        slow_motion_intervals = []
        df_list = [self.df_data]
        for sm_area in slow_motion_areas:
            t_start, t_stop, frame_multiplier = sm_area
            slow_motion_intervals.append((t_start, t_stop))
            _block: pd.DataFrame = self.df_data.loc[
                (slice(None), slice(None), slice(t_start, t_stop)), :
            ].copy()
            for subframe_idx in range(1, frame_multiplier + 1, 1):
                _block["subframe"] = np.full(
                    (_block.shape[0],), fill_value=subframe_idx, dtype=int
                )
                df_list.append(_block.copy())

        self.df_data = pd.concat(df_list, axis=0, levels=["timeStep", "faceId", "time"])
        self.df_data = self.df_data.set_index("subframe", append=True)
        self.df_data = self.df_data.reorder_levels(
            ["timeStep", "time", "subframe", "faceId"]
        )
        self.df_data = self.df_data.sort_index()
        change_timeStep = np.append(
            0, np.diff(self.df_data.index.get_level_values("timeStep"))
        ).astype(np.bool)
        change_subFrame = np.append(
            0, np.diff(self.df_data.index.get_level_values("subframe"))
        ).astype(np.bool)
        id_mask = np.logical_or(change_timeStep, change_subFrame)
        frame_series = pd.Series(np.cumsum(id_mask), name="frame",)
        self.df_data = self.df_data.set_index(frame_series, append=True)
        self.df_data = self.df_data.droplevel("subframe")
        self.df_data = self.df_data.reorder_levels(
            ["frame", "timeStep", "time", "faceId"]
        )
        self.df_data = self.df_data.sort_index()
        return slow_motion_intervals

    def __tripcolor(
        self,
        ax,
        triang,
        density_or_counts,
        option: PlotOptions = PlotOptions.DENSITY,
        **kwargs,
    ):

        if option == PlotOptions.COUNT:
            ax, tpc = tripcolor_costum(
                ax, triang, facecolors=density_or_counts, **kwargs,
            )
            title_option = "Counts per triangle"
            label_option = "Counts [-]"
        elif option == PlotOptions.DENSITY:
            ax, tpc = tripcolor_costum(
                ax, triang, density_or_counts, shading="gouraud", **kwargs,
            )  # shading = 'gouraud' or 'fla'
            title_option = "Mapped density"
            label_option = "Density [#/m^2]"
        elif option == PlotOptions.DENSITY_SMOOTH:
            ax, tpc = tripcolor_costum(
                ax, triang, density_or_counts, shading="gouraud", **kwargs,
            )  # shading = 'gouraud' or 'fla'
            title_option = "Smoothed density"
            label_option = "Density [#/m^2]"
        else:
            raise ValueError(
                f"unknown option received got: {option} allowed: {PlotOptions}"
            )

        return ax, tpc, title_option, label_option

    def __data_for(self, frame, data=None):
        if data is None:
            data = list(self.df_data.columns)[0]
        return self.df_data.loc[frame, data].copy()

    def __cmap_for(self, data=None):
        if data is None:
            data = list(self.df_data.columns)[0]
        return self.cmap_dict.get(data, None)

    def add_cmap(self, key, cmap):
        if type(key) == str:
            if key in self.cmap_dict:
                self.cmap_dict[key] = cmap
            else:
                self.cmap_dict.setdefault(key, cmap)
        elif type(key) == list:
            for k in key:
                self.add_cmap(k, cmap)
        else:
            raise ValueError(
                f"expected string or list for key attribute got: {key}{type(key)}"
            )

    def __get_smoothed_mesh(self, time):

        x_, y_, triangles_ = self._mesh.get_xy_elements()
        counts = self.__data_for(time).ravel()

        matrix = self._mesh.mapping_matrices
        areas = self._mesh.nodal_area

        denominator = matrix.dot(areas)

        sum_counts = matrix.dot(counts)
        nodal_density = 2 * sum_counts / denominator

        vertices = np.array([x_, y_, nodal_density]).T

        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles_)
        xyz0 = np.array(mesh.vertices)

        mesh = trimesh.smoothing.filter_laplacian(mesh)

        elements = mesh.edges[:, 0]
        elements = np.reshape(elements, (-1, 3))

        xyz = np.array(mesh.vertices)

        triang = tri.Triangulation(xyz0[:, 0], xyz0[:, 1], elements)

        nodal_density_smooth = xyz[:, 2]

        return triang, nodal_density_smooth

    def __cache_data(self, data):
        t = Timer.create_and_start("build pool_frames", label="__cache_data")
        proc = 8
        pool = Pool(processes=proc)
        min_f = self.df_data.index.get_level_values("frame").min()
        max_f = self.df_data.index.get_level_values("frame").max()
        lower = np.linspace(min_f, max_f, num=proc + 1, dtype=int)[:-1]
        upper = np.append(np.subtract(lower, 1)[1:], max_f)
        pool_frames = np.concatenate((lower, upper)).reshape((-1, 2), order="F")

        ret = []
        t.stop_start("create cache")
        for d in data:
            ret.extend(pool.starmap(self.get_count, [(f, d) for f in pool_frames]))

        t.stop_start("concat pools")
        df = pd.concat(ret, axis=1)
        t.stop()
        return df

    def select(self, df, data, type, frame):
        s = pd.IndexSlice[data, type, frame]
        mask = df.loc[:, s].notna()
        return df.loc[mask, s]

    def get_count(self, f, data):
        # t = Timer.create_and_start("count", label="get_count")
        df = self.df_data.loc[(slice(*f)), :].copy()
        min_f = df.index.get_level_values("frame").min()
        max_f = df.index.get_level_values("frame").max()
        # print(f"{min_f}:{max_f}")

        index_ret = np.array([])
        data_ret = []
        for frame in range(min_f, max_f + 1):
            counts = df.loc[frame, data].copy()

            matrix = self._mesh.mapping_matrices
            areas = self._mesh.nodal_area
            denominator = matrix.dot(areas)
            sum_counts = matrix.dot(counts)
            nodal_density = sum_counts / denominator
            index_ret = np.append(
                index_ret,
                [
                    np.array([data, PlotOptions.COUNT.name, frame]),
                    np.array([data, PlotOptions.DENSITY.name, frame]),
                ],
            )
            data_ret.extend([counts.reset_index(drop=True), pd.Series(nodal_density)])
        # t.stop_start("contact")
        i_arr = index_ret.reshape((3, -1), order="F")
        index_ret = pd.MultiIndex.from_arrays(
            [i_arr[0], i_arr[1], i_arr[2].astype(int)]
        )
        df = pd.concat(data_ret, ignore_index=True, axis=1)
        df.columns = index_ret
        # t.stop()
        return df

    def __cached_plot_data(
        self, cache, frame, data, option: PlotOptions = PlotOptions.DENSITY
    ):

        triang = self._mesh.tri
        if option == PlotOptions.COUNT:
            density_or_counts = self.select(cache, data, option.name, frame)
        elif option == PlotOptions.DENSITY:
            density_or_counts = self.select(cache, data, option.name, frame)
        elif option == PlotOptions.DENSITY_SMOOTH:
            # new triangulation !
            triang, nodal_density_smooth = self.__get_smoothed_mesh(frame)
            density_or_counts = nodal_density_smooth
        else:
            raise ValueError(
                f"unknown option received got: {option} allowed: {PlotOptions}"
            )
        return triang, density_or_counts

    def __get_plot_attributes(
        self, frame, data, option: PlotOptions = PlotOptions.DENSITY
    ):

        counts = self.__data_for(frame, data).ravel()

        matrix = self._mesh.mapping_matrices
        areas = self._mesh.nodal_area
        denominator = matrix.dot(areas)
        sum_counts = matrix.dot(counts)
        nodal_density = sum_counts / denominator
        triang = self._mesh.tri

        if option == PlotOptions.COUNT:
            density_or_counts = counts
        elif option == PlotOptions.DENSITY:
            density_or_counts = nodal_density
        elif option == PlotOptions.DENSITY_SMOOTH:
            # new triangulation !
            triang, nodal_density_smooth = self.__get_smoothed_mesh(frame)
            density_or_counts = nodal_density_smooth
        else:
            raise ValueError(
                f"unknown option received got: {option} allowed: {PlotOptions}"
            )

        return triang, density_or_counts

    def time_for_frame(self, frame):
        time = (
            self.df_data.loc[(frame, slice(None), slice(None), slice(1)), :]
            .index.get_level_values("time")
            .to_list()[0]
        )
        return time

    def is_time_slowmotion(self, time):
        for t_interval in self.slow_motion_intervals:
            if t_interval[0] <= time < t_interval[1]:
                return True
        return False

    def animate_density(
        self,
        option,
        save_mp4_as,
        animate_time=(-1.0, -1.0),
        plot_data=(None,),
        color_bar_from=(0,),
        title=None,
        cbar_lbl=(None,),
        norm=1.0,
        min_density=0.0,
        max_density=1.5,
        multi_pool: Pool = None,
    ):
        """

        """
        vid = f"{os.path.basename(save_mp4_as)}.mp4"
        if multi_pool is not None:
            print(f"build animate_density {vid} async in pool: {multi_pool}")
            multi_pool.apply_async(
                self.animate_density,
                (option, save_mp4_as),
                dict(
                    animate_time=animate_time,
                    plot_data=plot_data,
                    color_bar_from=color_bar_from,
                    title=title,
                    cbar_lbl=cbar_lbl,
                    norm=norm,
                    min_density=min_density,
                    max_density=max_density,
                    multi_pool=None,
                ),
                error_callback=lambda e: print(
                    f"Error while build {vid} in async pool.\n>>{e}"
                ),
            )
            return

        t = Timer.create_and_start("create_cache", label="animate_density")
        start_t, end_t = animate_time
        # df_cached = self.__cache_data(plot_data)
        frames = (
            self.df_data.loc[
                (slice(None), slice(None), slice(start_t, end_t), slice(1)), :
            ]
            .index.get_level_values("frame")
            .to_list()
        )

        fig, ax = plt.subplots()
        if len(cbar_lbl) != len(color_bar_from):
            raise ValueError(
                f"plot_data and color_bar must be of same length. {color_bar_from} --- {cbar_lbl}"
            )

        def animate(i):
            frame = i
            time_t = self.time_for_frame(frame)
            print(f"{vid} >> frame:{frame} time:{time_t}")
            t = Timer.create_and_start("animate", label="animate")
            fig.clf()
            fig.tight_layout()
            ax = fig.gca()
            ax.set_facecolor((0.66, 0.66, 0.66))
            default_labels = []

            for data in plot_data:
                cmap = self.__cmap_for(data)

                t = Timer.create_and_start("get_data", label="animate")
                triang, density_or_counts = self.__get_plot_attributes(
                    frame, data, option
                )

                # triang, density_or_counts = self.__cached_plot_data(
                #     df_cached, frame, data, option
                # )
                density_or_counts = density_or_counts / norm
                t.stop()
                t = Timer.create_and_start("plot", label="animate")
                ax, tpc, default_title, default_label = self.__tripcolor(
                    ax,
                    triang,
                    density_or_counts,
                    cmap=cmap,
                    vmin=min_density,
                    vmax=max_density,
                    override_cmap_alpha=False,
                )
                ax.set_title(title)
                default_labels.append(default_title)
                t.stop()

            ax.set_aspect("equal")
            sim_sec = np.floor(time_t)
            sim_msec = time_t - sim_sec
            if sim_msec == 0.0:
                sim_t_str = f"{str(datetime.timedelta(seconds=sim_sec))}.000000"
            else:
                sim_t_str = f"{str(datetime.timedelta(seconds=sim_sec))}.{datetime.timedelta(seconds=sim_msec).microseconds}"
            if self.is_time_slowmotion(time_t):
                sim_t_str = f"{sim_t_str} slow!"
            ax.text(30, -35, f"Time: {sim_t_str}", fontsize=12)
            if title is not None:
                ax.set_title(title)
            for idx, lbl in zip(color_bar_from, cbar_lbl):
                # choose given label or default if none.
                _lbl = default_labels[idx] if lbl is None else lbl
                fig.colorbar(ax.collections[idx], ax=ax, label=_lbl)
            # t.stop()

        # anim = animation.FuncAnimation(fig, animate, init_func=init, frames=time_steps)
        anim = animation.FuncAnimation(fig, animate, frames=frames)
        save_mp4_as = save_mp4_as + ".mp4"

        t.stop_start("save video")
        anim.save(save_mp4_as, fps=24, extra_args=["-vcodec", "libx264"])
        t.stop()

    def plot_density(
        self,
        time,
        option,
        plot_data=(None,),  # define intput data key for df_data and df_cmap <---
        fig_path=None,
        title=None,
        min_density=0.0,
        max_density=1.5,
        norm=1.0,
    ):
        fig2, ax2 = plt.subplots()
        ax2.set_aspect("equal")

        for data in plot_data:
            cmap = self.__cmap_for(data)
            triang, density_or_counts = self.__get_plot_attributes(time, data, option)
            density_or_counts = density_or_counts / norm
            ax2, tpc, title_option, label_option = self.__tripcolor(
                ax2,
                triang,
                density_or_counts,
                cmap=cmap,
                vmin=min_density,
                vmax=max_density,
                override_cmap_alpha=False,
            )

        if title is not None:
            title_option = title

        fig2.colorbar(tpc, label=label_option)
        ax2.set_title(title_option)
        if fig_path is not None:
            fig2.savefig(fig_path)
        fig2.show()
        return fig2, ax2


class NumPedTimeSeries(PlotHelper):
    def build(self, df, title, c_start, c_end, c_count, is_raw_data=True):
        """
        creates time series plot of number of pedestrians in the simulation based on
        the 'endTime' and 'startTime' processors.

        returns axis with plot and copy of DataFrame if ret_data is true.
        """
        created_fig = False

        if is_raw_data:
            df_in = df.loc[:, c_count].groupby(df[c_start]).count()
            if type(df_in) == pd.Series:
                df_in = df_in.to_frame()
            df_in = df_in.rename({c_count: "in"}, axis=1)

            df_out = df.loc[:, c_count].groupby(df[c_end]).count()
            df_out = df_out.to_frame()
            df_out = df_out.rename({c_count: "out"}, axis=1)
            df_io = pd.merge(
                df_in, df_out, how="outer", left_index=True, right_index=True
            )
            df_io = df_io.fillna(0)
            df_io["in_cum"] = df_io["in"].cumsum()
            df_io["out_cum"] = df_io["out"].cumsum()
            df_io["diff_cum"] = df_io["in_cum"] - df_io["out_cum"]
        else:
            df_io = df

        self.ax.scatter(df_io.index, df_io["diff_cum"], marker=".", linewidths=0.15)
        self.ax.set_title(f"{title}")
        self.ax.set_ylabel("number of Peds")
        self.ax.set_xlabel("simulation time [s]")
        self._plot_data = df_io

        return self
