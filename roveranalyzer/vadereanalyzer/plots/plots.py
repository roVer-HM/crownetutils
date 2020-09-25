import datetime
import os
import sys
from enum import Enum
from multiprocessing import Pool

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pandas as pd
import trimesh
from matplotlib.colors import ListedColormap

from roveranalyzer.uitls import Timer
from roveranalyzer.uitls.mesh import SimpleMesh
from roveranalyzer.uitls.plot_helper import PlotHelper
from roveranalyzer.vadereanalyzer.plots.custom_tripcolor import \
    tripcolor_costum

sys.path.append(
    os.path.abspath("")
)  # in case tutorial is called from the root directory
sys.path.append(os.path.abspath(".."))  # in tutorial directly


class PlotOptions(Enum):
    COUNT = (1, "counts")
    DENSITY = (2, "density")
    DENSITY_SMOOTH = (3, "density_smooth")


def t_cmap(
    cmap_name, replace_index=(0, 1, 1.0), use_colors=(0, 256), zero_color=None,
):
    cmap = plt.get_cmap(name=cmap_name)
    start, stop, alpha = replace_index
    colors = np.array(cmap(np.linspace(0, cmap.N, dtype=int)))
    colors = colors[use_colors[0] : use_colors[1]]
    colors[start:stop, -1] = alpha
    if zero_color is not None:
        colors[0] = zero_color
    return ListedColormap(colors)


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

    @classmethod
    def from_mesh_processor(
        cls,
        vadere_output,
        mesh_out_file,
        density_out_file,
        cmap_dict: dict,
        data_cols_rename: list,
    ):
        if not all([item in cmap_dict for item in data_cols_rename]):
            raise ValueError("cmap keys are not in data_cols")
        mesh_str = vadere_output.files[mesh_out_file].as_string(remove_meta=True)
        _mesh = SimpleMesh.from_string(mesh_str)
        _df = vadere_output.files[density_out_file].df(
            set_index=True, column_names=data_cols_rename,
        )
        return cls(_mesh, _df, cmap_dict)

    def __init__(
        self,
        mesh: SimpleMesh,
        df_data: pd.DataFrame,
        df_cmap: dict = None,
        time_resolution=0.4,
    ):
        self._mesh: SimpleMesh = mesh
        # data frame with count data.
        self.time_resolution = time_resolution
        self.df_data: pd.DataFrame = df_data.copy()

        _d = np.array(df_data.index.get_level_values("timeStep"))
        self.df_data["time"] = self.time_resolution * _d
        self.df_data = self.df_data.set_index("time", append=True)
        self.slow_motion_intervals = []
        frame_key = pd.Series(
            np.ones(self.df_data.shape[0], dtype=int), name="num_of_frames"
        )
        self.df_data = self.df_data.set_index(frame_key, append=True)
        self.cmap_dict: dict = df_cmap if df_cmap is not None else {}

    def set_slow_motion_intervals(self, slow_motion_intervals):
        self.slow_motion_intervals.extend(slow_motion_intervals)
        times = self.df_data.index.get_level_values("time").to_numpy()
        frames = self.df_data.index.get_level_values("num_of_frames").to_numpy(
            copy=True, dtype=int
        )
        for sm_area in slow_motion_intervals:
            t_start, t_stop, frame_multiplier = sm_area
            mask = (times >= t_start) & (times <= t_stop)
            frames[mask] = frame_multiplier

        frame_key = pd.Series(frames, name="num_of_frames")
        self.df_data = self.df_data.reset_index(level="num_of_frames", drop=True)
        self.df_data = self.df_data.set_index(frame_key, append=True)

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

    def __data_for(self, time_step, data=None):
        if data is None:
            data = list(self.df_data.columns)[0]
        return self.df_data.loc[time_step, data].copy()

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

    def __get_smoothed_mesh(self, time_step, data):

        x_, y_, triangles_ = self._mesh.get_xy_elements()
        counts = self.__data_for(time_step, data).ravel()

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

    def get_count(self, f, data):
        # t = Timer.create_and_start("count", label="get_count")
        df = self.df_data.loc[(slice(*f)), :].copy()
        min_f = df.index.get_level_values("frame").min()
        max_f = df.index.get_level_values("frame").max()

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
            triang, nodal_density_smooth = self.__get_smoothed_mesh(frame, data)
            density_or_counts = nodal_density_smooth
        else:
            raise ValueError(
                f"unknown option received got: {option} allowed: {PlotOptions}"
            )

        return triang, density_or_counts

    def frame_for_time(self, time):
        frame = (
            self.df_data.loc[(slice(None), slice(1), time), :]
            .index.get_level_values("timeStep")
            .to_list()[0]
        )
        return frame

    def time_for_frame(self, frame):
        time = (
            self.df_data.loc[(frame, slice(1), slice(None)), :]
            .index.get_level_values("time")
            .to_list()[0]
        )
        return time

    def is_time_slowmotion(self, time):
        if self.slow_motion_intervals is None:
            return False
        for t_interval in self.slow_motion_intervals:
            start, end, multiplier = t_interval
            if start <= time < end:
                return True
        return False

    def get_frame_multiplier(self, time):
        if self.slow_motion_intervals is None:
            return 1
        for t_interval in self.slow_motion_intervals:
            start, end, multiplier = t_interval
            if start <= time < end:
                return multiplier
        return 1

    def animate_density(
        self,
        option,
        save_mp4_as,
        animate_time=(-1.0, -1.0),
        plot_data=(None,),
        color_bar_from=(0,),
        title=None,
        cbar_lbl=(None,),
        norm=(1.0,),
        min_density=0.0,
        max_density=1.5,
        frame_rate=24,
        multi_pool: Pool = None,
    ):
        vid = f"{os.path.basename(save_mp4_as)}"
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
                    frame_rate=frame_rate,
                    multi_pool=None,
                ),
                error_callback=lambda e: print(
                    f"Error while build {vid} in async pool.\n>>{e}"
                ),
            )
            return

        t = Timer.create_and_start("create_cache", label="animate_density")
        start_t, end_t = animate_time
        frames = (
            self.df_data.loc[(slice(None), slice(1), slice(start_t, end_t)), :]
            .index.get_level_values("timeStep")
            .to_list()
        )
        frame_repates = (
            self.df_data.loc[
                (slice(None), slice(1), slice(start_t, end_t), slice(None)), :
            ]
            .index.get_level_values("num_of_frames")
            .to_list()
        )
        frames = np.repeat(frames, frame_repates)

        fig, ax = plt.subplots(figsize=(16, 9))
        if len(cbar_lbl) != len(color_bar_from):
            raise ValueError(
                f"plot_data and color_bar must be of same length. {color_bar_from} --- {cbar_lbl}"
            )

        def build(frame):
            fig.clf()
            _ax = fig.gca()

            default_labels = []
            time_t = self.time_for_frame(frame)
            # Build first plot. This is updated in #aniamte()
            for data in plot_data:
                cmap = self.__cmap_for(data)
                triang, density_or_counts = self.__get_plot_attributes(
                    frame, data, option
                )
                _ax, tpc, default_title, default_label = self.__tripcolor(
                    _ax,
                    triang,
                    density_or_counts,
                    cmap=cmap,
                    vmin=min_density,
                    vmax=max_density,
                    override_cmap_alpha=False,
                )
                default_labels.append(default_title)
            _ax.set_facecolor((0.66, 0.66, 0.66))
            _ax.set_title(title)
            _ax.set_aspect("equal")
            sim_sec = np.floor(time_t)
            sim_msec = time_t - sim_sec
            slow_motion_txt = "Send Entrance Closed"
            txt_info = _ax.text(30, -25, "", fontsize=12)
            if sim_msec == 0.0:
                sim_t_str = f"{str(datetime.timedelta(seconds=sim_sec))}.000000"
                txt_info.set_text("")
            else:
                sim_t_str = f"{str(datetime.timedelta(seconds=sim_sec))}.{datetime.timedelta(seconds=sim_msec).microseconds}"
                txt_info.set_text(slow_motion_txt)
            if self.get_frame_multiplier(time_t) > 1:
                sim_t_str = f"{sim_t_str} slow motion!"
            txt_time = _ax.text(30, -20, f"Time: {sim_t_str}", fontsize=12)
            if title is not None:
                _ax.set_title(title)
            for idx, lbl in zip(color_bar_from, cbar_lbl):
                # choose given label or default if none.
                _lbl = default_labels[idx] if lbl is None else lbl
                fig.colorbar(_ax.collections[idx], ax=_ax, label=_lbl)
            return txt_info, txt_time

        txt_info, txt_time = build(frames[0])
        curr_time = -1.0
        curr_f_count = 1

        def set_ax_text(txt1, txt2, time_t):
            _sim_s = np.floor(time_t)
            _sim_ms = time_t - _sim_s
            if _sim_ms == 0.0:
                _sim_t_str = f"{str(datetime.timedelta(seconds=_sim_s))}.000000"
            else:
                _sim_t_str = f"{str(datetime.timedelta(seconds=_sim_s))}.{datetime.timedelta(seconds=_sim_ms).microseconds}"
            slow_motion_txt = ""
            if self.get_frame_multiplier(time_t) > 1:
                _sim_t_str = f"{_sim_t_str} slow motion!"
                slow_motion_txt = "Send Entrance Closed"
            txt1.set_text(f"Time: {_sim_t_str}")
            txt2.set_text(slow_motion_txt)

        def pre_animate(frame):
            _ax = fig.gca()
            f_time = self.time_for_frame(frame)
            set_ax_text(txt_time, txt_info, f_time)

            print(
                f"\r{vid} >> {100 * ((frames[0] - frame) / (frames[0] - frames[-1])):02.1f}% frame:{frame} time:{f_time:.3f}",
                end="",
            )
            return _ax, f_time

        def animate_smooth(frame):
            pre_animate(frame)
            build(frame)

        def animate_count_density(frame):
            _ax, f_time = pre_animate(frame)
            nonlocal curr_time
            nonlocal curr_f_count

            if curr_time > 0 and curr_f_count < self.get_frame_multiplier(f_time):
                # do nothing to axes because we are in slow motion interval and the
                # previous frame is repeated.
                curr_f_count += 1
                return
            else:
                if self.is_time_slowmotion(f_time):
                    curr_time = f_time
                else:
                    curr_time = -1.0
                for idx, data in enumerate(plot_data):
                    # t = Timer.create_and_start("get_data", label="animate")
                    _, density_or_counts = self.__get_plot_attributes(
                        frame, data, option
                    )
                    density_or_counts = density_or_counts / norm[idx]
                    # t.stop()
                    # t = Timer.create_and_start("plot", label="animate")
                    _ax.collections[idx].update_data(density_or_counts)
                    # t.stop()

        if option == PlotOptions.DENSITY_SMOOTH:
            anim = animation.FuncAnimation(fig, animate_smooth, frames=frames)
        else:
            anim = animation.FuncAnimation(fig, animate_count_density, frames=frames)
        save_mp4_as = save_mp4_as

        t.stop_start("save video")
        anim.save(save_mp4_as, fps=24, extra_args=["-vcodec", "libx264"])
        t.stop()

    def plot_density(
        self,
        time,
        option,
        fig_path,
        plot_data=(None,),
        color_bar_from=(0,),
        title=None,
        cbar_lbl=(None,),
        norm=(1.0,),
        min_density=0.0,
        max_density=1.5,
        print_time=True,
    ):
        fig, _ax = plt.subplots(figsize=(16, 9))
        _ax.set_facecolor((0.66, 0.66, 0.66))
        _ax.set_title(title)
        _ax.set_aspect("equal")
        sim_sec = np.floor(time)
        sim_msec = time - sim_sec
        frame = self.frame_for_time(time)
        default_labels = []
        for idx, data in enumerate(plot_data):
            cmap = self.__cmap_for(data)
            triang, density_or_counts = self.__get_plot_attributes(frame, data, option)
            density_or_counts = density_or_counts / norm[idx]
            _ax, tpc, default_title, default_label = self.__tripcolor(
                _ax,
                triang,
                density_or_counts,
                cmap=cmap,
                vmin=min_density,
                vmax=max_density,
                override_cmap_alpha=False,
            )
            default_labels.append(default_title)
        if sim_msec == 0.0:
            sim_t_str = f"{str(datetime.timedelta(seconds=sim_sec))}.000000"
        else:
            sim_t_str = f"{str(datetime.timedelta(seconds=sim_sec))}.{datetime.timedelta(seconds=sim_msec).microseconds}"
        if print_time:
            _ax.text(30, -20, f"Time: {sim_t_str}", fontsize=12)
        for idx, lbl in zip(color_bar_from, cbar_lbl):
            # choose given label or default if none.
            _lbl = default_labels[idx] if lbl is None else lbl
            fig.colorbar(_ax.collections[idx], ax=_ax, label=_lbl)

        if fig_path is not None:
            fig.savefig(fig_path)
        fig.show()
        return fig, _ax


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
