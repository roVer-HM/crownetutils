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


class DensityPlots:
    @classmethod
    def from_path(cls, mesh_file_path, df_counts: pd.DataFrame):
        return cls(SimpleMesh.from_path(mesh_file_path), df_counts)

    def __init__(
        self, mesh: SimpleMesh, df_data: pd.DataFrame, df_cmap: dict = None,
    ):
        self._mesh: SimpleMesh = mesh
        # data frame with count data.
        self.df_data: pd.DataFrame = df_data
        self.cmap_dict: dict = df_cmap if df_cmap is not None else {}

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

    def __data_for(self, time, data=None):
        if data is None:
            data = list(self.df_data.columns)[0]
        return self.df_data.loc[time, data]

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

    def __get_plot_attributes(
        self, time, data, option: PlotOptions = PlotOptions.DENSITY
    ):

        counts = self.__data_for(time, data).ravel()

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
            triang, nodal_density_smooth = self.__get_smoothed_mesh(time)
            density_or_counts = nodal_density_smooth
        else:
            raise ValueError(
                f"unknown option received got: {option} allowed: {PlotOptions}"
            )

        return triang, density_or_counts

    def animate_density(
        self,
        time_steps,
        option,
        save_mp4_as,
        plot_data=(None,),
        color_bar_from=(0,),
        title=None,
        cbar_lbl=(None,),
        norm=1.0,
        min_density=0.0,
        max_density=1.5,
    ):
        fig, ax = plt.subplots()
        if len(cbar_lbl) != len(color_bar_from):
            raise ValueError(f"plot_data, color_bar ")

        def animate(i):
            time_step = i
            print(f"Timestep {time_step}")
            fig.clf()
            ax = fig.gca()
            default_labels = []

            for data in plot_data:
                triang, density_or_counts = self.__get_plot_attributes(
                    time_step, data, option
                )
                density_or_counts = density_or_counts / norm
                ax, tpc, default_title, default_label = self.__tripcolor(
                    ax,
                    triang,
                    density_or_counts,
                    vmin=min_density,
                    vmax=max_density,
                    override_cmap_alpha=False,
                )
                ax.set_title(title)
                default_labels.append(default_title)

            ax.set_aspect("equal")
            if title is not None:
                ax.set_title(title)
            for idx, lbl in zip(color_bar_from, cbar_lbl):
                # choose given label or default if none.
                _lbl = default_labels[idx] if lbl is None else lbl
                fig.colorbar(ax.collections[idx], ax=ax, label=_lbl)

        # anim = animation.FuncAnimation(fig, animate, init_func=init, frames=time_steps)
        anim = animation.FuncAnimation(fig, animate, frames=time_steps)
        save_mp4_as = save_mp4_as + ".mp4"

        anim.save(save_mp4_as, fps=24, extra_args=["-vcodec", "libx264"])
        # plt.show()

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
