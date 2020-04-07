import os
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pandas as pd

import trimesh
from uitls.mesh import SimpleMesh
from uitls.plot_helper import PlotHelper
from multiprocessing import Pool, TimeoutError

sys.path.append(
    os.path.abspath("")
)  # in case tutorial is called from the root directory
sys.path.append(os.path.abspath(".."))  # in tutorial directly


class DensityPlots:

    @classmethod
    def from_path(cls, mesh_file_path, df_counts: pd.DataFrame):
        return cls(SimpleMesh.from_path(mesh_file_path), df_counts)

    def __init__(self, mesh: SimpleMesh, df_counts: pd.DataFrame):
        self._mesh: SimpleMesh = mesh
        # data frame with count data.
        self.df_counts: pd.DataFrame = df_counts

    def __read_counts(self, t):
        df = self.df_counts
        df_30 = df.loc[df["timeStep"] == t]

        counts = np.array(df_30.iloc[:, 2])

        return counts

    def __get_smoothed_mesh(self, time):

        x_, y_, triangles_ = self._mesh.get_xy_elements()
        counts = self.__read_counts(time).ravel()

        matrix = self._mesh.get_mapping_matrices()
        areas = self._mesh.get_nodal_areas()

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

    def __get_plot_attributes(self, time, option):

        x_, y_, triangles_ = self._mesh.get_xy_elements()
        counts = self.__read_counts(time).ravel()

        matrix = self._mesh.get_mapping_matrices()
        areas = self._mesh.get_nodal_areas()
        denominator = matrix.dot(areas)
        sum_counts = matrix.dot(counts)
        nodal_density = sum_counts / denominator
        triang = tri.Triangulation(x_, y_, triangles_)

        if option == "counts":
            density_or_counts = counts

        if option == "density":
            density_or_counts = nodal_density

        if option == "density_smooth":
            # new triangulation !
            triang, nodal_density_smooth = self.__get_smoothed_mesh(time)
            density_or_counts = nodal_density_smooth

        return triang, density_or_counts

    def animate_density(
        self, time_steps, option, save_mp4_as, title="title", label="xx", norm=1.0, min_density=0.0, max_density=1.5
    ):

        triang, density_or_counts = self.__get_plot_attributes(time_steps[0], option)
        density_or_counts = density_or_counts/norm
        fig = plt.figure()

        if option == "counts":
            ax = plt.tripcolor(
                triang,
                facecolors=density_or_counts,
                vmin=min_density,
                vmax=max_density,
            )
        else:
            ax = plt.tripcolor(
                triang,
                density_or_counts,
                shading="gouraud",
                vmin=min_density,
                vmax=max_density,
            )  # shading = 'gouraud' or 'fla'

        plt.gca().set_aspect("equal")

        def init():
            plt.clf()
            __, density_or_counts = self.__get_plot_attributes(time_steps[0], option)

            if option == "counts":
                ax = plt.tripcolor(
                    triang,
                    facecolors=density_or_counts,
                    vmin=min_density,
                    vmax=max_density,
                )
            else:
                ax = plt.tripcolor(
                    triang,
                    density_or_counts,
                    shading="gouraud",
                    vmin=min_density,
                    vmax=max_density,
                )  # shading = 'gouraud' or 'fla'
            plt.gca().set_aspect("equal")

        def animate(i):
            time_step = i
            print(f"Timestep {time_step}")
            plt.clf()
            __, density_or_counts = self.__get_plot_attributes(time_step, option)

            if option == "counts":
                ax = plt.tripcolor(
                    triang,
                    facecolors=density_or_counts,
                    vmin=min_density,
                    vmax=max_density,
                )
            else:
                ax = plt.tripcolor(
                    triang,
                    density_or_counts,
                    shading="gouraud",
                    vmin=min_density,
                    vmax=max_density,
                )  # shading = 'gouraud' or 'fla'
            plt.gca().set_aspect("equal")
            fig.colorbar(ax, label=label)
            plt.title(title)

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=time_steps)
        save_mp4_as = save_mp4_as + ".mp4"

        anim.save(save_mp4_as, fps=24, extra_args=["-vcodec", "libx264"])
        plt.show()

    def plot_density(
        self, time, option, fig_path=None, title=None, min_density=0.0, max_density=1.5, norm=1.0
    ):

        triang, density_or_counts = self.__get_plot_attributes(time, option)
        density_or_counts = density_or_counts/norm
        fig2, ax2 = plt.subplots()
        ax2.set_aspect("equal")

        tpc = None

        if option == "counts":
            tpc = ax2.tripcolor(
                triang, facecolors=density_or_counts, vmin=min_density, vmax=max_density
            )
            title_option = "Counts per triangle"
            label_option = "Counts [-]"

        if option == "density":
            tpc = ax2.tripcolor(
                triang,
                density_or_counts,
                shading="gouraud",
                vmin=min_density,
                vmax=max_density,
            )  # shading = 'gouraud' or 'fla'
            title_option = "Mapped density"
            label_option = "Density [#/m^2]"

        if option == "density_smooth":
            tpc = ax2.tripcolor(
                triang,
                density_or_counts,
                shading="gouraud",
                vmin=min_density,
                vmax=max_density,
            )  # shading = 'gouraud' or 'fla'
            title_option = "Smoothed density"
            label_option = "Density [#/m^2]"

        if title is not None:
            title_option = title

        fig2.colorbar(tpc, label=label_option)
        ax2.set_title(title_option)
        if fig_path is not None:
            fig2.savefig(fig_path)
        plt.show()
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
