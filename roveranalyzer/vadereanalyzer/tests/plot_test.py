import os
import sys
import numpy as np
import matplotlib.animation as animation

import trimesh
import matplotlib.tri as tri

import scipy.sparse as sp

import matplotlib.pyplot as plt
from roveranalyzer.vadereanalyzer.scenario_output import ScenarioOutput


sys.path.append(
    os.path.abspath("")
)  # in case tutorial is called from the root directory
sys.path.append(os.path.abspath(".."))  # in tutorial directly

class DensityPlots:

    def __init__(self, directory, file_names, count_names):
        self.directory = directory
        self.file_names = file_names
        self.count_names = count_names

    def __get_mapping_matrices(self):

        __, __, triangles_ = self.__get_mesh()
        rows, cols = np.array([], dtype=int), np.array([], dtype=int)

        ind = 0
        for triangle in triangles_:
            rows = np.append(rows, triangle)
            cols = np.append(cols, [ind, ind, ind])
            ind += 1

        data = np.ones((1, len(rows))).ravel()
        mapping_matrix = sp.coo_matrix((data, (rows.ravel(), cols.ravel())))

        return mapping_matrix

    def __get_nodal_areas(self):

        x_, y_, triangles_ = self.__get_mesh()
        triang = tri.Triangulation(x_, y_, triangles_)

        # vertices = np.array([x_, y_, 0 * x_]).T
        # mesh = trimesh.Trimesh(vertices=vertices, faces=triangles_)
        # areas = mesh.area_faces

        areas = []

        for triangle in triangles_:
            v0 = triangle[0]
            v1 = triangle[1]
            v2 = triangle[2]

            v0v1 = ([x_[v1] - x_[v0], y_[v1] - y_[v0]])
            v0v2 = ([x_[v2] - x_[v0], y_[v2] - y_[v0]])

            area = 0.5 * np.linalg.norm(np.cross(v0v1, v0v2))
            areas.append(area)

        return areas

    def __read_counts(self, t):

        count_names = self.count_names

        p = os.path.dirname(count_names)
        output = ScenarioOutput.create_output_from_project_output(p)

        file = os.path.basename(count_names)
        df = output.files[file]()

        df_30 = df.loc[df["timeStep"] == t]

        counts = np.array(df_30.iloc[:, 2])

        return counts

    def __get_smoothed_mesh(self, time):

        x_, y_, triangles_ = self.__get_mesh()
        counts = self.__read_counts(time).ravel()

        matrix = self.__get_mapping_matrices()
        areas = self.__get_nodal_areas()

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

    def __get_mesh(self):

        v, v1 = np.array([]), np.array([])

        with open(self.file_names) as file:
            text = file.read().split('#')

        xy = text[1].splitlines()
        xy = xy[2:]
        for xy_ in xy:
            vals = np.fromstring(xy_, dtype=float, sep=" ")
            v = np.append(v, [vals[3], vals[4]])

        elements = text[4].splitlines()
        elements = elements[1:]

        for ele_ in elements:
            vals = np.fromstring(ele_, dtype=int, sep=" ")
            v1 = np.append(v1, [vals[1], vals[2], vals[3]])

        xy_ = v.reshape((-1, 2)).T
        x = xy_[0]
        y = xy_[1]
        elements_ = v1.reshape((-1, 3)).astype(int) - 1

        return x, y, elements_

    def __get_plot_attributes(self, time, option):

        x_, y_, triangles_ = self.__get_mesh()
        counts = self.__read_counts(time).ravel()

        matrix = self.__get_mapping_matrices()
        areas = self.__get_nodal_areas()
        denominator = matrix.dot(areas)
        sum_counts = matrix.dot(counts)
        nodal_density = sum_counts / denominator
        triang = tri.Triangulation(x_, y_, triangles_)

        if option == 'counts':
            density_or_counts = counts

        if option == 'density':
            density_or_counts = nodal_density

        if option == "density_smooth":
            # new triangulation !
            triang, nodal_density_smooth = self.__get_smoothed_mesh(time)
            density_or_counts = nodal_density_smooth

        return triang, density_or_counts

    def animate_density(self, time_steps, option, save_mp4_as):

        triang, density_or_counts = self.__get_plot_attributes(time_steps[0], option)
        fig = plt.figure()

        if option == 'counts':
            ax = plt.tripcolor(triang, facecolors=density_or_counts)
        else:
            ax = plt.tripcolor(triang, density_or_counts, shading='gouraud')  # shading = 'gouraud' or 'fla'

        plt.gca().set_aspect('equal')

        def init():
            plt.clf()
            __, density_or_counts = self.__get_plot_attributes( time_steps[0], option)

            if option == 'counts':
                ax = plt.tripcolor(triang, facecolors=density_or_counts)
            else:
                ax = plt.tripcolor(triang, density_or_counts, shading='gouraud')  # shading = 'gouraud' or 'fla'
            plt.gca().set_aspect('equal')

        def animate(i):
            time_step = i
            print(f"Timestep {time_step}")
            plt.clf()
            __, density_or_counts = self.__get_plot_attributes(time_step, option)

            if option == 'counts':
                ax = plt.tripcolor(triang, facecolors=density_or_counts)
            else:
                ax = plt.tripcolor(triang, density_or_counts, shading='gouraud')  # shading = 'gouraud' or 'fla'
            plt.gca().set_aspect('equal')

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=time_steps)
        save_mp4_as = save_mp4_as + '.mp4'

        anim.save(save_mp4_as, fps=24, extra_args=['-vcodec', 'libx264'])
        plt.show()

    def plot_density(self, time, option):

        triang, density_or_counts = self.__get_plot_attributes(time, option)
        fig2, ax2 = plt.subplots()
        ax2.set_aspect('equal')

        tpc = None

        if option == 'counts':
            tpc = ax2.tripcolor(triang, facecolors=density_or_counts)
            title_option = 'Counts per triangle'
            label_option = "Counts [-]"

        if option == 'density':
            tpc = ax2.tripcolor(triang, density_or_counts, shading='gouraud')  # shading = 'gouraud' or 'fla'
            title_option = 'Mapped density'
            label_option = "Density [#/m^2]"

        if option == "density_smooth":
            tpc = ax2.tripcolor(triang, density_or_counts, shading='gouraud')  # shading = 'gouraud' or 'fla'
            title_option = 'Smoothed density'
            label_option = "Density [#/m^2]"

        fig2.colorbar(tpc, label=label_option)
        ax2.set_title(title_option)
        plt.show()


if __name__ == "__main__":

    directory = "testData/simple_detour" # directory where following files are stored
    file_names = os.path.join(directory, 'Mesh_trias1.txt') # from vadere data processor: MeshProcessor
    count_names = os.path.join(directory, 'Density_trias1.csv')   # count_names = from vadere data processor: MeshDensityCountingProcessor

    # plot (smoothed) density or counts for one frame /one timestep
    frame = 200 # time = frame* 0.4
    densityPlots = DensityPlots(directory,file_names,count_names)
    densityPlots.plot_density(frame,'counts')
    densityPlots.plot_density(frame, 'density')
    densityPlots.plot_density(frame, 'density_smooth')

    # plot (smoothed) density or counts for time series
    frames = np.arange(390, 400, 1)  # frames, time = 0.4* frame, min = 1!
    densityPlots.animate_density(frames,'counts','counts_movie')
    densityPlots.animate_density(frames, 'density', 'density_movie')
    densityPlots.animate_density(frames, 'density_smooth', 'density_smooth_movie')



