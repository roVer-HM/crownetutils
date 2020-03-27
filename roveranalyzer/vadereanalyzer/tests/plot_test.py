import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.animation as animation
import time

import trimesh
import  matplotlib.tri as tri

import scipy.sparse as sp

import matplotlib.pyplot as plt
from drawnow import drawnow, figure
from roveranalyzer.uitls.path import PathHelper
from roveranalyzer.vadereanalyzer.plots import plots as plot
from roveranalyzer.vadereanalyzer.scenario_output import ScenarioOutput

sys.path.append(
    os.path.abspath("")
)  # in case tutorial is called from the root directory
sys.path.append(os.path.abspath(".."))  # in tutorial directly


###############

def test():
    pass

def xxx(t, df):

    df_30 = df.loc[df["timeStep"] == t, ("x", "y", "gridCount-PID8")]
    df_30 = df_30.pivot("y", "x", "gridCount-PID8")

    z = df.iloc[:,4]
    z_min = np.min(z)
    z_max = np.max(z)

    myheatmap(df_30, z_min, z_max)
    obstacles()
    divide_tiles()
    plt.axis('equal')

def divide_tiles():
    pass

def myheatmap(df_30, z_min, z_max):

    x = df_30.axes[0].values
    tile_size_x = x[1] - x[0]
    x = np.arange(x[0], (len(x) + 1) * tile_size_x, tile_size_x)

    y = df_30.axes[1].values
    tile_size_y = tile_size_x
    y = np.arange(y[0], (len(y) + 1) * tile_size_y, tile_size_y)

    z = df_30.T.values
    y, x = np.meshgrid(x, y)

    fig, ax = plt.subplots()
    c = ax.pcolormesh(x, y, z, cmap='seismic',vmin=z_min, vmax =z_max)
    ax.set_title('pcolormesh')
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)





def density():

    #p_helper = RelPath.from_user_home()
    #p = p_helper.join("/rover/simulations/simple_detoure/vadere/output/test" )
    p = "/home/christina/repos/rover-main/rover/simulations/simple_detoure/vadere/output/test/"
    output = ScenarioOutput.create_output_from_project_output(p)
    df = output.files["gridDensity.csv"]()
    fig = plt.figure(figsize=(7, 7 / 2))


    t_end=df['timeStep'].iloc[-1]
    for t in range(400, 500, 2):
        kwargs = {"t": float(t), "df": df}
        drawnow(xxx, show_once=True, confirm=False, stop_on_close=False, **kwargs)



def fig_num_peds_series():

    p_helper = RelPath.from_env("ROVER_MAIN")
    trajectories = p_helper.glob(
        "simulation-campaigns", "simpleDetour.sh-results_20200*_mia*/**/postvis.traj"
    )
    # trajectories = p_helper.glob('simulation-campaigns', 'simple_detour_100x177_long*/**/postvis.traj')
    output_dirs = [os.path.split(p)[0] for p in trajectories]
    outputs = [ScenarioOutput.create_output_from_project_output(p) for p in output_dirs]

    ratio = 16 / 9
    size_x_ax = 10
    size_y_ax = size_x_ax / ratio
    fig, axes = plt.subplots(
        len(outputs), 1, figsize=(size_x_ax, len(outputs) * size_y_ax)
    )

    for idx, o in enumerate(outputs):
        df = o.files["startEndtime.csv"]()
        ax = plot.num_pedestrians_time_series(
            df,
            axes[idx],
            c_start="startTime-PID7",
            c_end="endTime-PID5",
            c_count="pedestrianId",
            title=o.path("name"),
        )
        info_txt = (
            f"inter arrival times: \n"
            f"{o.path('scenario/topography/sources[*]/distributionParameters')}"
        )
        ax.text(0.75, 0.2, info_txt, ha="left", transform=ax.transAxes)

    return fig


def obstacles():
    tree = ET.parse('/home/christina/repos/rover-main/rover/simulations/simple_detoure/vadere/output/test/shapes.xml')
    root = tree.getroot()
    polygons = root.getchildren()

    shapes = list()
    c = 0

    for polygon in polygons:
        location = polygon.get('position').split(" ")
        location = np.array([ float(location[ind]) for ind in range(1,3) ])

        shape = polygon.get('shape').split(" ")
        shape = np.array( [ float(shape[edge]) for edge in range(2,len(shape)) ] )
        shape = np.reshape(shape, (-1,2))
        shape +=location
        shapes.append(shape)

    y_vals = np.array([shape.T[1] for shape in shapes ]).ravel()
    y_max = np.max(y_vals)

    for shape in shapes:
        shape.T[1] = y_max - shape.T[1]
        plt.gca().add_patch(plt.Polygon(shape, color='grey'))





def get_mapping_matrices(file_name):

    __,__, triangles_ = get_mesh(file_name)
    rows, cols = np.array([],dtype=int), np.array([],dtype=int)

    ind  = 0
    for triangle in triangles_:
        rows = np.append(rows,triangle)
        cols = np.append(cols,[ind,ind,ind])
        ind += 1

    data = np.ones((1,len(rows))).ravel()
    mapping_matrix = sp.coo_matrix((data, (rows.ravel(), cols.ravel())))

    return mapping_matrix



def get_nodal_areas(file_name):

    x_, y_, triangles_ = get_mesh(file_name)
    triang = tri.Triangulation(x_, y_, triangles_)

    vertices = np.array([x_, y_, 0 * x_]).T

    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles_)

    areas = mesh.area_faces

    return areas

def read_counts(count_names, t):
    p = "/home/christina/repos/rover-main/rover/simulations/simple_detoure/vadere/output/test_use_me/"
    output = ScenarioOutput.create_output_from_project_output(p)
    df = output.files["Density_trias2.csv"]()

    df_30 = df.loc[df["timeStep"] == t, ("faceId", "meshDensityCounting-PID13")]

    counts = np.array( df_30.iloc[:,1] )

    return counts




def get_mesh(file_names):

    v, v1 = np.array([]),np.array([])

    # currently 1 mesh

    #for file_name in file_names:
    with open(file_names) as file:
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
        v1 = np.append(v1, [vals[1],vals[2],vals[3]] )

    xy_ = v.reshape((-1, 2)).T
    x = xy_[0]
    y = xy_[1]
    elements_ = v1.reshape((-1, 3)).astype(int) - 1

    return x,y,elements_

# initialization function: plot the background of each frame

def animate_heat_map():

    i = 1

    directory = "/home/christina/repos/rover-main/rover/simulations/simple_detoure/vadere/output/test_use_me"
    file_names = os.path.join(directory, 'Mesh_trias2.txt')
    count_names = os.path.join(directory, 'Density_trias2.csv')

    x_, y_, triangles_ = get_mesh(file_names)
    counts = read_counts(count_names, i).ravel()

    matrix = get_mapping_matrices(file_names)
    areas = get_nodal_areas(file_names)

    denominator = matrix.dot(areas)

    sum_counts = matrix.dot(counts)
    nodal_density = sum_counts / denominator
    triang = tri.Triangulation(x_, y_, triangles_)

    time_steps = np.arange(100, 400, 1)

    fig = plt.figure()
    ax = plt.tripcolor(triang, nodal_density, shading='gouraud')  # shading = 'gouraud' or 'fla'
    plt.gca().set_aspect('equal')


    def init():
        plt.clf()
        ax = plt.tripcolor(triang, nodal_density, shading='gouraud')
        plt.gca().set_aspect('equal')

    def animate(i):
        plt.clf()
        print(f"Timestep {i}")
        #
        counts = read_counts(count_names,i).ravel()
        sum_counts = matrix.dot(counts)
        nodal_density = sum_counts / denominator

        ##
        ax = plt.tripcolor(triang, nodal_density, shading='gouraud')
        plt.gca().set_aspect('equal')


    anim = animation.FuncAnimation(fig, animate, init_func=init, frames = time_steps)
    anim.save('my_animation_15.mp4', fps=24, extra_args=['-vcodec', 'libx264'])
    plt.show()











if __name__ == "__main__":
    os.environ['ROVER_MAIN'] = '/home/christina/repos/rover-main'

    directory = "/home/christina/repos/rover-main/rover/simulations/simple_detoure/vadere/output/test_use_me"
    file_names = os.path.join(directory,'Mesh_trias2.txt')
    count_names = os.path.join(directory,'Density_trias2.csv')


    x_, y_, triangles_ = get_mesh(file_names)
    counts = read_counts(count_names, 500).ravel()


    fig3, ax3 = plt.subplots()
    ax3.set_aspect('equal')
    tpc = ax3.tripcolor(x_, y_, triangles_, facecolors=counts)
    fig3.colorbar(tpc)
    ax3.set_title('Number of pedestrians in a triangle (elemental values)')

    plt.show()

    matrix = get_mapping_matrices(file_names)
    areas = get_nodal_areas(file_names)

    denominator =  matrix.dot(areas)


    sum_counts = matrix.dot(counts)
    nodal_density = sum_counts / denominator
    triang = tri.Triangulation(x_, y_, triangles_)


    fig2, ax2 = plt.subplots()
    ax2.set_aspect('equal')
    tpc = ax2.tripcolor(triang, nodal_density, shading='gouraud') # shading = 'gouraud' or 'fla'
    fig2.colorbar(tpc)
    ax2.set_title('Mapped density (nodal values)')

    #plt.show()

    animate_heat_map()

    plt.show()

    print("finished")
