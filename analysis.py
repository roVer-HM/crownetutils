import os

import matplotlib
from matplotlib.lines import Line2D

# from roveranalyzer.simulators.rover.dcd.interactive import Interactive2DDensityPlot, InteractiveDelayOverDistance

matplotlib.use('TkAgg')

import matplotlib.animation as animation

from roveranalyzer.simulators.opp.opp_analysis import Opp, OppAccessor
from roveranalyzer.simulators.opp.utils import ScaveTool
from roveranalyzer.utils import PathHelper, from_pickle
from roveranalyzer.simulators.rover.dcd.dcd_map import DcdMap2DMulti
from itertools import product
import matplotlib.pyplot as plt
from roveranalyzer.utils import check_ax
import pandas as pd
import numpy as np

ROOT = "/home/max/Git/crownet/analysis/roveranalyzer/data/"
# RUN = "test"  # 7s 120 (vec) [only selected]
# RUN = "vadereBase_20210224"
RUN = "sumoBase_20210228"

PATH_ROOT = f"{ROOT}/{RUN}"
p = PathHelper(PATH_ROOT)

# @from_pickle(path=PATH_ROOT + "/analysis.p")
def read_data():
    global_map_path = p.glob(
        "global.csv", recursive=False, expect=1
    )
    node_map_paths = p.glob("dcdMap_*.csv")
    # scenario_path = p.glob("vadere.d/*.scenario", expect=1)

    dcd = DcdMap2DMulti.from_paths(
        global_data=global_map_path,
        node_data=node_map_paths,
        real_coords=True,
    )

    return dcd


# @from_pickle(path=PATH_ROOT + "/paramters.p")
def read_param():
    scave_tool = ScaveTool()
    SCA = f"{ROOT}/{RUN}/vars_rep_0.sca"
    # scave_filter = scave_tool.filter_builder()\
    #     .t_parameter()\
    #     .AND()\
    #     .module("**nTable")\
    #     .AND().name("maxAge")\
    #     .build()
    scave_filter = scave_tool.filter_builder().t_parameter().build()
    df_parameters = scave_tool.read_parameters(SCA, scave_filter=scave_filter)
    return df_parameters


# @from_pickle(path=PATH_ROOT + "/rcvdPkLifetimeVec.p")
def read_app_data():
    inputs = f"{ROOT}/{RUN}/vars_rep_0.vec"
    scave = ScaveTool()

    """
    scave_f = scave.filter_builder() \
        .gOpen().module("*World.node[1].lteNic.macd.densityMapApp").OR().module("*.node[*].aid.beaconApp").gClose() \
        .AND().name("rcvdPkLifetime:vector")
    """
    scave_f = scave.filter_builder() \
        .gOpen().module("*World.node[*].lteNic.mac").OR().module("*.node[*].lteNic.mac").gClose() \
        .AND().name("sentPacketToUpperLayer:vector*")

    _df = scave.load_df_from_scave(input_paths=inputs, scave_filter=scave_f)
    return _df


# def foo():
#     if make_interactive:
#         return (
#             (
#                 f,
#                 ax,
#                 InteractiveDensityPlot(
#                     dcd=self,
#                     data=pd.DataFrame(),
#                     ax=ax,
#                     time_step=time_step,
#                     node_id=node_id,
#                 ),
#             ),
#             ax,
#         )
#     else:

def analyse_interactive():
    # read data from raw input or access pickled object
    dcd = read_data()
    print(dcd.map.head())
    time = 2
    id = 0
    fig, ax = dcd.plot_density_map(time_step=time, node_id=id,
                                 pcolormesh_dic=dict(vmin=0, vmax=4))

    # fig, ax = dcd.plot_delay_over_distance(8, 3, "measurement_age")

    # dcd.update_delay_over_distance(12, 3, "measurement_age", ax.collections[0])
    i = Interactive2DDensityPlot(dcd, ax)
    # i = InteractiveDelayOverDistance(dcd, ax)
    i.show()

def make_density_plot(dcd):
    # make density_map_plots
    time = [140]
    ids = [0]
    for time, id in list(product(time, ids)):
        f, ax = dcd.plot_density_map(time_step=time, node_id=id, make_interactive=False,
                                     pcolormesh_dic=dict(vmin=0, vmax=4))
        print(f"create out/density_map_{id}_t{time}.png")
        ax.set_title("")
        f.savefig(p.join(f"out/density_map_{id}_t{time}.png"))
        plt.close(f)


def make_count_plot(dcd, para):
    # make count plot
    f1, ax = dcd.plot_count_diff()
    maxAge = para.loc[para["name"] == "maxAge", ["value"]].iloc[0].value
    title = f"{ax.title.get_text()} with neighborhood table maxAge {maxAge}"
    ax.set_title(title)
    os.makedirs(p.join("out"), exist_ok=True)
    out_p = p.join("out/count.png")
    f1.savefig(out_p)

def make_count_plot(delay):
    df_all = delay.opp.filter().vector().normalize_vectors(axis=0)

    f2, ax2 = check_ax()

    # plots = [["beacon", df_beacon], ["map", df_dMap], ["all", df_all]]:
    plots = [["packet delay", df_all]]
    time_per_bin = 1.0  # seconds
    for n, df in plots:
        bins = int(np.floor(df["time"].max() / time_per_bin))
        df = df.groupby(pd.cut(df["time"], bins)).mean()
        df = df.dropna()
        ax2.plot("time", "value", data=df, label=n)

    ax2.set_title("rcvdPkLifetime (delay) of all packets (beacon + map)")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel(f"mean delay (window size: {time_per_bin}s )[s]")
    ax2.legend()
    f2.savefig(p.join("out/delay1.png"))


def make_delay_plot(dcd, para, delay):

    # make count plot
    """
    f1, ax = dcd.plot_count_diff()
    maxAge = para.loc[para["name"] == "maxAge", ["value"]].iloc[0].value
    title = f"{ax.title.get_text()} with neighborhood table maxAge {maxAge}"
    ax.set_title(title)
    """

    # delay plot
    #
    # df_dMap = delay.opp.filter().vector().module_regex(".*densityMapApp.*").normalize_vectors(axis=0)
    # df_beacon = delay.opp.filter().vector().module_regex(".*beaconApp.*").normalize_vectors(axis=0)
    df_all = delay.opp.filter().vector().normalize_vectors(axis=0)

    f2, ax2 = check_ax()

    # plots = [["beacon", df_beacon], ["map", df_dMap], ["all", df_all]]:
    plots = [["packet delay", df_all]]
    time_per_bin = 1.0  # seconds
    for n, df in plots:
        bins = int(np.floor(df["time"].max() / time_per_bin))
        df = df.groupby(pd.cut(df["time"], bins)).mean()
        df = df.dropna()
        ax2.plot("time", "value", data=df, label=n)

    ax2.set_title("rcvdPkLifetime (delay) of all packets (beacon + map)")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel(f"mean delay (window size: {time_per_bin}s )[s]")
    ax2.legend()
    f2.savefig(p.join("out/delay.png"))

    """
    ax_twin = ax.twinx()
    ax_twin.plot("time", "value", data=df, color='r', label="packet delays")
    ax_twin.set_ylim(0, 120)
    ax_twin.set_ylabel(f"mean delay (window size: {time_per_bin}s )[s]")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='r', linewidth=1))
    labels.append("packet delay")
    ax.legend().remove()
    ax_twin.legend().remove()
    plt.legend().remove()
    f1.legends.clear()
    f1.legend(handles, labels)
    f1.savefig(p.join("out/count_delay.png"))
        """

    # plt.close(f1)
    plt.close(f2)

def animate_map_plot(dcd):
    node_id = 0
    frames = dcd.unique_level_values("simtime").to_list()

    f, ax = dcd.plot_density_map(time_step=frames[0],
                                 node_id=node_id,
                                 make_interactive=False,
                                 pcolormesh_dic=dict(vmin=0, vmax=4),
                                 )
    ax.set_title("")

    def plot_f(frame, dcd, ax):
        print(f"plot_f frame {frame}/{frames[-1]}")
        dcd.update_color_mesh(ax.collections[0], time_step=frame, node_id=node_id)
        # ax.set_title(f"{frame} s", fontsize=20)

    anim = animation.FuncAnimation(fig=f, func=plot_f, frames=frames[1:], blit=False, fargs=(dcd, ax))
    anim.save(p.join("out/map.mp4"))


def animate_map_plot2(dcd):
    node_id = 1
    frames_node = dcd.unique_level_values("simtime", df_slice=(node_id,)).to_list()

    node_glb = 0
    frames_glb = dcd.unique_level_values("simtime").to_list()

    fig, axes = plt.subplots(2, 1, figsize=(9, 16))
    # fig.tight_layout()

    _, ax_node = dcd.plot_density_map(time_step=frames_node[0],
                                      node_id=node_id,
                                      ax=axes[0],
                                      pcolormesh_dic=dict(vmin=0, vmax=4),
                                      )

    _, ax_global = dcd.plot_density_map(time_step=frames_glb[0],
                                        node_id=node_glb,
                                        ax=axes[1],
                                        pcolormesh_dic=dict(vmin=0, vmax=4),
                                        )

    ax_global.set_title("Global view (ground truth)")

    def plot_f(frame, dcd):
        print(f"plot_f frame {frame}/{frames_glb[-1]}")
        # global
        dcd.update_color_mesh(ax_global.collections[0], time_step=frame, node_id=node_glb)
        fig.suptitle(f"Current time: {frame} s", fontsize=20)

        # node
        if frame in frames_node:
            ax_node.set_title(f"Local view (Node {node_id})")
            dcd.update_color_mesh(ax_node.collections[0], time_step=frame, node_id=node_id)
        else:
            ax_node.set_title(f"Local view (Node {node_id}) - NO DATA ")
            dcd.clear_color_mesh(ax_node.collections[0])

    anim = animation.FuncAnimation(fig=fig, func=plot_f, frames=frames_glb[1:], blit=False, fargs=(dcd,))
    anim.save(p.join("out/map2.mp4"))


if __name__ == "__main__":
    dcd = read_data()
    para = read_param()
    delay = read_app_data()

    # delay = read_app_data()
    # animate_map_plot(dcd)
    analyse_interactive()
    # make_count_plot(dcd, para)
    # make_delay_plot(dcd, para, delay)
    make_count_plot(delay)
