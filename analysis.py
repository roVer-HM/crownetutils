import os, fnmatch, re
import matplotlib
from matplotlib.lines import Line2D

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
import sqlite3
import numpy as np

ROOT = "/home/mkilian/repos/crownet/analysis/roveranalyzer/data"
RUN = "sumoSimple"
SPECIFIC_RUN = ""
IS_VADERE_ANALYSIS = False
NODE_NAME = "node" if IS_VADERE_ANALYSIS else "pedestrianNode"

PATH_ROOT = f"{ROOT}/{RUN}"
PATH_SPECIFIC_RUN = f"{ROOT}/{RUN}/{SPECIFIC_RUN}"
p = PathHelper(PATH_ROOT)
p_specific = PathHelper(PATH_SPECIFIC_RUN)


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
    SCA = f"{ROOT}/{RUN}/{SPECIFIC_RUN}/vars_rep_0.sca"
    scave_filter = scave_tool.filter_builder().t_parameter().build()
    df_parameters = scave_tool.read_parameters(SCA, scave_filter=scave_filter)
    return df_parameters


def read_spawn_times():
    df_list = list()
    vec_files = find("*.vec", PATH_ROOT)
    for i in range(len(vec_files)):
        vec_file = vec_files[i]
        con = sqlite3.connect(vec_file)
        df = pd.read_sql_query(
            "SELECT v.moduleName, v.vectorName, v.startSimtimeRaw, v.endSimtimeRaw "
            "FROM vector v "
            f"WHERE v.moduleName LIKE 'World.{NODE_NAME}[%].aid.beaconApp' "
            "AND v.vectorName = 'packetSent:vector(packetBytes)' "
            "ORDER BY v.startSimtimeRaw ASC", con)
        df['startTime'] = df['startSimtimeRaw'].apply(lambda x: int(str(x)[:4]) / 1000)
        df['endTime'] = df['endSimtimeRaw'].apply(lambda x: int(str(x)[:4]) / 10)
        df['id'] = df['moduleName'].apply(lambda x: find_number(x))
        df['runId'] = re.search('vars_rep_(.+?).vec', vec_files[i]).group(1)
        df = df.drop(columns=['moduleName', 'vectorName', 'startSimtimeRaw', 'endSimtimeRaw'])
        df_list.append(df)
    return pd.concat(df_list, axis=0)


def find_number(text):
    num = re.findall(r'[0-9]+', text)
    return " ".join(num)


# @from_pickle(path=PATH_ROOT + "/rcvdPkLifetimeVec.p")
def read_app_data(callback):
    df_list = list()
    vec_files = find("*.vec", PATH_ROOT)
    for i in range(len(vec_files)):
        # inputs = f"{ROOT}/{RUN}/vars_rep_{i}.vec"
        # scave = ScaveTool()
        scave_f = callback()

        """
        scave = ScaveTool()
        scave_f = scave.filter_builder() \
            .gOpen().module(f"*.{NODE_NAME}[*].aid.densityMapApp").OR().module(f"*.{NODE_NAME}[*].aid.beaconApp").gClose() \
            .AND().name("rcvdPkLifetime:vector")

        
        scave_f = ScaveTool()..filter_builder() \
            .gOpen().module("*World.{NODE_NAME}[*].lteNic.mac").OR().module("*.{NODE_NAME}[*].lteNic.mac").gClose() \
            .AND().name("sentPacketToUpperLayer:vector*")
        """

        _df = ScaveTool().load_df_from_scave(input_paths=vec_files[i], scave_filter=scave_f)
        rep_id = re.search('vars_rep_(.+?).vec', vec_files[i]).group(1)
        _df['repetitionId'] = str(rep_id)
        df_list.append(_df)

    df_all = df_list[0]
    for i in range(1, len(df_list)):
        df_all = df_all.append(df_list[i], ignore_index=True)
    # df_all = pd.concat(df_list, axis=1)
    return df_all


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def filter_for_packageDelay():
    scave = ScaveTool()
    return scave.filter_builder() \
        .gOpen().module(f"*.{NODE_NAME}[*].aid.densityMapApp").OR().module(f"*.{NODE_NAME}[*].aid.beaconApp").gClose() \
        .AND().name("rcvdPkLifetime:vector")


def filter_for_sentPacketToUpper():
    scave = ScaveTool()
    return scave.filter_builder() \
        .gOpen().module(f"*World.{NODE_NAME}[*].lteNic.mac").OR().module("*.{NODE_NAME}[*].lteNic.mac").gClose() \
        .AND().name("sentPacketToUpperLayer:vector*")


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
    f2.savefig(p.join("out/delay_rcvdPackage_median.png"))


def generate_plots_rcvdPackage_delay_combined():
    delay_vadere = read_app_data(filter_for_packageDelay)
    IS_VADERE_ANALYSIS = False
    delay_sumo = read_app_data(filter_for_packageDelay)
    df_all_sumo = delay_sumo.opp.filter().vector().normalize_vectors(axis=0)
    df_all_vadere = delay_vadere.opp.filter().vector().normalize_vectors(axis=0)
    f2, ax2 = check_ax()
    f3, ax3 = check_ax()

    # plots = [["beacon", df_beacon], ["map", df_dMap], ["all", df_all]]:
    plot = None
    plots = [["packet delay sumo", df_all_sumo]]
    time_per_bin = 1.0  # seconds
    for n, df in plots:
        bins = int(np.floor(df["time"].max() / time_per_bin))
        df = df.groupby(pd.cut(df["time"], bins)).mean()
        df = df.dropna()
        # ax2.plot("time", "value", data=df, label=n)
        plot = df.plot("time", "value", label=n)

    plots = [["packet delay vadere", df_all_vadere]]
    time_per_bin = 1.0  # seconds
    for n, df in plots:
        bins = int(np.floor(df["time"].max() / time_per_bin))
        df = df.groupby(pd.cut(df["time"], bins)).mean()
        df = df.dropna()
        data = df.plot("time", "value", label=n, ax=plot, figsize=(16, 9))
        data.set_xlabel("time [s]")
        data.set_ylabel(f"mean delay (window size: {time_per_bin}s )[s]")
        # data.set_ylim(0, 30)
        # ax3.plot("time", "value", data=data, label=n)
        # fig = bla.get_figure()
        data.get_figure().savefig(p.join("out/delay_sentPacketToUpper_combined.png"))

    ax3.set_title("rcvdPkLifetime (delay) of all packets (beacon + map)")
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel(f"mean delay (window size: {time_per_bin}s )[s]")
    ax3.legend()

    # f3.savefig(p.join("out/delay_rcvdPackage_combined.png"))


def generate_mean_delay_per_run(delay):

    f, ax = check_ax()
    ax.set_title("rcvdPkLifetime (delay) of all packets (beacon + map)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(f"median delay (window size: 1.0s )[s]")
    for a in range(len(delay.repetitionId.unique())):
        f1, ax1 = check_ax()
        df_filtered = delay[delay['repetitionId'] == f"{a}"]
        df_filtered = df_filtered.opp.filter().vector().normalize_vectors(axis=0)
        plots = [[f"Packet Delay Run #{a}", df_filtered]]
        time_per_bin = 1.0  # seconds
        for n, df in plots:
            bins = int(np.floor(df["time"].max() / time_per_bin))
            df_mean = df.groupby(pd.cut(df["time"], bins)).mean()
            df_mean = df_mean.dropna()
            ax.plot("time", "value", data=df_mean, label=n)
            ax1.plot("time", "value", data=df_mean, label=n)
            ax.set_xlim(0, 200)
            ax1.set_title("rcvdPkLifetime (delay) of all packets (beacon + map)")
            ax1.set_xlabel("time [s]")
            ax1.set_ylabel(f"median delay (window size: {time_per_bin}s )[s]")
            ax1.legend()
            ax.legend()

            f1.savefig(p.join(f"out/delay/delay_rcvdPackage_mean_{a}.png"))

    f.savefig(p.join(f"out/delay/delay_rcvdPackage_mean.png"))

def generate_plots_rcvdPackage_delay_median(delay):
    df_all = delay.opp.filter().vector().normalize_vectors(axis=0)

    f1, ax1 = check_ax()
    f2, ax2 = check_ax()

    plots = [["packet delay", df_all]]
    time_per_bin = 1.0  # seconds
    for n, df in plots:
        bins = int(np.floor(df["time"].max() / time_per_bin))
        df_mean = df.groupby(pd.cut(df["time"], bins)).mean()
        df_median = df.groupby(pd.cut(df["time"], bins)).median()
        df_mean = df_mean.dropna()
        df_median = df_median.dropna()
        ax1.plot("time", "value", data=df_median, label=n)
        ax2.plot("time", "value", data=df_mean, label=n)
        # ax2.set_ylim(0, 35)
        # ax2.set_xlim(0, 550)

    label = "time [s]"
    title = "rcvdPkLifetime (delay) of all packets (beacon + map)"
    ax1.set_title(title)
    ax2.set_title(title)
    ax1.set_xlabel(label)
    ax2.set_xlabel(label)
    # plt.xticks(np.arange(0, 550, 50))

    ax1.set_ylabel(f"median delay (window size: {time_per_bin}s )[s]")
    ax2.set_ylabel(f"mean delay (window size: {time_per_bin}s )[s]")
    ax1.legend()
    ax2.legend()

    f1.savefig(p.join("out/delay_rcvdPackage_median.png"))
    f2.savefig(p.join("out/delay_rcvdPackage_mean.png"))


def generate_plots_sendPacketToUpper_delay_median(delay):
    df_all = delay.opp.filter().vector().normalize_vectors(axis=0)

    f1, ax1 = check_ax()
    f2, ax2 = check_ax()

    # plots = [["beacon", df_beacon], ["map", df_dMap], ["all", df_all]]:
    plots = [["sentPacketUpper amount", df_all]]
    time_per_bin = 1.0  # seconds
    for n, df in plots:
        bins = int(np.floor(df["time"].max() / time_per_bin))
        df_mean = df.groupby(pd.cut(df["time"], bins)).mean()
        df_median = df.groupby(pd.cut(df["time"], bins)).median()
        df_mean = df_mean.dropna()
        df_median = df_median.dropna()
        ax1.plot("time", "value", data=df_median, label=n)
        ax2.plot("time", "value", data=df_mean, label=n)
        # ax2.set_ylim(0, 500)
        # ax2.set_xlim(0, 550)

    label = "time [s]"
    title = "sendPacketToUpper of all packets (beacon + map)"
    ax1.set_title(title)
    ax2.set_title(title)
    ax1.set_xlabel(label)
    ax2.set_xlabel(label)
    # plt.xticks(np.arange(0, 550, 50))

    ax1.set_ylabel(f"median sentPacketUpper (window size: {time_per_bin}s )[s]")
    ax2.set_ylabel(f"mean sentPacketUpper (window size: {time_per_bin}s )[s]")

    ax1.legend()
    ax2.legend()

    f1.savefig(p.join("out/delay_sentPacketUpper_median.png"))
    f2.savefig(p.join("out/delay_sentPacketUpper_mean.png"))


def all_pedestrians_instantiated():
    df = read_spawn_times()
    latest_spawn_time = df["startTime"].max()


def mean_simulation_time():
    data = []
    df = read_spawn_times()
    for i in range(10):
        df_filtered = df[df['runId'] == f"{i}"]
        last_person_time = int(df_filtered["endTime"].max())
        data.append([i, last_person_time])

    df_sim_time = pd.DataFrame(data, columns=['runId', 'sim_time'])

    # fig, ax = plt.subplots(1, 1)
    y = df_sim_time.sim_time.values
    x = df_sim_time.runId.values
    # ax.scatter(x, y, marker='.', color='red')
    ax = df_sim_time.plot.bar(x='runId', y='sim_time', rot=0)

    # plt.xticks(np.arange(0, 10, 1))
    ax.set_xlabel("Run Id")
    ax.set_ylabel("Time in [s]")
    plt.title("Length of Simulation per Run")

    fig = ax.get_figure()
    fig.savefig(p.join("out/simulation_times.png"))


def mean_pedestrian_count():
    list = []
    df = read_spawn_times()
    for i in range(10):
        df_filtered = df[df['runId'] == f"{i}"]
        last_person_time = int(df_filtered["endTime"].max())

        for a in range(10, last_person_time, 10):
            tmp = df_filtered[df_filtered['endTime'] > a]
            count = len(tmp)
            list.append([i, a, count])

    df_ped_count = pd.DataFrame(list, columns=['runId', 'time', 'pedCount'])
    df_ped_count_agg = df_ped_count.groupby(['time'], as_index=False).agg({"pedCount": np.mean})
    df_reduced = df_ped_count_agg[df_ped_count_agg['time'] % 50 == 0]

    ax = df_reduced.plot.bar(x='time', y='pedCount', rot=0)
    ax.set_xlabel("Time in [s]")
    ax.set_ylabel("Pedestrian Count")
    plt.title("Mean Pedestrian Count Per Time Over All Simulations")

    fig = ax.get_figure()
    fig.savefig(p.join("out/mean_ped_count_over_time.png"))


def pedestrian_count_plot():
    df = read_spawn_times()
    for i in range(10):
        data = []
        df_filtered = df[df['runId'] == str(i)]
        last_person_time = int(df_filtered["endTime"].max())

        for a in range(10, last_person_time, 10):
            tmp = df_filtered[df_filtered['endTime'] > a]
            count = len(tmp)
            data.append([a, count])

        df_ped_count = pd.DataFrame(data, columns=['time', 'pedCount'])

        fig, ax = plt.subplots(1, 1)
        x = df_ped_count.time.values
        y = df_ped_count.pedCount.values
        ax.plot(x, y)
        ax.set_xlabel("Time in [s]")
        ax.set_ylabel("Pedestrian Count")
        plt.title("Pedestrian Count over Time")

        fig = ax.get_figure()
        fig.savefig(p.join(f"out/ped_count/pedestrian_count_{i}.png"))


def make_delay_plot(dcd, para, delay):
    # make count plot
    f1, ax = dcd.plot_count_diff()
    maxAge = para.loc[para["name"] == "maxAge", ["value"]].iloc[0].value
    title = f"{ax.title.get_text()} with neighborhood table maxAge {maxAge}"
    ax.set_title(title)

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

    # plt.close(f1)
    plt.close(f2)


if __name__ == "__main__":
    # all_pedestrians_instantiated()
    # mean_simulation_time()
    # mean_pedestrian_count()

    # dcd = read_data()
    # para = read_param()
    delay = read_app_data(filter_for_packageDelay)
    # upper = read_app_data(filter_for_sentPacketToUpper)

    # make_delay_plot(dcd, para, delay)
    # pedestrian_count(dcd)

    # Mean/Median delay and sentPacketToUpper
    generate_mean_delay_per_run(delay)
    # generate_plots_rcvdPackage_delay_median(delay)
    # generate_plots_sendPacketToUpper_delay_median(upper)

    # Generate Pedestrian Count Plots
    # pedestrian_count()

    # generate_plots_rcvdPackage_delay_combined()
