import fnmatch
import os
import re
from typing import List
import logging as log

from config import *
from roveranalyzer.simulators.opp.utils import ScaveTool
from roveranalyzer.utils import PathHelper
from roveranalyzer.simulators.crownet.dcd.dcd_map import DcdMap2DMulti
from roveranalyzer.utils import check_ax

import pandas as pd
import numpy as np
import sqlite3
import matplotlib
import matplotlib.pyplot as plt
import roveranalyzer.simulators.opp as OMNeT

matplotlib.use('TkAgg')


"""
def read_param():
    scave_tool = ScaveTool()
    SCA = f"{ROOT}/{RUN}/{SPECIFIC_RUN}/vars_rep_0.sca"
    scave_filter = scave_tool.filter_builder().t_parameter().build()
    df_parameters = scave_tool.read_parameters(SCA, scave_filter=scave_filter)
    return df_parameters
"""


def sqls_matching_sim_config(
        sim_config: str = SIM_CONFIG,
        search_path: str = ROOT
) -> List[OMNeT.CrownetSql]:
    sqls = []
    for vec_file in vec_files_matching_sim_config(sim_config, search_path):
        sca_file = ".".join(vec_file.split(".")[0:-1]) + ".sca"
        sqls.append(
           OMNeT.CrownetSql(
               vec_path=vec_file,
               sca_path=sca_file,
               network="World")
        )
    return sqls


def vec_files_matching_sim_config(
        sim_config: str = SIM_CONFIG,
        search_path: str = ROOT
) -> List[str]:
    vec_files = find("*.vec", search_path)
    vec_files = [file for file in vec_files if sim_config in file.split("/")[-2]]
    validate_run_count(vec_files)
    return vec_files


def validate_run_count(vec_files: List[str]):
    target = set([i for i in range(RUN_COUNT)])
    for file in vec_files:
        run_index = int(file.split(".")[0].split("_")[-1])
        if run_index >= RUN_COUNT:
            log.warning(f"Using simulation run with index higher than run count. Index: {run_index}")
        else:
            try:
                target.remove(run_index)
            except KeyError:
                log.warning(f"Duplicate simulation run with index: {run_index}")
    if len(target) > 0:
        log.warning(f"Missing simulation run with index: {str(target)}")


def read_spawn_times():
    df_list = list()
    vec_files = vec_files_matching_sim_config()
    for i in range(len(vec_files)):
        vec_file = vec_files[i]
        con = sqlite3.connect(vec_file)
        df = pd.read_sql_query(
            "SELECT v.moduleName, v.vectorName, v.startSimtimeRaw, v.endSimtimeRaw "
            "FROM vector v "
            f"WHERE v.moduleName LIKE 'World.{NODE_NAME}[%].app[{BEACON_APP_INDEX}].app' "
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
    vec_files = vec_files_matching_sim_config()
    for i in range(len(vec_files)):
        # inputs = f"{ROOT}/{RUN}/vars_rep_{i}.vec"
        # scave = ScaveTool()
        scave_f = callback()

        """
        scave = ScaveTool()
        scave_f = scave.filter_builder() \
            .gOpen().module(f"*.{NODE_NAME}[*].app[{DENSITY_APP_INDEX}].app")
            .OR().module(f"*.{NODE_NAME}[*].app[{BEACON_APP_INDEX}].app").gClose() \
            .AND().name("rcvdPkLifetime:vector")

        
        scave_f = ScaveTool()..filter_builder() \
            .gOpen().module("*World.{NODE_NAME}[*].cellularNic.mac").OR().module("*.{NODE_NAME}[*].cellularNic.mac").gClose() \
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


def find(pattern, path) -> List[str]:
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def filter_for_packageDelay():
    scave = ScaveTool()
    return scave.filter_builder() \
        .gOpen().module(f"*.{NODE_NAME}[*].app[{DENSITY_APP_INDEX}].app")\
        .OR().module(f"*.{NODE_NAME}[*].app[{BEACON_APP_INDEX}].app").gClose() \
        .AND().name("rcvdPkLifetime:vector")


def filter_for_sentPacketToUpper():
    scave = ScaveTool()
    return scave.filter_builder() \
        .gOpen().module(f"*World.{NODE_NAME}[*].cellularNic.mac").OR().module(f"*.{NODE_NAME}[*].cellularNic.mac").gClose() \
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
    f2.savefig(os.path.join(OUT_PATH, f"delay_rcvdPackage_median.png"))


def generate_mean_delay_per_run(delay):
    f, ax = check_ax()
    ax.set_title("rcvdPkLifetime (delay) of all packets (beacon + map)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(f"median delay (window size: 1.0s )[s]")
    repList = delay.repetitionId.unique()
    repList.sort()
    for a in repList:
        f1, ax1 = check_ax()
        df_filtered = delay[delay['repetitionId'] == f"{a}"]
        if not df_filtered.empty:
            df_filtered = df_filtered.opp.filter().vector().normalize_vectors(axis=0)
            plots = [[f"Packet Delay Run #{a}", df_filtered]]
            time_per_bin = 1.0  # seconds
            for n, df in plots:
                bins = int(np.floor(df["time"].max() / time_per_bin))
                df_mean = df.groupby(pd.cut(df["time"], bins)).mean()
                df_mean = df_mean.dropna()
                ax.plot("time", "value", data=df_mean, label=n)
                ax1.plot("time", "value", data=df_mean, label=n)

                ax1.set_ylim(0, 30)
                ax1.set_xlim(0, 800)

                ax.set_title("rcvdPkLifetime (delay) of all packets (beacon + map)", y=1.05)
                ax1.set_title("rcvdPkLifetime (delay) of all packets (beacon + map)", y=1.05)
                ax1.set_xlabel("time [s]")
                ax1.set_ylabel(f"median delay (window size: {time_per_bin}s )[s]")
                ax1.legend()
                ax.legend()

                f1.savefig(os.path.join(OUT_PATH_DELAY, f"delay_rcvdPackage_mean_{a}.png"))

    ax = plot_vspans(ax)
    f.savefig(os.path.join(OUT_PATH_DELAY, f"delay_rcvdPackage_mean.png"))


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
        plt.yticks(np.arange(0, 45, 5))

        ax1.set_ylim(0, 30)
        ax1.set_xlim(0, 800)
        ax2.set_ylim(0, 30)
        ax2.set_xlim(0, 800)

    label = "time [s]"
    title = "rcvdPkLifetime (delay) of all packets (beacon + map)"
    ax1 = plot_vspans(ax1)
    ax2 = plot_vspans(ax2)
    ax1.set_title(title, y = 1.05)
    ax2.set_title(title, y = 1.05)
    ax1.set_xlabel(label)
    ax2.set_xlabel(label)
    # plt.xticks(np.arange(0, 550, 50))

    ax1.set_ylabel(f"median delay (window size: {time_per_bin}s )[s]")
    ax2.set_ylabel(f"mean delay (window size: {time_per_bin}s )[s]")
    ax1.legend()
    ax2.legend()

    f1.savefig(os.path.join(OUT_PATH, f"delay_rcvdPackage_median.png"))
    f2.savefig(os.path.join(OUT_PATH, f"delay_rcvdPackage_mean.png"))


def delay_with_ped_count_and_distance_to_enb(delay):
    import position_analysis as pos

    df_all = delay.opp.filter().vector().normalize_vectors(axis=0)
    distancPedCountPlot = pos.calculate_distance_between_pedestrians_and_enb()

    plots = [["packet delay", df_all]]
    time_per_bin = 1.0  # seconds
    for n, df in plots:
        bins = int(np.floor(df["time"].max() / time_per_bin))
        df_mean = df.groupby(pd.cut(df["time"], bins)).mean()
        df_mean = df_mean.dropna()
        distancPedCountPlot.plot("time", "value", data=df_mean, label=n)

    fig = distancPedCountPlot.get_figure()
    fig.savefig(os.path.join(OUT_PATH, f"delay_ped_count_distance_enb.png"))


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
        ax1 = plot_vspans(ax1)
        ax2 = plot_vspans(ax2)

        ax1.set_ylim(0, 500)
        ax1.set_xlim(0, 800)
        ax2.set_ylim(0, 500)
        ax2.set_xlim(0, 800)

        # plt.xticks(np.arange(0, 700, 100))
        # ax2.set_xlim(0, 550)

    label = "time [s]"
    title = "Amount of bytes sent (beacon + map)"
    ax1.set_title(title, y = 1.05)
    ax2.set_title(title, y = 1.05)
    ax1.set_xlabel(label)
    ax2.set_xlabel(label)
    # plt.xticks(np.arange(0, 550, 50))

    ax1.set_ylabel(f"Median amount of bytes (window size: {time_per_bin}s )[s]")
    ax2.set_ylabel(f"Mean amount of bytes (window size: {time_per_bin}s )[s]")

    ax1.legend()
    ax2.legend()

    f1.savefig(os.path.join(OUT_PATH, f"delay_sentPacketUpper_median.png"))
    f2.savefig(os.path.join(OUT_PATH, f"delay_sentPacketUpper_mean.png"))


def all_pedestrians_instantiated():
    df = read_spawn_times()
    latest_spawn_time = df["startTime"].max()
    print(latest_spawn_time)


def mean_simulation_time():
    data = []
    df = read_spawn_times()
    for i in range(10):
        df_filtered = df[df['runId'] == f"{i}"]
        last_person_time = int(df_filtered["endTime"].max())
        data.append([i, last_person_time])

    df_sim_time = pd.DataFrame(data, columns=['runId', 'sim_time'])
    if SIM_CONFIG == "vadereBottleneck":
        df_sim_time = df_sim_time[df_sim_time['runId'] != 1]

    return df_sim_time['sim_time'].mean()


def simulation_duration_per_run():
    data = []
    df = read_spawn_times()
    for i in range(10):
        df_filtered = df[df['runId'] == f"{i}"]
        last_person_time = int(df_filtered["endTime"].max())
        data.append([i, last_person_time])

    df_sim_time = pd.DataFrame(data, columns=['runId', 'sim_time'])

    ax = df_sim_time.plot.bar(x='runId', y='sim_time', label="Pedestrian Count", rot=0)
    ax.set_xlabel("Run Id")
    ax.set_ylabel("Time in [s]")
    ax.set_ylim(0, 1000)
    ax.axhline(mean_simulation_time(), label="Mean Simulation Duration", color="red", alpha=1, linestyle="--")
    ax.legend()

    plt.yticks(np.arange(0, 1000, 100))
    plt.title("Length of Simulation per Run")

    fig = ax.get_figure()
    fig.savefig(os.path.join(OUT_PATH, f"simulation_times.png"))


def data_with_ped_count(data):
    # Mean Pedestrian Count
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
    df_reduced = df_ped_count_agg[df_ped_count_agg['time'] % 10 == 0]

    # Mean Package Delay
    df_all = data.opp.filter().vector().normalize_vectors(axis=0)

    f1, ax1 = check_ax()
    ax2 = ax1.twinx()

    plots = [["Mean Packet Delay", df_all]]
    time_per_bin = 1.0  # seconds
    for n, df in plots:
        bins = int(np.floor(df["time"].max() / time_per_bin))
        df_mean = df.groupby(pd.cut(df["time"], bins)).mean()
        df_mean = df_mean.dropna()
        ax1.plot("time", "value", data=df_mean, label=n)
        ax2.plot("time", "pedCount", color="purple", label="Mean Pedestrian Count", data=df_reduced)

    label = "time [s]"
    title = "Mean Bytes Sent"
    ax1.set_ylim(0, 500)
    ax1.set_xlim(0, 800)
    ax1 = plot_vspans(ax1)
    ax1.set_title(title, y=1.05)
    ax1.set_xlabel(label)
    ax1.set_ylabel(f"Mean Bytes Sent (window size: {time_per_bin}s )[s]")
    ax2.set_ylabel("Number of Pedestrians")
    ax1.legend()
    # ax1.legend(loc='center left', bbox_to_anchor=(1.0, 1.0))

    fig = ax1.get_figure()
    fig.savefig(os.path.join(OUT_PATH, f"data_with_ped_count.png"), bbox_inches="tight")


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
    df_reduced = df_ped_count_agg[df_ped_count_agg['time'] % 10 == 0]

    ax = df_reduced.plot.bar(x='time', y='pedCount', rot=0, figsize=(8, 6))
    ax.set_xlabel("Time in [s]")
    ax.set_ylabel("Pedestrian Count")
    plt.yticks(np.arange(0, 31, 1))
    plt.title("Mean Pedestrian Count Per Time Over All Simulations")

    fig = ax.get_figure()
    fig.savefig(os.path.join(OUT_PATH, f"mean_ped_count_over_time.png"))

    return df_reduced


def pedestrian_count_per_run():
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
        fig.savefig(os.path.join(OUT_PATH_PED_COUNT, f"pedestrian_count_{i}.png"))


intervalList = [[10, 80], [90, 120], [150, 300], [350, 430], [500, 640]] # Sumo Bottleneck
# intervalList = [[10, 70], [80, 120], [150, 350], [400, 510]]  # Sumo Simple


def plot_vspans(ax):
    if PAINT_INTERVALS:
        intervalCounter = 1

        ax.axvline(mean_simulation_time(), label="Mean Simulation Time", color="red", alpha=1, linestyle="--")
        for begin, end in intervalList:
            # Interval
            ax.axvspan(begin, end, color="yellow", alpha=0.35, ec="red")
            ax.annotate(f"{intervalCounter}", xy=(((begin + end) / 2) - 6, ax.get_ylim()[1]), fontsize=12)
            intervalCounter += 1

        return ax


def delay_mean_variance_per_interval(data):
    df_all = data.opp.filter().vector().normalize_vectors(axis=0)
    for i, x in intervalList:
        df_filtered = df_all.query(f"time >= {i}").query(f"time <= {x}")
        print(f"Interval {i} to {x}")
        print(f"{df_filtered['value'].max()}\t{df_filtered['value'].min()}\t{df_filtered['value'].mean()}\t{df_filtered['value'].median()}\t{df_filtered['value'].var()}\t{df_filtered['value'].std()}")
        # print(f"Maximum: {df_filtered['value'].max()}")
        # print(f"Minimum: {df_filtered['value'].min()}")
        # print(f"Mean: {df_filtered['value'].mean()}")
        # print(f"Median: {df_filtered['value'].median()}")
        # print(f"Variance: {df_filtered['value'].var()}")
        # print(f"Standard Deviation: {df_filtered['value'].std()}")
        print("-------------------------")


if __name__ == "__main__":

    # all_pedestrians_instantiated()
    # simulation_duration_per_run()
    # mean_pedestrian_count()
    print(mean_simulation_time())

    delay = read_app_data(filter_for_packageDelay)
    upper = read_app_data(filter_for_sentPacketToUpper)

    # Reduce failed Vadere run
    if SIM_CONFIG == "vadereBottleneck":
        delay = delay[delay['repetitionId'] != "1"]
        # upper = upper[upper['repetitionId'] != "1"]

    # Delay with Mean Pedestrian Count
    data_with_ped_count(upper)
    # delay_with_ped_count_and_distance_to_enb(delay)

    # Mean/Median Delay
    # generate_mean_delay_per_run(delay)
    # generate_plots_rcvdPackage_delay_median(delay)
    # delay_mean_variance_per_interval(delay)

    # Mean/Median sentPacketToUpper
    # delay_mean_variance_per_interval(upper)
    # generate_plots_sendPacketToUpper_delay_median(upper)
