import fnmatch
import math
import os
import sys
from typing import List, Union
import logging as log

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import *
from analysis import plot_vspans, intervalList, mean_pedestrian_count, sqls_matching_sim_config
import roveranalyzer.simulators.opp as OMNeT


def positions_dataframes_all() -> List[pd.DataFrame]:
    df_all = []
    for sql in sqls_matching_sim_config():
        df_all.append(positions_dataframe(sql))

    df = pd.concat(df_all, ignore_index=True)

    if SIM_CONFIG == "vadereBottleneck":  # TODO ???
        df = df[df['runId'] != 1]

    return df


def positions_dataframe(sql: OMNeT.CrownetSql):
    df = sql.host_position()
    offset = sql.sca_data(
        module_name=f"{sql.network}.coordConverter",
        scalar_name=sql.OR(["simOffsetX:last", "simOffsetY:last"]),
    )["scalarValue"].to_numpy()
    df["x"] = df["x"] + offset[0]
    df["y"] = df["y"] + offset[1]

    run_id = sql.query_sca(f"select attrValue from runAttr r where r.attrName = 'runnumber' ").iloc[0]['attrValue']
    run_id = int(run_id)
    df['runId'] = run_id
    df.rename(columns={"vecIdx": "id"}, inplace=True)
    return df


DATAFRAME_ALL_RUNS: Union[pd.DataFrame, None] = None


def read_position_files():
    global DATAFRAME_ALL_RUNS
    if DATAFRAME_ALL_RUNS is None:
        DATAFRAME_ALL_RUNS = positions_dataframes_all()
    return DATAFRAME_ALL_RUNS.copy(deep=True)
# def read_position_files():
#     position_list = find('positions.txt', PATH_ROOT)
#     position_list.sort()
#
#     df_all = []
#     columns = ["time", "id", "x", "y"]
#     for i in range(len(position_list)):
#         tmp = pd.read_csv(position_list[i], delimiter="\t")
#         tmp.columns = columns
#         tmp['runId'] = i
#         df_all.append(tmp)
#
#     df = pd.concat(df_all, ignore_index=True)
#
#     if SIM_CONFIG == "vadereBottleneck":
#         df = df[df['runId'] != 1]
#
#     return df


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def scatter_pedestrian_positions():
    run_id = 2
    df = read_position_files()
    df_filtered = df[df['runId'] == run_id]

    for begin, end in intervalList:
        begin = int(begin)
        end = int(end)
        if begin > df_filtered["time"].max():
            continue
        df_all = df_filtered.query(f'time >= {begin}').query(f'time <= {end}')
        minimum = round((df_all['y'].min() - 10) / 10) * 10
        maximum = round((df_all['y'].max() + 10) / 10) * 10

        for u in [begin, end - (round(((end - begin) / 2) / 10) * 10), end]:
            df_filter = df_filtered[df_filtered['time'] == u]

            x = df_filter.x.values
            y = df_filter.y.values

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.scatter(x, y, label='Pedestrian', marker='.', color='red')
            ax.set_ylim(minimum, maximum)
            ax.set_xlabel("x-coordinate")
            ax.set_ylabel("y-coordinate")
            ax.legend()

            plt.title("Pedestrian positions at time step " + str(u))
            plt.yticks(np.arange(minimum, maximum, 10))

            fig = ax.get_figure()
            fig.savefig(OUT_PATH_POSITION + '/position_' + str(u) + '_' + str(run_id) + '.png', bbox_inches='tight')


def scatter_plot_min_max():
    df = read_position_files()
    df = df.drop(columns="id")
    df_time = df.groupby("time").agg({"y": [np.max, np.min], "x": [np.max, np.min]})
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    x = df_time.index.values
    y_min = df_time.loc[:, ('y', 'amin')]
    y_max = df_time.loc[:, ('y', 'amax')]

    ax.scatter(x, y_max, label='max y coord', marker='_', color='blue', alpha=1)
    ax.scatter(x, y_min, label='min y coord', marker='_', color='green', alpha=1)
    ax = plot_vspans(ax)
    ax.set_xlabel("time in [s]")
    ax.set_ylabel("y-coordinates")
    ax.legend()

    plt.title("Min/Max Positions", y=1.05)

    plt.xticks(np.arange(0, 900, 100))
    # plt.grid()

    fig = ax.get_figure()
    fig.savefig(OUT_PATH_POSITION + '/min_max.png')


def average_speed():
    p = "source/Straight_Line_Deviation_Sumo"
    type = '.csv'
    df: pd.DataFrame = pd.read_csv(p + type)
    df_id = df[df['id'] == 10]
    fig, ax = plt.subplots(1, 1)

    x = df_id.time.values
    distance = []
    for i in range(len(df_id)):
        if i == 0:
            distance.append(580 - df_id.iloc[i].y)
        elif i == len(df_id) - 1:
            distance.append(df_id.iloc[i - 1].y - df_id.iloc[i].y)
            break
        else:
            row1, row2 = df_id.iloc[i - 1], df_id.iloc[i]
            distance.append(row1.y - row2.y)
    df_id['distance'] = distance
    y = df_id.distance.values
    plt.figure(figsize=(8, 8), dpi=80)
    plt.scatter(x, y, marker='.')
    # ax.scatter(x,y,marker='_', color='red')
    # plt.xticks(np.arange(0, 300, 10))
    # df_id.plot(x = 'time', y = 'distance', kind='line')
    plt.show()


def average_speed_per_pedestrian_per_run():
    df_master = read_position_files()
    for z in range(0, 10):
        df_filtered = df_master[df_master['runId'] == z]
        df = df_filtered.groupby(["time", "id"], as_index=False).agg({"x": np.mean, "y": np.mean})

        fig, ax = plt.subplots(1, 1)
        x = df.id.unique()
        df_speed = pd.DataFrame(columns=['id', 'averageSpeed'])
        for a in range(min(x), max(x) + 1):
            node_id = 0
            distance = []
            df_id = df[df['id'] == a]
            for b in range(len(df_id)):
                node_id = a
                if b == 0:
                    distance.append(580 - df_id.iloc[b].y)
                elif b == len(df_id) - 1:
                    distance.append(df_id.iloc[b - 1].y - df_id.iloc[b].y)
                    break
                else:
                    row1, row2 = df_id.iloc[b - 1], df_id.iloc[b]
                    distance.append(row1.y - row2.y)
            df_id['distance'] = distance
            df = pd.concat([df, df_id]).drop_duplicates(['x', 'y', 'time', 'id'], keep='last')
            df_average_speed = df_id['distance'].sum() / df_id['distance'].count()
            df_speed = df_speed.append({'id': node_id, 'averageSpeed': df_average_speed}, ignore_index=True)

        x = df_speed.id.values
        y = df_speed.averageSpeed.values

        # Scatter plot
        ax.scatter(x, y, marker='.', color='red')
        plt.xticks(np.arange(min(x), max(x) + 1, 1))
        fig = ax.get_figure()
        fig.savefig(OUT_PATH_SPEED + f"/average_speed_scatter{z}.png")

        # Boxplot
        boxplot = df.boxplot(by='id', column='distance', grid=False, showfliers=False)
        plt.yticks(np.arange(0, 20, 1))
        fig2 = boxplot.get_figure()
        fig2.savefig(OUT_PATH_SPEED + f"/average_speed_boxplot_{z}.png")


def average_speed_per_pedestrian():
    df = read_position_files()
    df['distance'] = 0
    df = df.groupby(["time", "id"], as_index=False).agg({"x": np.mean, "y": np.mean})

    fig, ax = plt.subplots(1, 1)

    x = df.id.unique()
    df_speed = pd.DataFrame(columns=['id', 'averageSpeed'])
    for a in range(min(x), max(x) + 1):
        node_id = 0
        distance = []
        df_id = df[df['id'] == a]
        for i in range(len(df_id)):
            node_id = a
            if i == 0:
                distance.append(580 - df_id.iloc[i].y)
            elif i == len(df_id) - 1:
                distance.append(df_id.iloc[i - 1].y - df_id.iloc[i].y)
                break
            else:
                row1, row2 = df_id.iloc[i - 1], df_id.iloc[i]
                distance.append(row1.y - row2.y)
        df_id['distance'] = distance
        df = pd.concat([df, df_id]).drop_duplicates(['x', 'y', 'time', 'id'], keep='last')
        average_speed = df_id['distance'].sum() / df_id['distance'].count()
        df_speed = df_speed.append({'id': node_id, 'averageSpeed': average_speed}, ignore_index=True)

    x = df_speed.id.values
    y = df_speed.averageSpeed.values

    # Scatter plot
    ax.scatter(x, y, marker='.', color='red')
    plt.xticks(np.arange(min(x), max(x) + 1, 1))
    fig = ax.get_figure()
    fig.savefig(OUT_PATH_POSITION + '/average_speed_scatter.png')

    # Boxplot
    boxplot = df.boxplot(by='id', column='distance', grid=False, showfliers=False)
    plt.yticks(np.arange(0, 20, 1))
    fig2 = boxplot.get_figure()
    fig2.savefig(OUT_PATH_POSITION + '/average_speed_boxplot.png')


def calculate_distance_between_pedestrians_and_enb():
    distance = []
    df = read_position_files()

    run_ids = df.runId.unique()
    run_ids.sort()

    for run in run_ids:
        print(f"Aggregating run {run}")
        df_filtered = df[df["runId"] == run]
        for i in range(df["time"].min().astype(np.int), df["time"].max().astype(np.int), 10):
            df_tmp = df_filtered[df_filtered["time"] == i]
            id_list = df_tmp.id.unique()
            id_list.sort()
            for index, id in enumerate(id_list):
                x1 = df_tmp[df_tmp["id"] == id]["x"].values[0]
                y1 = df_tmp[df_tmp["id"] == id]["y"].values[0]
                enb_x = 300
                enb_y = 300
                for k in range(index + 1, len(df_tmp.id), 1):
                    x2 = df_tmp[df_tmp["id"] == id_list[k]]["x"].values[0]
                    y2 = df_tmp[df_tmp["id"] == id_list[k]]["y"].values[0]
                    dist = math.sqrt(abs((x2 - x1) ** 2 - (y2 - y1) ** 2))
                    dist_enb = math.sqrt(abs((enb_x - x1) ** 2 - (enb_y - y1) ** 2))
                    distance.append([i, dist, dist_enb])

    df = pd.DataFrame(distance)
    df.columns = ["time", "distance", "distance_enb"]

    tmp = []
    for x in range(df["time"].min(), df["time"].max() + 10, 10):
        df_tmp = df[df["time"] == x]
        tmp.append([x, "{0:,.2f}".format(df_tmp['distance'].mean()), "{0:,.2f}".format(df_tmp['distance_enb'].mean())])

    df_all = pd.DataFrame(tmp)
    df_all = df_all.astype(float)
    df_all.columns = ["time", "distance", "distance_enb"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.plot("time", "distance", data=df_all, label="Distance among pedestrians", color="green")
    ax.plot("time", "distance_enb", data=df_all, label="Distance Pedestrian/eNB", color="purple")
    ax.set_xlabel("Time in [s]")
    ax.set_ylabel("Distance")
    ax.set_ylim(0, 160)
    ax.set_xlim(0, 800)

    # Ped Count
    df_reduced = mean_pedestrian_count()

    ax1 = ax.twinx()
    ax1.plot("time", "pedCount", color="orange", label="Mean Pedestrian Count", data=df_reduced)
    ax1.set_ylabel("Mean Pedestrian Count")
    ax1.set_ylim(0, 33)
    ax1.set_yticks(np.arange(0, 33, 3))

    lines_1, labels1 = ax.get_legend_handles_labels()
    lines_2, labels2 = ax1.get_legend_handles_labels()

    lines = lines_1 + lines_2
    labels = labels1 + labels2

    ax.legend(lines, labels, loc=0)

    plot_vspans(ax)

    fig.savefig(OUT_PATH_DISTANCE + f"/distance_mean_to_enb.png")

    return ax


def main():
    log.basicConfig(stream=sys.stdout, level=log.WARNING)
    calculate_distance_between_pedestrians_and_enb()
    scatter_pedestrian_positions()
    scatter_plot_min_max()
    # average_speed()  # ???
    average_speed_per_pedestrian()
    average_speed_per_pedestrian_per_run()


if __name__ == "__main__":
    main()
