import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns
import numpy as np
import os, fnmatch

from analysis import ROOT, RUN, plot_vspans

# ROOT = "/home/mkilian/repos/crownet/analysis/roveranalyzer/data"
# RUN = "sumoBottleneck"
PATH_ROOT = f"{ROOT}/{RUN}"
OUTPUT_PATH = f"{PATH_ROOT}/out/position"
SPEED_OUTPUT_PATH = f"{PATH_ROOT}/out/speed"
PAINT_INTERVALS = True

def read_position_files():
    position_list = find('positions.txt', PATH_ROOT)
    position_list.sort()

    df_all = []
    columns = ["time", "id", "x", "y"]
    for i in range(len(position_list)):
        tmp = pd.read_csv(position_list[i], delimiter="\t")
        tmp.columns = columns
        tmp['runId'] = i
        df_all.append(tmp)

    return pd.concat(df_all, ignore_index=True)


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def scatter_pedestrian_positions():
    time = 50
    to = 70
    runId = 3
    df = read_position_files()
    for u in range(time, to + 10, 10):
        df_filter = df[df['time'] == u]
        df_filter = df_filter[df_filter['runId'] == runId]
        fig, ax = plt.subplots(1, 1)
        x = df_filter.x.values
        y = df_filter.y.values
        ax.scatter(x, y, label='pedestrian', marker='.', color='red')
        # ax.set_ylim(320, 460)
        ax.set_xlabel("x-coordinate")
        ax.set_ylabel("y-coordinate")
        plt.title("Pedestrian positions at time step " + str(u))
        ax.legend()
        fig = ax.get_figure()
        fig.savefig(OUTPUT_PATH + '/position' + str(u) + '_' + str(runId) + '.png')


def scatter_plot_min_max():
    df = read_position_files()
    df = df.drop(columns="id")
    df_time = df.groupby("time").agg({"y": [np.max, np.min], "x": [np.max, np.min]})
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    x = df_time.index.values
    y_min = df_time.loc[:, ('y', 'amin')]
    y_max = df_time.loc[:, ('y', 'amax')]

    ax.scatter(x, y_max, label='max y coord', marker='_', color='red')
    ax.scatter(x, y_min, label='min y coord', marker='_', color='green')
    ax = plot_vspans(ax)

    ax.set_xlabel("time in [s]")
    # ax.set_ylim(0, 600)
    ax.set_ylabel("y-coordinates")
    ax.legend()

    plt.xticks(np.arange(0, 1000, 100))
    # plt.grid()

    fig = ax.get_figure()
    fig.savefig(OUTPUT_PATH + '/min_max.png')


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
        elif i == len(df_id)-1:
            distance.append(df_id.iloc[i-1].y - df_id.iloc[i].y)
            break
        else:
            row1, row2 = df_id.iloc[i-1], df_id.iloc[i]
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
            average_speed = df_id['distance'].sum() / df_id['distance'].count()
            df_speed = df_speed.append({'id': node_id, 'averageSpeed': average_speed}, ignore_index=True)

        x = df_speed.id.values
        y = df_speed.averageSpeed.values

        # Scatter plot
        ax.scatter(x, y, marker='.', color='red')
        plt.xticks(np.arange(min(x), max(x) + 1, 1))
        fig = ax.get_figure()
        fig.savefig(SPEED_OUTPUT_PATH + f"/average_speed_scatter{z}.png")

        # Boxplot
        boxplot = df.boxplot(by='id', column='distance', grid=False, showfliers=False)
        plt.yticks(np.arange(0, 20, 1))
        fig2 = boxplot.get_figure()
        fig2.savefig(SPEED_OUTPUT_PATH + f"/average_speed_boxplot_{z}.png")


def average_speed_per_pedestrian():
    df = read_position_files()
    df['distance'] = 0
    df = df.groupby(["time", "id"], as_index=False).agg({"x": np.mean, "y": np.mean})

    fig, ax = plt.subplots(1, 1)

    x = df.id.unique()
    df_speed = pd.DataFrame(columns=['id', 'averageSpeed'])
    for a in range(min(x), max(x)+1):
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
    plt.xticks(np.arange(min(x), max(x)+1, 1))
    fig = ax.get_figure()
    fig.savefig(OUTPUT_PATH + '/average_speed_scatter.png')

    # Boxplot
    boxplot = df.boxplot(by='id', column='distance', grid=False, showfliers=False)
    plt.yticks(np.arange(0, 20, 1))
    fig2 = boxplot.get_figure()
    fig2.savefig(OUTPUT_PATH + '/average_speed_boxplot.png')


if __name__ == "__main__":
    # scatter_pedestrian_positions()
    scatter_plot_min_max()
    # average_speed_per_pedestrian()
    # average_speed_per_pedestrian_per_run()
