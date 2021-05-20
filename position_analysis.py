import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, fnmatch

ROOT = "/home/max/Git/crownet/analysis/roveranalyzer/data"
RUN = "vadereBase_Simple"
PATH_ROOT = f"{ROOT}/{RUN}"
OUTPUT_PATH = f"{PATH_ROOT}/out/position"

def read_position_files():
    # path = f"sumoBase_Simple\\sumoBase_1934\\positions.txt"
    position_list = find('positions.txt', PATH_ROOT)

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


def scatter_plot_positions():
    time = 250
    runId = 3
    threshold = 20
    df = read_position_files()
    # df_time = df.groupby(["time", "id"], as_index=False).agg({"y": np.mean, "x": np.mean})
    df_filter = df[df['time'] == time]
    df_filter = df_filter[df_filter['runId'] == runId]
    fig, ax = plt.subplots(1, 1)
    x = df_filter.x.values
    y = df_filter.y.values
    ax.scatter(x, y, label='pedestrian', marker='.', color='red')
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("y-coordinate")
    ax.set_ylim(df_filter['y'].min() - threshold, df_filter['y'].max() + threshold)
    plt.title("Pedestrian positions at time step " + str(time))
    ax.legend()
    fig = ax.get_figure()
    fig.savefig(OUTPUT_PATH + '/position' + str(time) + '_' + str(runId) + '.png')


def scatter_plot_min_max():
    # df: pd.DataFrame = pd.read_csv(path + type, index_col="time")
    df = read_position_files()
    df = df.drop(columns="id")
    df_time = df.groupby("time").agg({"y": [np.max, np.min], "x": [np.max, np.min]})
    fig, ax = plt.subplots(1, 1)

    x = df_time.index.values
    y_min = df_time.loc[:, ('y', 'amin')]
    y_max = df_time.loc[:, ('y', 'amax')]
    ax.scatter(x, y_max, label='max y coord', marker='_', color='red')
    ax.scatter(x, y_min, label='min y coord', marker='_', color='green')
    ax.set_xlabel("time in [s]")
    ax.set_ylabel("y-coordinates")
    ax.set_xlim(0, 500)
    ax.legend()
    plt.xticks(np.arange(0, 550, 50))
    plt.yticks(np.arange(0, 600, 50))
    plt.grid()
    fig = ax.get_figure()
    fig.savefig(OUTPUT_PATH + '/min_max.png')
    # fig.show()


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
    scatter_plot_positions()
    # scatter_plot_min_max()
    # average_speed_per_pedestrian()
