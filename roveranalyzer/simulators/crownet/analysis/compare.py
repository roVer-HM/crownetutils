import math
import os
from enum import Enum
from typing import List, Union, Tuple, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from roveranalyzer.simulators.opp.scave import ScaveTool
from roveranalyzer.simulators.opp.utils import Simulation


class How(Enum):
    """Enum for different data aggregation functions"""

    mean = "mean"
    min = "min"
    max = "max"
    sum = "sum"
    first = "first"


def _read_position_data(sim: Simulation, as_tuples: bool = False) -> pd.DataFrame:
    """Reads positional data from a simulations .vec file into a Dataframe.
    For each second long interval reads the first position recorded in that interval.

    :param sim: The simulation object containing the path to the output files
    :param as_tuples: If enabled, concatenates posX and posY into tuples
    :return: Dataframe containing the positional data of all pNodes

    eg. with as_tuples=False

    |   time |   pNode[0].posX.value |   pNode[0].posY.value |   pNode[10].posX.value | ...
    |-------:|----------------------:|----------------------:|-----------------------:|
    |      0 |               182.922 |               50.1919 |                185.835 |
    |      1 |               182.326 |               50.9673 |                185.58  |
    |      2 |               181.84  |               52.1457 |                185.385 |
    |      3 |               181.518 |               53.5683 |                185.705 |
        .
        .
        .

     eg. with as_tuples=True

    |time| pNode[0].pos                       | pNode[10].pos                      | pNode[11].pos                     |
    |---:|:-----------------------------------|:-----------------------------------|:----------------------------------|
    |  0 | (182.92204028562, 50.191903876669) | (185.83502131957, 48.695005375431) | (186.92110845269, 49.354618027868)|
    |  1 | (182.32603278673, 50.967296558008) | (185.5798519689, 49.071483400592)  | (187.33921690538, 49.354618027868)|
    |  2 | (181.84010222571, 52.145725751748) | (185.38509323013, 50.082316825988) | (187.90848474818, 49.975954651257)|
    |  3 | (181.51821225762, 53.568250381364) | (185.70459880849, 50.874390696694) | (189.010190339, 51.384198632296)  |
        .
        .
        .

    """
    module = "*World.pNode[*]"
    vector_x = "posX:vector"
    vector_y = "posY:vector"
    df = _aggregate_vectors_from_simulation(
        sim, module, [vector_x, vector_y], How.first
    )

    if not as_tuples:
        return df

    df_tuples = pd.DataFrame()
    for i in range(int(len(df.columns) / 2)):
        column_name = f"{df.columns[i * 2].split('.')[0]}.pos"
        df_tuples[column_name] = list(
            zip(df[df.columns[i * 2]], df[df.columns[i * 2 + 1]])
        )
    df_tuples.index.name = "time"
    return df_tuples


def plot_pnode_positions(
    sim: Simulation,
    start: int,
    end: int,
    interval: int,
    border_margin: int = 5,
    combine_plots=True,
    xlim=None,
    ylim=None,
):
    """
    For a simulation: Returns multiple comparable matplotlib figures plotting pNode positions.
    :param sim: Simulation object containing output path
    :param start: simulation time in seconds for the first plot
    :param end: simulation time after which no plots will be generated
    :param interval: interval in seconds between plots
    :param border_margin: border margin around plots for readability
    :param combine_plots: if true, combines all axes into one figure
    :param ylim: custom ylim value for all axes
    :param xlim: custom xlim value for all axes
    :return: (fig, ax) or list of (fig, ax) Tuples as returned by pylot.subplots(), one for each figure
    """
    df = _read_position_data(sim)
    dfs_plot = []
    times = list(range(start, end + 1, interval))
    for i in times:
        df_plot = pd.DataFrame()
        df_plot["x"] = df.filter(like="posX", axis=1)[df.index >= i].iloc[0].values
        df_plot["y"] = df.filter(like="posY", axis=1)[df.index >= i].iloc[0].values
        dfs_plot.append(df_plot)

    if not ylim:
        ylim = (
            min([df["y"].min() for df in dfs_plot]) - border_margin,
            max([df["y"].max() for df in dfs_plot]) + border_margin,
        )
    if not xlim:
        xlim = (
            min([df["x"].min() for df in dfs_plot]) - border_margin,
            max([df["x"].max() for df in dfs_plot]) + border_margin,
        )

    if combine_plots:
        fig, ax = plt.subplots(1, len(dfs_plot))
        figs = [(fig, a) for a in ax]
        ret = (fig, ax)
    else:
        figs = [plt.subplots() for _ in range(len(dfs_plot))]
        ret = figs

    for i, df_plot in enumerate(dfs_plot):
        fig, ax = figs[i]
        df_plot.plot.scatter(x="x", y="y", ax=ax, xlim=xlim, ylim=ylim, style="o", s=3)
        ax.set_aspect("equal", "box")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.invert_yaxis()
        fig.suptitle(f"pNode positions for {sim.desc}")
        fig.supylabel("y [m]")
        fig.supxlabel("x [m]")
        ax.set_title(f"at {times[i]}s")
        ax.tick_params(axis="x", labelrotation=30)

    return ret


def distance_plot_mean(
    sims: List[Simulation], title: str = "", cutoff: float = 0, fig=None, ax=None
) -> Any:
    """Generates a plot displaying the average distance between nodes and enb, average distance in between nodes and
    the average active node count, averaged over all simulations (runs) provided.

    :param sims: The list of simulations to be averaged (e.g. multiple runs of a simulation)
    :param title: The title of the plot
    :param cutoff: if >= 1, will cutoff data as soon as less than <cutoff> simulations contain data
                    if < 1 will cutoff data as soon as the ratio of simulations
                    still containing data is less than <cutoff>
    :param fig: Figure to plot on, if None, a new one will be created
    :param ax: Axes to plot on, if None, a new one will be created
    :return: fig, ax as returned by pyplot.subplots()
    """
    sca_paths = [
        os.path.join(sims[0].path, f)
        for f in os.listdir(sims[0].path)
        if f.endswith(".sca")
    ]
    df_sca = ScaveTool().load_df_from_scave(sca_paths[0])
    enb = (
        df_sca.loc[df_sca.attrname == "*.eNB[0].mobility.initialX", "attrvalue"].values[
            0
        ],
        df_sca.loc[df_sca.attrname == "*.eNB[0].mobility.initialY", "attrvalue"].values[
            0
        ],
    )
    enb = (
        float("".join(c for c in enb[0] if (c.isdigit() or c == "."))),
        float("".join(c for c in enb[1] if (c.isdigit() or c == "."))),
    )
    dfs = [_read_position_data(sim, True) for sim in sims]
    dfs_dist_nodes = [_distances_between_nodes(df) for df in dfs]

    dfs_dist_enb = [_distance_between_nodes_enb(df, enb) for df in dfs]
    df_dist_nodes = _average_sim_data(
        dfs_dist_nodes, active_vectors_column=False, cutoff=cutoff
    )
    df_dist_enb = _average_sim_data(dfs_dist_enb, cutoff=cutoff)
    fig, ax = _plot_comparison_from_dfs(
        [df_dist_enb, df_dist_nodes],
        df_identifiers=["enb-node", "node-node"],
        vector_description="avg. distance",
        unit="[m]",
        rolling_only=True,
        title=title,
        fig=fig,
        ax=ax,
    )
    return fig, ax


def _plot_comparison_from_dfs(
    dfs: List[pd.DataFrame],
    df_identifiers: List[str],
    vector_description: str,
    unit: str,
    rolling_only=False,
    title: str = "",
    fig=None,
    ax=None,
) -> Any:
    """Compares two dataframes containing aggregated vector data of simulations (e.g. as returned by the
        aggregate_vectors_from_simulation() or average_sim_data() functions.

    :param dfs: dataframes containing the data to be compared. Eg as returned by the
            aggregate_vectors() / average_sim_data() function:
    :param df_identifiers: identifiers/names of the dataframes/data
    :param vector_description: description of the vector being compared
    :param unit: unit of the vector data
    :param rolling_only: if ture, will plot only the rolling average
    :param title: The title of the plot
    :param fig: figure containing the axes to be used, if none a new will be created
    :param ax: axes to be used
    :return: fig, ax as returned by pyplot.subplots()
    """
    colors_1 = ["darkgreen", "darkblue", "darkred", "darkorange"]
    # colors_node_count = ["limegreen", "royalblue", "firebrick", "gold"]
    colors_node_count = ["black", "red", "purple", "brown"]
    colors_3 = ["yellowgreen", "cornflowerblue", "tomato", "yellow"]
    linestyles = ["-", "--", "."]

    if not fig:
        fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    columns = [column for columns in [df.columns for df in dfs] for column in columns]
    plot_active_nodes = True if "active_pNodes" in columns else False
    if plot_active_nodes:
        ax2 = ax.twinx()
        ax2.set_ylabel("active pNodes")

    for i, df in enumerate(dfs):
        if not rolling_only:
            df["value"].dropna(how="all").plot(
                ax=ax,
                label=f"{df_identifiers[i]} - {vector_description}",
                color=colors_3[i],
                style="-",
            )
        df["value"].dropna(how="all").rolling(10, center=True).mean().plot(
            ax=ax,
            label=f"{df_identifiers[i]}" f" - {vector_description}" f" - SMA",
            color=colors_1[i],
            style="-",
        )
        if plot_active_nodes and "active_pNodes" in df.columns:
            df["active_pNodes"].dropna(how="all").plot(
                ax=ax2,
                label=f"{df_identifiers[i]} - active pNodes",
                color=colors_node_count[i],
                style="-",
            )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"{vector_description} {unit}")
    h1, l1 = ax.get_legend_handles_labels()
    if plot_active_nodes:
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right")
    else:
        ax.legend(h1, l1, loc="upper right")
    fig.suptitle(title)
    if plot_active_nodes:
        ax = [ax, ax2]
    return fig, ax


def _distance_between_nodes_enb(
    df: pd.DataFrame, enb: Tuple[float, float]
) -> pd.DataFrame:
    """For a dataframe containing positional data of a simulations nodes. Returns d dataframe containing the distance
    to the eNB for each node for each timeframe

    :param df: DataFrame containing positional data in tuple form (e.g. as returned by read_position_data())
    :param enb: the coordinates of the eNB to which the distances should be calculated
    :return: the DataFrame containing the distance of each node to the enb over time
    |   time |   pNode[0].pos |   pNode[10].pos |   pNode[11].pos |   pNode[12].pos |  ...
    |-------:|---------------:|----------------:|----------------:|----------------:|
    |      0 |        277.101 |         277.101 |         277.101 |         277.101 |
    |      1 |        275.796 |         277.101 |         276.991 |         277.046 |
    |      2 |        274.505 |         276.373 |         276.468 |         276.881 |
    |      3 |        273.251 |         275.328 |         275.29  |         276.118 |
            .               .               .                  .                .
            .               .               .                  .                .
            .               .               .                  .                .
    """
    res = df.apply(_apply_distance_between_nodes_enb, raw=True, axis=1, enb=enb)
    return res


def _apply_distance_between_nodes_enb(
    ndarray: np.ndarray, enb: Tuple[float, float]
) -> List[float]:
    """Calculates the average distance between nodes for a row of positional data (in tuple form).
    Meant to be used as argument for Dataframe.apply()

    :param ndarray: rows of a Dataframe created by read_position_data() with the as_tuples option enabled
    :param enb: the coordinates of the eNB the distance shall be calculated to
    :return: row containing the average distance for each node to the enb
    """
    res = []
    node_count = len(ndarray)
    for i in range(node_count):
        node = ndarray[i]
        if math.isnan(node[0]):
            res.append(np.nan)
            continue
        res.append(math.dist(node, enb))
    return res


def _distances_between_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """For a dataframe containing positional data of a simulations nodes. Returns d dataframe containing the average distance
    for each node to all other nodes for each time frame

    :param df: DataFrame containing positional data in tuple form (e.g. as returned by read_position_data())
    :return: the DataFrame containing the average distance of each node to all other nodes over time

    |   time |   pNode[0].pos |   pNode[10].pos |   pNode[11].pos |   pNode[12].pos | .....
    |-------:|---------------:|----------------:|----------------:|----------------:|
    |      0 |        0       |        0        |        0        |        0        |
    |      1 |        1.42488 |        0.185838 |        0.271876 |        0.179354 |
    |      2 |        2.70006 |        0.945283 |        0.741382 |        0.553128 |
    |      3 |        3.78109 |        1.85041  |        1.62499  |        1.13149  |
            .               .               .                  .                .
            .               .               .                  .                .
            .               .               .                  .                .
    """
    res = df.apply(_apply_distances_between_nodes, raw=True, axis=1)
    return res


def _apply_distances_between_nodes(ndarray: np.ndarray) -> List[float]:
    """Calculates the average distance between nodes for a row of positional data (in tuple form).
    Meant to be used as argument for Dataframe.apply()

    :param ndarray: rows of a Dataframe created by read_position_data() with the as_tuples option enabled
    :return: row containing the average distance for each node to all other nodes.
    """
    res = []
    node_count = len(ndarray)
    active_node_count = len([t for t in ndarray if not math.isnan(t[0])])
    if active_node_count == 1:
        return [np.nan] * node_count
    for i in range(node_count):
        node = ndarray[i]
        if math.isnan(node[0]):
            res.append(np.nan)
            continue
        sum_dist = 0.0
        for k in range(node_count):
            other = ndarray[k]
            if k == i or math.isnan(other[0]):
                continue
            sum_dist += math.dist(node, other)
        res.append(sum_dist / (active_node_count - 1))
    return res


def _aggregate_vectors(
    df: pd.DataFrame, how: How = How.mean, interval: int = 1
) -> pd.DataFrame:
    """
    This function will group a DataFrame containing time/value vectors into bins of the specified size and apply the
    chosen function to the values contained in the bins to summarize them.

    :param df: DataFrame containing the time/value vectors,
                e.g. as returned by the Opp.normalize_vectors(axis=1) method:


    |    |   pNode[8].rcvdPkLifetime.time |   pNode[8].rcvdPkLifetime.value |   pNode[0].rcvdPkLifetime.time |  ...
    |---:|-------------------------------:|--------------------------------:|-------------------------------:|---
    |  0 |                        1.12113 |                        0.021134 |                        1.12213 |  ...
    |  1 |                        1.12213 |                        0.022134 |                        1.12413 |  ...
    |  2 |                        1.12413 |                        0.024134 |                        1.12613 |  ...
    |  3 |                        1.12613 |                        0.026134 |                        1.12913 |  ...
    |  4 |                        1.12913 |                        0.029134 |                        1.13413 |  ...

    :param how: Determines the way of aggregating the metrics for each interval
    :param interval: the interval in seconds over which metrics are aggregated
    :return: A DataFrame, whose index represents the start time of each bin, the rows containing the
             values calculated for each bin/vector. E.g:



        time |   pNode[0].rcvdPkLifetime.value |   pNode[1].rcvdPkLifetime.value |   pNode[2].rcvdPkLifetime.value | ...
    |-------:|--------------------------------:|--------------------------------:|--------------------------------:|---
    |      0 |                     nan         |                     nan         |                     nan         | ...
    |      1 |                       0.0405024 |                       0.0415784 |                       0.0415784 | ...
    |      2 |                       0.0379911 |                       0.0392054 |                       0.038134  | ...
    |      3 |                       0.0429197 |                       0.0402769 |                       0.0396673 | ...
    |      4 |                       0.042134  |                       0.0443483 |                       0.0432054 | ...
    """
    df_res = None
    num_vectors = int(len(df.columns) / 2)
    for k in range(num_vectors):
        df_vector: pd.DataFrame = df.iloc[:, [k * 2, k * 2 + 1]]
        # group values in second long intervals
        df_tmp_group_by = df_vector.groupby(
            pd.cut(
                df_vector.iloc[:, 0],
                range(0, int(df_vector.iloc[:, 0].max()) + 1, interval),
            )
        )
        # apply to groups
        if how == How.sum:
            df_vector = df_tmp_group_by.sum()
        elif how == How.min:
            df_vector = df_tmp_group_by.min()
        elif how == How.max:
            df_vector = df_tmp_group_by.max()
        elif how == How.mean:
            df_vector = df_tmp_group_by.mean()
        elif how == How.first:
            df_vector = df_tmp_group_by.first()
        else:
            raise ValueError(f"Value '{how}' not recognized for 'how' kwarg")

        df_vector.drop(df_vector.columns[0], axis=1, inplace=True)
        df_vector.index.name = "time"  # index
        if df_res is None:
            df_res = df_vector
        else:
            df_res = df_res.join(df_vector, how="outer")

    df_res.reset_index(drop=True, inplace=True)
    df_res.index.name = "time"
    return df_res


def _active_vectors(df: pd.DataFrame) -> pd.Series:
    """For a DataFrame containing aggregated vector data (e.g. as returned by the aggregate_vectors() function):
        Returns a pd.Series over time representing how many of these vectors were still recording data at the time
        or a later time in the simulation

    :param df: the DataFrame (e.g. as returned by the aggregate_vectors() function)
    :return: the Series over time representing how many of these vectors were still recording data at the time
        or a later time in the simulation
    """
    df_t = df.copy(deep=True)
    for column in df_t.columns:
        last = df_t[column].last_valid_index()
        # first = df_t[column].first_valid_index()
        df_t[column].loc[:last] = df_t[column].loc[:last].ffill()

    active_nodes = df_t.apply(func=lambda x: sum(~np.isnan(x)), raw=True, axis=1)
    return active_nodes


def _average_sim_data(
    dfs: List[pd.DataFrame],
    cutoff: float = 0,
    active_vectors_column=True,
    active_sims_column=True,
) -> pd.DataFrame:
    """Will average vector data over time from one or more simulations.

    :param dfs: Dataframes containing aggregated simulation data (e.g. as returned by the aggregate_vectors() function)
    :param cutoff: if >= 1, will cutoff data as soon as less than <cutoff> simulations contain data
                    if < 1 will cutoff data as soon as the ratio of simulations
                    still containing data is less than <cutoff>
    :param active_vectors_column: whether the result should contain a column with the average active vectors
                                    of all simulations still containing data
    :param active_sims_column: whether the result should contain a column with the number of simulations
                                still containing data
    :return: dataframe containing aggreated values averaged over all simulations
        |   time |   value |   active_pNodes |   active_sims |
        |-------:|--------:|----------------:|--------------:|
        |      0 | 277.101 |        30       |            10 |
        |      1 | 276.91  |        30       |            10 |
        |      2 | 276.64  |        30       |            10 |
        |      3 | 276.18  |        30       |            10 |
        |      4 | 275.576 |        30       |            10 |
        |      5 | 274.806 |        30       |            10 |
               .      .              .                     .
               .      .              .                     .
               .      .              .                     .

    """
    df_means = None
    df_node_counts = None
    for i, df in enumerate(dfs):
        mean_values = df.mean(axis=1)
        active_nodes = _active_vectors(df)
        df[f"mean_value_sim_{i}"] = mean_values
        df[f"active_pNodes_sim_{i}"] = active_nodes

        if df_means is None:
            df_means = df[[f"mean_value_sim_{i}"]].copy()
            df_node_counts = df[[f"active_pNodes_sim_{i}"]].copy()
        else:
            df_means = df_means.join(df[[f"mean_value_sim_{i}"]], how="outer")
            df_node_counts = df_node_counts.join(
                df[[f"active_pNodes_sim_{i}"]], how="outer"
            )

    df_node_counts["value"] = df_node_counts.mean(axis=1)
    means = df_means.mean(axis=1)
    active_sims = _active_vectors(df_means)
    df_means["value"] = means
    df_means["active_sims"] = active_sims
    df_means = df_means.drop(df_means[df_means.active_sims == 0].index)
    df_dict = {f"value": df_means["value"]}
    if active_vectors_column:
        df_dict["active_pNodes"] = df_node_counts["value"]
    if active_sims_column:
        df_dict["active_sims"] = df_means["active_sims"]
    res = pd.DataFrame(df_dict).dropna(how="any")

    if cutoff <= 0:
        return res
    if cutoff < 1:
        res = res.drop(res[(res.active_sims / res.active_sims.max()) < cutoff].index)
    else:
        res = res.drop(res[res.active_sims < cutoff].index)
    return res


def _aggregate_vectors_from_simulation(
    sim: Simulation,
    module: str,
    vector_names: Union[str, List[str]],
    how: How = How.mean,
) -> pd.DataFrame:
    """This function will read vector data of the given simulation and aggregate it with the given method over
        one second long intervals.


    :param sim: The simulation whose data will be aggregated
    :param module: name of the module of the vectors to be read
    :param vector_names: names of the vectors to be read
    :param how: Determines the way of aggregating the value for each time bin
    :return: a DataFrame containing the aggregated data (as returned by the aggregate_vectors() function)
    """
    if isinstance(vector_names, str):
        vector_names = [vector_names]
    sfilter = (
        ScaveTool().filter_builder().module(module).AND().gOpen().name(vector_names[0])
    )
    for vector_name in vector_names[1:]:
        sfilter.OR().name(vector_name)
    sfilter.gClose()
    vec_paths = [
        os.path.join(sim.path, f) for f in os.listdir(sim.path) if f.endswith(".vec")
    ]
    df_sim = ScaveTool().load_df_from_scave(vec_paths[0], sfilter)
    df_sim = df_sim.opp.filter().vector().normalize_vectors(axis=1)
    df_data = _aggregate_vectors(df_sim, how)
    df_data = df_data.reindex(sorted(df_data.columns), axis=1)
    return df_data
