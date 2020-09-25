import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from roveranalyzer.oppanalyzer.rover_analysis import Opp
from roveranalyzer.oppanalyzer.utils import (
    Config,
    OppDataProvider,
    ScaveTool,
    build_time_series,
    parse_cmdEnv_outout,
)
from roveranalyzer.uitls import Timer


def stacked_bar(
    ax,
    data,
    data_sum,
    c_map="PuBuGn",
    c_interval=[0.2, 0.7],
    bar_width=0.4,
    percentage=False,
    horizontal=False,
    with_table=False,
    tbl_bbox=(0.01, -0.25, 0.99, 0.2),
):
    index = np.arange(data.shape[1]) + 0.3
    offset = np.zeros(data.shape[1])
    c = plt.cm.get_cmap(c_map)(np.linspace(c_interval[0], c_interval[1], data.shape[0]))
    c = np.append(c, [[1.0, 1.0, 1.0, 1.0]], axis=0)

    if horizontal:
        ax.set_yticklabels(list(data.columns))
        ax.set_yticks(index)
    else:
        ax.set_xticklabels(list(data.columns))
        ax.set_xticks(index)

    cell_text = []
    for row in range(data.shape[0]):
        if horizontal:
            ax.barh(
                index, data.iloc[row, :], bar_width, left=offset, color=c[row],
            )
        else:
            ax.bar(
                index, data.iloc[row, :], bar_width, bottom=offset, color=c[row],
            )
        offset = offset + data.iloc[row, :]  # data.loc[:, "value"]
        if with_table:
            cell_text.append([f"{x}" for x in data.iloc[row, :]])

    if with_table:
        cell_text.append([f"{x}" for x in data_sum.iloc[0, :]])
        row_lables = list(data.index)
        row_lables.extend(list(data_sum.index))

        ax.table(
            cellText=cell_text,
            rowLabels=row_lables,
            rowColours=c,
            colLabels=data.columns,
            loc="bottom",
            bbox=list(tbl_bbox),
        )
    return ax


def mac_drop_ration(_df: pd.DataFrame):
    """
    rx_drop_ration = sum(rxDrop)/sum(rx)
    tx_drop_ration = sum(txDrop)/sum(tx)

    Note: packetDropOther:count is added to rxDrop. #TODO check code if this is correct.
    """
    stats = {
        "rxDrop": [
            "packetDropDuplicateDetected:count",
            "packetDropIncorrectlyReceived:count",
            "packetDropNotAddressedToUs:count",
            "packetDropOther:count",
        ],
        "rx": ["packetReceivedFromLower:count"],
        "txDrop": [
            "packetDropQueueOverflow:count",
            "packetDropRetryLimitReached:count",
        ],
        "tx": ["packetReceivedFromUpper:count"],
    }
    all_stats = list(stats.values())
    all_stats = list(itertools.chain.from_iterable(all_stats))
    _df = (
        _df.opp.filter()
        .scalar()
        .module_regex(".*mac$")
        .name_in(all_stats)
        .apply(columns=("name", "value", "module", "run"), copy=True)
    )
    runs = {run: idx for idx, run in enumerate(_df.run.unique())}
    _df["run"] = _df.run.apply(lambda x: runs[x])
    _df["module"] = _df.module.apply(lambda x: Opp.module_path(x, 1) + ".mac")
    _df = _df.pivot_table(index=["run", "module"], columns=["name"], values="value")
    _df["rx_drop_ration"] = _df.loc[:, stats["rxDrop"]].sum(axis=1) / _df.loc[
        :, stats["rx"]
    ].sum(axis=1)
    _df["tx_drop_ration"] = _df.loc[:, stats["txDrop"]].sum(axis=1) / _df.loc[
        :, stats["tx"]
    ].sum(axis=1)
    _df_network = _df.groupby("run").sum()
    _df_network["module"] = "network"
    _df_network.set_index(keys=["module"], append=True, inplace=True)
    _df_network["rx_drop_ration"] = _df_network.loc[:, stats["rxDrop"]].sum(
        axis=1
    ) / _df_network.loc[:, stats["rx"]].sum(axis=1)
    _df_network["tx_drop_ration"] = _df_network.loc[:, stats["txDrop"]].sum(
        axis=1
    ) / _df_network.loc[:, stats["tx"]].sum(axis=1)
    _df = _df.append(_df_network)

    return {"data": _df, "run_keys": runs}


def mac_drop_ration_bar_chart(_df: pd.DataFrame, run, use_stat="mac_pkt_drop"):

    stats = {
        "mac_pkt_drop": {
            "module_filter": ".*hostMobile.*\.mac$",
            "total": "packetDrop:count",
            "data": [
                "packetDropDuplicateDetected:count",  # rx
                "packetDropIncorrectlyReceived:count",  # rx
                "packetDropNotAddressedToUs:count",  # rx
                "packetDropOther:count",  # rx
                "packetDropQueueOverflow:count",  # tx
                "packetDropRetryLimitReached:count",  # tx
            ],
        },
        "mac_dcf_rx": {
            "module_filter": ".*hostMobile.*\.mac\.dcf$",
            "total": "packetReceivedFromPeer:count",
            "data": [
                "packetReceivedFromPeerBroadcast:count",
                "packetReceivedFromPeerMulticast:count",
                "packetReceivedFromPeerMulticast:count",
            ],
        },
        "mac_dcf_rx_retry": {
            "module_filter": ".*hostMobile.*\.mac\.dcf$",
            "total": "packetReceivedFromPeer:count",
            "data": [
                "packetReceivedFromPeerWithoutRetry:count",
                "packetReceivedFromPeerWithRetry:count",
            ],
        },
        "mac_dcf_tx": {
            "module_filter": ".*hostMobile.*\.mac\.dcf$",
            "total": "packetSentToPeer:count",
            "data": [
                "packetSentToPeerBroadcast:count",
                "packetSentToPeerMulticast:count",
                "packetSentToPeerUnicast:count",
            ],
        },
        "mac_dcf_tx_retry": {
            "module_filter": ".*hostMobile.*\.mac\.dcf$",
            "total": "packetSentToPeer:count",
            "data": [
                "packetSentToPeerWithoutRetry:count",
                "packetSentToPeerWithRetry:count",
            ],
        },
    }
    stat = stats[use_stat]
    data = (
        _df.opp.filter()
        .run(run)
        .module_regex(stat["module_filter"], allow_number_range=True)
        .name_in(stat["data"])
        .apply(columns=("name", "value", "module"), copy=True)
    )
    data["module"] = data.module.apply(
        lambda x: "m" + Opp.module_path(x, 1, tuple_on_vector=True)[1]
    )
    data = data.pivot(index="name", columns="module", values="value")

    data_global = (
        _df.opp.filter()
        .run(run)
        .module_regex(stat["module_filter"], allow_number_range=True)
        .name(stat["total"])
        .apply(columns=("name", "value", "module"), copy=True)
    )
    data_global = data_global.pivot(index="name", columns="module", values="value")

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_axes([0.19, 0.2, 0.80, 0.79,])
    ax = stacked_bar(ax, data, data_global, horizontal=True, with_table=True)
    ax.set_facecolor((0.5, 0.5, 0.5))
    fig.set_facecolor((0.4, 0.4, 0.4))
    fig.show()


def build_df_mac_pkt_drop_ts(builder: OppDataProvider, hdf_key="", get_raw_df=False):
    """
    build time series data frame with received packets (bytes/packet) at the mac layer (from lower)
    and the drops occurring at the mac layer. A drop may occure because of packetDropDuplicateDetected,
    packetDropIncorrectlyReceived, packetDropNotAddressedToUs or packetDropOther

    Result format:
    rowIndex: (run, time_step, time, module_id)
    columnIndex: None
    columns (Data): received_bytes, droped_bytes
    N/A values: filled with 0.0
    """
    timer = Timer.create_and_start(
        "load df from opp results", label=build_df_mac_pkt_drop_ts.__name__
    )
    _df = builder.df_from_csv()

    _df["mod"] = _df["module"].apply(lambda r: Opp.module_path(r, 1))

    timer.stop_start("normalize data frame")
    d_ret = build_time_series(
        opp_df=_df,
        opp_vector_names=[
            "packetReceivedFromLower:vector(packetBytes)",
            "packetDrop:vector(packetBytes)",
        ],
        opp_vector_col_names=["received_bytes", "droped_bytes"],
        opp_index=("run", "mod", "module", "name"),
        index=["run", "time_step", "time", "mod"],
        time_bin_size=0.4,
        fill_na=0.0,
    )
    if hdf_key != "":
        timer.stop_start(f"save dataframe to HDF5: {builder.hdf_path}:{hdf_key}")
        with builder.hdf_store.ctx(mode="a") as store:
            d_ret.to_hdf(store, key=hdf_key)
            mapping = builder.get_converter().mapping_data_frame()
            mapping.to_hdf(store, key=f"{hdf_key}/mapping")

    timer.stop()
    if get_raw_df:
        return d_ret, _df
    else:
        return d_ret


def create_mac_pkt_drop_figures(
    builder: OppDataProvider,
    figure_title,
    log_file="",
    use_hdf=True,
    hdf_key="df_mac_pkt_drop_ts",
    show_fig=False,
    figure_prefix="",
):
    """
    Times series plot of packet drop rate over simulation time.
    Expected input:
      rowIndex: (run, time_step, time, module_id)
      columnIndex: None
      columns (Data): received_bytes, droped_bytes
      N/A values: filled with 0.0
    """
    timer = Timer.create_and_start("build graphic")

    if use_hdf and builder.hdf_store.has_key(hdf_key):
        df = builder.hdf_store.get_data(key=hdf_key)
    else:
        df = build_df_mac_pkt_drop_ts(builder=builder, hdf_key=hdf_key)

    df_sum = df.sum(level="time_step")
    df_sum["drop_ratio"] = df_sum["droped_bytes"] / df_sum["received_bytes"]
    df_sum["drop_ratio_sma_3"] = df_sum["drop_ratio"].rolling(window=3).mean()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    l1 = ax.plot(
        df_sum.index.get_level_values("time_step"),
        df_sum["drop_ratio_sma_3"],
        color="g",
        label="drop_ratio",
    )
    ax.set_title(figure_title)
    ax.set_ylabel("pkg-drop-ratio (sma=3) ")
    ax.set_xlabel("time_step [0.4s] ")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower right")

    builder.save_to_output(ax.figure, f"{figure_prefix}pkg_drop_sma_3.png")
    if show_fig:
        ax.figure.show()

    if log_file != "":
        df_msg = parse_cmdEnv_outout(log_file)
        ax_msg = ax.twinx()
        l2 = ax_msg.plot("time", "msg_present", data=df_msg, label="num of messages")
        l3 = ax_msg.plot("time", "msg_in_fes", data=df_msg, label="num in fes")
        ax_msg.set_ylabel("num messages")

        lns = l1 + l2 + l3
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc="upper left")

        # fig 2
        builder.save_to_output(
            ax.figure, f"{figure_prefix}pkg_drop_sma_3_num_messages.png"
        )
        if show_fig:
            ax.figure.show()

        # fig 3
        ax_msg.set_yscale("log")
        builder.save_to_output(
            ax.figure, f"{figure_prefix}pkg_drop_sma_3_log_num_messages.png"
        )
        if show_fig:
            ax.figure.show()

    timer.stop()
