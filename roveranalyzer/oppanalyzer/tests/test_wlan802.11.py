import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import roveranalyzer.oppanalyzer.detour as detour
import roveranalyzer.oppanalyzer.wlan80211 as w80211
from roveranalyzer.oppanalyzer.configuration import Config
from roveranalyzer.oppanalyzer.rover_analysis import Opp, OppPlot
from roveranalyzer.oppanalyzer.utils import (
    ScaveRunConverter,
    ScaveTool,
    build_time_series,
    cumulative_messages,
    parse_cmdEnv_outout,
    simsec_per_sec,
    stack_vectors,
)
from roveranalyzer.uitls import Timer
from roveranalyzer.uitls.path import RelPath


def drop_ration():
    cfg = Config()
    scv = ScaveTool(cfg)
    path = RelPath.from_env("ROVER_MAIN").extend_base("simulation-campaigns", "results")
    single_run = "simpleDetour_miat0_85_20200313"
    csv = scv.create_or_get_csv_file(
        csv_path=path.join(single_run, "macDropCount.csv"),
        input_paths=[path.join(single_run, "**/*.sca")],
        scave_filter='module("*.hostMobile[*].mac") OR module("*.hostMobile[*].mobility")',
        override=False,
        recursive=True,
    )
    df = scv.load_csv(csv)
    w80211.mac(df, "vadereDimensional-1-20200313-17:07:02-11")


def detour_stats():
    cfg = Config()
    scv = ScaveTool(cfg)
    path = RelPath.from_env("ROVER_MAIN").extend_base("simulation-campaigns", "results")
    single_run = "simpleDetour_miat0_85_20200313"
    csv = scv.create_or_get_csv_file(
        csv_path=path.join(single_run, "detour20_80_app.csv"),
        input_paths=[
            path.join(single_run, "*0.2*0.8*rep_0.sca"),
            path.join(single_run, "*0.2*0.8*rep_0.vec"),
        ],
        scave_filter='module("*.app[*]") AND '
        '(name("*HopId") OR '
        'name("*HopTime") OR '
        'name("*Age:vector") OR '
        'name("packetSent:count") OR '
        'name("packetReceived:count"))',
        override=False,
        recursive=True,
    )
    df = scv.load_csv(csv)
    df.opp.info()


def sim_messages():
    cfg = Config()
    scv = ScaveTool(cfg)
    path = RelPath.from_env("ROVER_MAIN").extend_base("simulation-campaigns", "results")
    single_run = "simpleDetour_miat0_85_20200313"
    d = parse_cmdEnv_outout(path.join(single_run, "vars_p1Rate0.2_p2Rate0.8_rep_0.out"))
    fig, ax = cumulative_messages(d)
    fig.show()


def mac_pending_queue_hist():
    cfg = Config()
    scv = ScaveTool(cfg)
    path = RelPath.from_env("ROVER_MAIN").extend_base("simulation-campaigns", "results")
    single_run = "simpleDetour_miat0_85_20200313"

    img_path = path.make_dir(
        single_run, "mac.channelAccess.queue20_80.d", exist_ok=True
    )
    # "dcf.channelAccess.inProgressFrames.["queueingTime:vector", "queueingLength:vector"]"
    # "dcf.channelAccess.pendingQueue.["queueingTime:vector", "queueingLength:vector"]"
    csv = scv.create_or_get_csv_file(
        csv_path=path.join(single_run, "mac.channelAccess.queue20_80.csv"),
        input_paths=[
            path.join(single_run, "*0.2*0.8*rep_0.sca"),
            path.join(single_run, "*0.2*0.8*rep_0.vec"),
        ],
        scave_filter='module("*.hostMobile[*].*.mac.dcf.channelAccess.*") AND name("queue*vector")',
        override=False,
        recursive=True,
    )
    df = scv.load_csv(csv)
    df.opp.info()
    pending_length = (
        df.opp.filter()
        .vector()
        .module_regex(".*dcf.channelAccess.pendingQueue")
        .name("queueingTime:vector")
        .apply()
    )

    df_n = pd.DataFrame(columns=["time", "qtime"])
    idx = 0
    while idx < pending_length.shape[0]:
        data = pending_length.iloc[idx]["vectime"].copy()
        data = np.append(data, pending_length.iloc[idx]["vecvalue"].copy())
        data = data.reshape((-1, 2), order="F")
        df_n = df_n.append(
            pd.DataFrame(data, columns=["time", "qtime"]), ignore_index=True
        )
        idx += 1

    fig, axes = plt.subplots(3, 1, figsize=(16, 9))

    axes[0].hist(
        df_n["qtime"], 60, density=True,
    )
    df.opp.plot.create_histogram(
        axes[1],
        pending_length.iloc[int(np.random.uniform(0, pending_length.shape[0]))],
        bins=60,
    )
    df.opp.plot.create_histogram(
        axes[2],
        pending_length.iloc[int(np.random.uniform(0, pending_length.shape[0]))],
        bins=60,
    )
    for a in axes:
        a.set_xlim(0.00, 0.06)

    fig.show()


def build_df_mac_pkt_drop_ts(csv_path, img_dir, input_path, hdf_store_path, hdf_key):
    timer = Timer.create_and_start("create csv", tabstop=1)
    scv = ScaveTool()
    csv = scv.create_or_get_csv_file(
        csv_path=csv_path,
        input_paths=input_path,
        scave_filter='module("*.hostMobile[*].*.mac")',
        override=False,
        recursive=True,
    )
    timer.stop_start("load dataframe from csv")
    converter = ScaveRunConverter(run_short_hand="r")
    _df = scv.load_csv(csv, converters=converter)

    timer.stop_start("add mod column")
    _df["mod"] = _df["module"].apply(lambda r: Opp.module_path(r, 1))

    hdf_store = pd.HDFStore(os.path.join(img_dir, hdf_store_path))
    converter.mapping_data_frame().to_hdf(
        hdf_store, key="df_mac_pkt_drop_ts_mapping", mode="a"
    )
    d_ret = build_time_series(
        opp_df=_df,
        opp_vector_names=[
            "packetReceivedFromLower:vector(packetBytes)",
            "packetDropIncorrectlyReceived:vector(packetBytes)",
        ],
        opp_vector_col_names=["received_bytes", "droped_bytes"],
        opp_index=("run", "mod", "module", "name"),
        hdf_store=hdf_store,
        hdf_key=hdf_key,
        index=["run", "time_step", "time", "mod"],
        time_bin_size=0.4,
        fill_na=0.0,
    )
    timer.stop()
    return d_ret


def res_20():
    timer = Timer.create_and_start("build graphic")
    path = RelPath.from_env("ROVER_MAIN").extend_base(
        "simulation-campaigns", "results", "simpleDetour_miat0_85_20200313"
    )
    csv_p = path.join("mac_20_80.csv")
    img_d = path.make_dir("mac_20_80.d", exist_ok=True)
    input_path = [path.join("*0.2*0.8*rep_0.vec")]

    # drop ratio
    # df = build_df_mac_pkt_drop_ts(
    #     csv_p, img_d, input_path, "mac_pkt_drop.h5", "df_mac_pkt_drop_ts"
    # )
    hdf_store: pd.DataFrame = pd.HDFStore(os.path.join(img_d, "mac_pkt_drop.h5"))
    df = hdf_store.get("df_mac_pkt_drop_ts")
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
    ax.set_title("Mac package drop ratio (20% module penetration)")
    ax.set_ylabel("pkg-drop-ratio (sma=3) ")
    ax.set_xlabel("time_step [0.4s] ")

    # fig 1
    ax.legend(l1, l1[0].get_label(), loc="upper left")
    ax.figure.savefig(os.path.join(img_d, "pkg_drop_20_80_sma_3.png"))
    ax.figure.show()

    df_msg = parse_cmdEnv_outout(path.join("vars_p1Rate0.2_p2Rate0.8_rep_0.out"))
    ax_msg = ax.twinx()
    l2 = ax_msg.plot("time", "msg_present", data=df_msg, label="num of messages")
    l3 = ax_msg.plot("time", "msg_in_fes", data=df_msg, label="num in fes")
    ax_msg.set_ylabel("num messages")

    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="upper left")

    # fig 2
    ax.figure.savefig(os.path.join(img_d, "pkg_drop_20_80_sma_3_num_messages.png"))
    ax.figure.show()

    # fig 3
    ax_msg.set_yscale("log")
    ax.figure.savefig(os.path.join(img_d, "pkg_drop_20_80_sma_3_log_num_messages.png"))
    ax.figure.show()

    timer.stop()


def res_80():
    timer = Timer.create_and_start("build path helper")
    path = RelPath.from_env("ROVER_MAIN").extend_base(
        "simulation-campaigns", "results", "simpleDetour_miat0_85_20200313"
    )
    csv_p = path.join("mac_80_20.csv")
    img_d = path.make_dir("mac_80_20.d", exist_ok=True)
    input_path = [path.join("*0.8*0.2*rep_0.vec")]

    # drop ratio
    timer.stop_start("read scave data")
    df = build_df_mac_pkt_drop_ts(
        csv_p, img_d, input_path, "mac_pkt_drop.h5", "df_mac_pkt_drop_ts"
    )
    # timer.stop_start("read HDFStore")
    # hdf_store = pd.HDFStore(os.path.join(img_d, "mac_pkt_drop.h5"))
    # df = hdf_store.get("df_mac_pkt_drop_ts")
    timer.stop_start("build graphics")
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
    ax.set_title("Mac package drop ratio (80% module penetration)")
    ax.set_ylabel("pkg-drop-ratio (sma=3) ")
    ax.set_xlabel("time_step [0.4s] ")

    # fig 1
    ax.legend(l1, l1[0].get_label(), loc="upper left")
    ax.figure.savefig(os.path.join(img_d, "pkg_drop_80_20_sma_3.png"))
    ax.figure.show()

    df_msg = parse_cmdEnv_outout(path.join("vars_p1Rate0.8_p2Rate0.2_rep_0.out"))
    ax_msg = ax.twinx()
    l2 = ax_msg.plot("time", "msg_present", data=df_msg, label="num of messages")
    l3 = ax_msg.plot("time", "msg_in_fes", data=df_msg, label="num in fes")
    ax_msg.set_ylabel("num messages")

    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="upper left")

    # fig 2
    ax.figure.savefig(os.path.join(img_d, "pkg_drop_80_20_sma_3_num_messages.png"))
    ax.figure.show()

    # fig 3
    ax_msg.set_yscale("log")
    ax.figure.savefig(os.path.join(img_d, "pkg_drop_80_20_sma_3_log_num_messages.png"))
    ax.figure.show()

    timer.stop()


if __name__ == "__main__":
    res_80()
