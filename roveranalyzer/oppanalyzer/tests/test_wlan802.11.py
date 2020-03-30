import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import roveranalyzer.oppanalyzer.detour as detour
import roveranalyzer.oppanalyzer.wlan80211 as w80211
from roveranalyzer.oppanalyzer.configuration import Config
from roveranalyzer.oppanalyzer.rover_analysis import Opp, OppPlot
from roveranalyzer.oppanalyzer.utils import (
    RoverBuilder,
    ScaveRunConverter,
    ScaveTool,
    build_time_series,
    cumulative_messages,
    parse_cmdEnv_outout,
    simsec_per_sec,
    stack_vectors,
)
from roveranalyzer.uitls import Timer
from roveranalyzer.uitls.path import PathHelper


def drop_ration():
    cfg = Config()
    scv = ScaveTool(cfg)
    path = PathHelper.from_env("ROVER_MAIN").extend_base(
        "simulation-campaigns", "results"
    )
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
    path = PathHelper.from_env("ROVER_MAIN").extend_base(
        "simulation-campaigns", "results"
    )
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
    path = PathHelper.from_env("ROVER_MAIN").extend_base(
        "simulation-campaigns", "results"
    )
    single_run = "simpleDetour_miat0_85_20200313"
    d = parse_cmdEnv_outout(path.join(single_run, "vars_p1Rate0.2_p2Rate0.8_rep_0.out"))
    fig, ax = cumulative_messages(d)
    fig.show()


def mac_pending_queue_hist():
    cfg = Config()
    scv = ScaveTool(cfg)
    path = PathHelper.from_env("ROVER_MAIN").extend_base(
        "simulation-campaigns", "results"
    )
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


def build_df_mac_pkt_drop_ts(builder: RoverBuilder, hdf_key="", get_raw_df=False):
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
            "packetDropIncorrectlyReceived:vector(packetBytes)",
        ],
        opp_vector_col_names=["received_bytes", "droped_bytes"],
        opp_index=("run", "mod", "module", "name"),
        index=["run", "time_step", "time", "mod"],
        time_bin_size=0.4,
        fill_na=0.0,
    )
    if hdf_key != "":
        timer.stop_start(f"save dataframe to HDF5: {builder.hdf_path}:{hdf_key}")
        with builder.store_ctx(mode="a") as store:
            d_ret.to_hdf(store, key=hdf_key)
            mapping = builder.get_converter().mapping_data_frame()
            mapping.to_hdf(store, key=f"{hdf_key}/mapping")

    timer.stop()
    if get_raw_df:
        return d_ret, _df
    else:
        return d_ret


def create_mac_pkt_drop_figures(
    builder: RoverBuilder,
    log_file,
    figure_title,
    use_hdf=True,
    hdf_key="df_mac_pkt_drop_ts",
    show_fig=False,
    figure_prefix="",
):
    timer = Timer.create_and_start("build graphic")

    if use_hdf and builder.store_exists():
        df = builder.hdf_get(key=hdf_key)
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

    # fig 1
    ax.legend(l1, l1[0].get_label(), loc="upper left")
    builder.save_to_output(ax.figure, f"{figure_prefix}pkg_drop_sma_3.png")
    if show_fig:
        ax.figure.show()

    df_msg = parse_cmdEnv_outout(log_file)
    ax_msg = ax.twinx()
    l2 = ax_msg.plot("time", "msg_present", data=df_msg, label="num of messages")
    l3 = ax_msg.plot("time", "msg_in_fes", data=df_msg, label="num in fes")
    ax_msg.set_ylabel("num messages")

    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="upper left")

    # fig 2
    builder.save_to_output(ax.figure, f"{figure_prefix}pkg_drop_sma_3_num_messages.png")
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


def mac_pkt_drop():
    # builder_80 = RoverBuilder(
    #     path=PathHelper.from_env(
    #         "ROVER_MAIN", "simulation-campaigns/results/simpleDetour_miat0_85_20200313"
    #     ),
    #     analysis_name="mac_8020",
    #     hdf_key="/df/mac_pkt_drop_timeseries",
    # )
    # builder_80.set_scave_filter('module("*.hostMobile[*].*.mac")')
    # builder_80.set_scave_input_path("*0.8*0.2*rep_0.vec")
    # create_mac_pkt_drop_figures(
    #     builder=builder_80,
    #     log_file=builder_80.root.join("vars_p1Rate0.8_p2Rate0.2_rep_0.out"),
    #     figure_title="Mac package drop ratio (80% module penetration)",
    # )

    builder_20 = RoverBuilder(
        path=PathHelper.from_env(
            "ROVER_MAIN", "simulation-campaigns/results/simpleDetour_miat0_85_20200313"
        ),
        analysis_name="mac_2080",
        hdf_key="/df/mac_pkt_drop_timeseries",
    )
    builder_20.set_scave_filter('module("*.hostMobile[*].*.mac")')
    builder_20.set_scave_input_path("*0.2*0.8*rep_0.vec")
    create_mac_pkt_drop_figures(
        builder=builder_20,
        log_file=builder_20.root.join("vars_p1Rate0.2_p2Rate0.8_rep_0.out"),
        figure_title="Mac package drop ratio (20% module penetration)",
    )


if __name__ == "__main__":
    mac_pkt_drop()
