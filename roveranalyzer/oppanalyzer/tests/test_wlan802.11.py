import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import roveranalyzer.oppanalyzer.wlan80211 as w80211
from roveranalyzer.oppanalyzer.configuration import Config
from roveranalyzer.oppanalyzer.utils import (
    OppDataProvider,
    ScaveTool,
    cumulative_messages,
    parse_cmdEnv_outout,
)
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
    w80211.mac_drop_ration_bar_chart(df, "vadereDimensional-1-20200313-17:07:02-11")


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
        df_n["qtime"],
        60,
        density=True,
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


def mac_analysis(base_dir, fig_title, vec_input, prefix, out_input=None):
    builder = OppDataProvider(
        path=PathHelper(base_dir),
        analysis_name=f"{prefix}mac",
        analysis_dir="analysis.d",
        hdf_store_name="analysis.h5",
    )
    builder.set_scave_filter('module("*.hostMobile[*].*.mac")')
    builder.set_scave_input_path(vec_input)
    if out_input is not None:
        _log_file = builder.root.join(out_input)
    else:
        _log_file = ""
    w80211.create_mac_pkt_drop_figures(
        builder=builder,
        log_file=_log_file,
        figure_title=fig_title,
        figure_prefix=prefix,
        hdf_key=f"/df/{prefix}mac_pkt_drop_ts",
        show_fig=True,
    )


if __name__ == "__main__":
    pass
