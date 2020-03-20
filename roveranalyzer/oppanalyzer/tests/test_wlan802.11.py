import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import roveranalyzer.oppanalyzer.detour as detour
import roveranalyzer.oppanalyzer.wlan80211 as w80211
from roveranalyzer.oppanalyzer.configuration import Config
from roveranalyzer.oppanalyzer.rover_analysis import Opp
from roveranalyzer.oppanalyzer.utils import ScaveTool
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


if __name__ == "__main__":
    cfg = Config()
    scv = ScaveTool(cfg)
    path = RelPath.from_env("ROVER_MAIN").extend_base("simulation-campaigns", "results")
    single_run = "simpleDetour_miat0_85_20200313"
    img_path = path.make_dir(
        single_run, "mac.channelAccess.queue20_80.d", exist_ok=True
    )
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
    idx = 44
    fig, axes = plt.subplots(1, 1, figsize=(16, 9))
    fig.show()
    x = np.array()
    x.reshape()

    ax = df.opp.plot.create_histogram(axes, pending_length.iloc[idx],)
    stats = {
        "xx": [
            {
                "relPath": "dcf.channelAccess.inProgressFrames",
                "stat": ["queueingTime:vector", "queueingLength:vector"],
            },
            {
                "relPath": "dcf.channelAccess.pendingQueue",
                "stat": ["queueingTime:vector", "queueingLength:vector"],
            },
        ]
    }
