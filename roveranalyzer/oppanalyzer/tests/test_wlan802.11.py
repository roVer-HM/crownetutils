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


if __name__ == "__main__":
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
