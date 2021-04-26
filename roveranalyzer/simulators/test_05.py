import time

import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pandas as pd

from roveranalyzer.simulators.crownet.dcd.dcd_factory import (
    DcdBuilder,
    PickleState,
    Timer,
)
from roveranalyzer.simulators.crownet.dcd.interactive import (
    InteractiveAreaPlot,
    InteractiveValueOverDistance,
)
from roveranalyzer.simulators.crownet.dcd.util import (
    create_error_df,
    remove_not_selected_cells,
)
from roveranalyzer.simulators.opp.scave import ScaveData, ScaveTool
from roveranalyzer.simulators.vadere.plots.plots import pcolormesh_dict

matplotlib.use("TkAgg")

from roveranalyzer.utils import PathHelper, check_ax

ROOT = "/home/vm-sts/results/"


def get_env(run):
    root = f"{ROOT}/{run}"
    p = PathHelper(root)
    return p


def read_data(path, *args, **kwargs):
    global_map_path = path.glob("global.csv", recursive=False, expect=1)
    node_map_paths = path.glob("dcdMap_*.csv")
    scenario_path = path.glob("vadere.d/*.scenario", expect=1)
    p_name = "dcdMap_full.p"
    # p_name = "dcdMap_merged.p"

    _b = (
        DcdBuilder()
        .use_real_coords(True)
        .all()
        .clear_single_filter()
        .plotter(scenario_path)
        .csv_paths(global_map_path, node_map_paths)
        .pickle_base_path(path.get_base())
        .pickle_name(p_name)
        .pickle_as(PickleState.FULL)
        # .pickle_as(PickleState.MERGED)
    )

    # strip not selected values to speed up
    _b.add_single_filter([remove_not_selected_cells])

    return _b.build()


def analyse_interactive(dcd, what):
    if what == "map":
        # edgecolors="black"
        time = 2
        id = 0
        fig, ax = dcd.area_plot(
            time_step=time,
            node_id=id,
            value="count",
            pcolormesh_dict=pcolormesh_dict(vmin=0, vmax=4),
            title="",
        )
        i = InteractiveAreaPlot(dcd, ax, value="count")
    elif what == "err":
        fig, ax = dcd.plot_error_over_distance(time_step=2, node_id=1, value="sqerr")
        i = InteractiveValueOverDistance(
            dcd, ax, value="sqerr", update_f=dcd.update_error_over_distance
        )
    else:

        fig, ax = dcd.plot_delay_over_distance(
            time_step=14, node_id=1, value="measurement_age"
        )
        i = InteractiveValueOverDistance(
            dcd, ax, value="measurement_age", update_f=dcd.update_delay_over_distance
        )

    i.show()


def main():
    path = PathHelper("/home/mweidner/data/vadere00_60_20210214-21:51:11")
    dcd = read_data(path)
    analyse_interactive(
        dcd, "err"
    )  # selects owner_dist and squerr at timestamp 2 for node #1
    # analyse_interactive(dcd, "map")
    # analyse_interactive(dcd, "age")

    path = "/home/mweidner/data/vadere00_60_20210214-21:51:11/vars_rep_1.vec"
    scave = ScaveTool()
    scave_f = (
        scave.filter_builder()
        .gOpen()
        .module("*.node[*].aid.densityMapApp")
        .OR()
        .module("*.node[*].aid.beaconApp")
        .gClose()
        .AND()
        .gOpen()
        .name("packetSent:vector?packetBytes?")
        .OR()
        .name("rcvdPkLifetime:vector")
        .gClose()
    )
    c_map = dcd.count_map
    print(scave_f.str())
    _df = scave.load_df_from_scave(input_paths=path, scave_filter=scave_f)
    timer = Timer.create_and_start("new_way", label="XXX")
    b = (
        _df.opp.filter()
        .vector()
        .name_in(["packetSent:vector(packetBytes)", "rcvdPkLifetime:vector"])
        .normalize_vectors(axis=0)
    )
    # dcd.count_map.to_hdf('count_map.h5', key='df1', format='t')

    # df = pd.read_hdf("testdata.h5", "data")
    # store = pd.HDFStore('testdata.h5')
    # for lvl in range(0, dcd.count_map.columns.levshape[0]):
    #     store.append('data', dcd.count_map[0])
    print(b.shape)
    # timer.stop()
    print("done")


if __name__ == "__main__":
    main()
