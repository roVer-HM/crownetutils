import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from roveranalyzer.oppanalyzer.utils import (DcdMetaData,
                                             build_global_density_map,
                                             build_local_density_map)
from roveranalyzer.uitls.path import PathHelper
from roveranalyzer.vadereanalyzer.plots.plots import t_cmap
from roveranalyzer.vadereanalyzer.plots.scenario import VaderScenarioPlotHelper

# matplotlib.use("TkAgg")


def on_mouse_move(event):
    print("Event received:", event.x, event.y)


def not_null(df):
    return df[df["count"] != 0.0]


def time_delta(df, time, delta=5.0):

    return df[(df["received_t"] > time - delta) & (df["received_t"] <= time)]


def df_time_delta(df, time, delta=5.0, copy=True):
    _df = df.copy()
    # mask: all values where received_t is older than t = time-delta
    # clear values for these rows.
    mask = (_df["received_t"] < (time - delta)) & (_df["received_t"] > time)
    _df[mask] = 0
    return _df


if __name__ == "__main__":
    s = PathHelper.rover_sim(
        "mucFreiNetdLTE2dMulticast/",
        "vadere00_geo_20201012_2/vadere.d/mf_2peds.scenario",
    ).abs_path()

    scenario = VaderScenarioPlotHelper(s)
    # read DataFrame. Use real coordinates and not the cell
    # count to allow background mesh
    node_paths = [
        "0a:aa:00:00:00:02",
        "0a:aa:00:00:00:03",
        "0a:aa:00:00:00:04",
        "0a:aa:00:00:00:05",
        "0a:aa:00:00:00:06",
        "0a:aa:00:00:00:07",
        "0a:aa:00:00:00:08",
    ]
    df_locs = []
    for node in node_paths:
        path_loc = PathHelper.rover_sim(
            "mucFreiNetdLTE2dMulticast/", f"vadere00_geo_20201012_2/{node}.csv",
        ).abs_path()
        df_locs.append(
            build_local_density_map(path_loc, real_coords=True, full_map=False)
        )

    path_glb = PathHelper.rover_sim(
        "mucFreiNetdLTE2dMulticast/", "vadere00_geo_20201012_2/global.csv",
    ).abs_path()
    meta_glb, df_glb = build_global_density_map(
        path_glb, real_coords=True, with_id_list=True, full_map=False
    )
    dcd: DcdMap2D = DcdMap2D.from_separated_frames((meta_glb, df_glb), df_locs)
    dcd.set_scenario_plotter(scenario)
    # dcd.describe_raw()
    # dcd.describe_raw(global_only=True)
    # d = dcd.count_diff("abs_diff")
    # dcd.plot_count()
    dcd.plot_count_diff().figure.show()
    # a = dcd.plot2(12.0, 3, title="(all data)")
    # a.show()

    d = dcd.with_age("measurement_age", 2.0)
    aa = d.plot_count_diff()
    aa.figure.show()
    print("hi")
    # _, df_loc = df_locs[0]
    # diff = df_glb - df_loc
    # diff = diff.apply(lambda x: np.abs(x))

    # X = df_glb.index.levels[1].to_numpy()
    # Y = df_glb.index.levels[2].to_numpy()

    # time = 2.0 * 12
    # d = df_loc.loc[(time)]  # .unstack().transpose()
    # d_clear = df_time_delta(d, time, delta=3.0)
    # d_diff = diff.loc[(time), ("count")].unstack().transpose()

    # fig, ax = plt.subplots(1, 2, figsize=(16, 16))
    # for aa in ax:
    #     aa.set_aspect("equal")
    #     aa.set_xlim([0, 100 * 3.0])
    #     aa.set_ylim([0, 80 * 3.0])
    #     scenario.add_obstacles(aa)

    # # # use transparent color map (transparent 0)
    # cmap = t_cmap(cmap_name="Reds", replace_index=(0, 1, 0.0))
    # m1 = ax[0].pcolormesh(X, Y, d, cmap=cmap)
    # fig.colorbar(m1, ax=ax[0], shrink=0.5)
    # ax[0].set_title(f"t={time} Map")
    # m2 = ax[1].pcolormesh(X, Y, d_diff, cmap=cmap)
    # fig.colorbar(m1, ax=ax[1], shrink=0.5)
    # ax[1].set_title(f"t={time} (Diff)")
    # # ax[1, 1].pcolorreceived_tmesh(X, Y, dfs3, cmap=cmap)
    # # ax[1, 1].set_title("t=50.0 (ID=9:1:268)")

    # plt.show()
    # l = df_loc[df_loc["count"] > 0]
    # l = l.loc[(24.0)]
    # received_tl
    # df_glb.loc[(24.0, 174.0, 222.0)]
    # df_glb.loc[(24.0), ("count")].sum()
    # df_loc.loc[(24.0), ("count")].sum()
    # d24 = df_loc.loc[(24.0)]
    # d24
    # d24[d24["count"] > 0]
    # d24[(d24["count"] > 0)]
    # d24[(d24["count"] > 0)].sum()
    # d24[(d24["count"] > 0) & (d24["received_t"] > 16.0)].sum()
    # d24[(d24["count"] > 0) & (d24["received_t"] > 20.0)].sum()
    # d24[(d24["count"] > 0) & (d24["received_t"] > 23.0)].sum()
    # d24[(d24["count"] > 0) & (d24["received_t"] > 23.0)]
    # df_glb.loc[(24.0), ("count")]
    # dg24 = df_glb.locppa:linrunner/thinkpad-extras[(24.0)]
    # dg24 = dg24[dg24["count"] > 0]
    # dg24
    # d24[(d24["count"] > 0) & (d24["received_t"] > 20.0)]
