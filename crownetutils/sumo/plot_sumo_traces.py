import glob
from itertools import repeat
import os
from functools import partial
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.ticker import MultipleLocator

from crownetutils.sumo import SimDir
from crownetutils.sumo.bonnmotion_reader import frame_from_bm
from crownetutils.utils.parallel import ExecutionItem, run_args_map, run_items
from crownetutils.utils.plot import enb_with_hex


def save_descriptive_stats(
    traces: pd.DataFrame,
    output_path: str,
    sim_id: int,
    ax_apply: dict,
    enb_pos,
    inner_r,
):
    ped_count_ts = (
        traces.groupby("time")[["id"]].count().set_axis(["count"], axis=1).reset_index()
    )
    route_dist_df = (
        traces.groupby("id")[["dist"]]
        .sum()
        .reset_index()
        .set_axis(["node_id", "route_length"], axis=1)
    )
    min_ = (
        traces.loc[traces.groupby("id")["time"].idxmin(), ["id", "x", "y"]]
        .set_index("id", drop=True)
        .add_prefix("min_")
    )
    max_ = (
        traces.loc[traces.groupby("id")["time"].idxmax(), ["id", "x", "y"]]
        .set_index("id", drop=True)
        .add_prefix("max_id")
    )
    #
    v = pd.concat([min_, max_], axis=1).values.reshape((-1, 2, 2))
    route_dist_df["start_end_dist"] = np.linalg.norm(v[:, 0] - v[:, 1], axis=1)

    fig, axes = plt.subplot_mosaic("AADD;BCDD", figsize=(32, 18))
    a1 = axes["A"]
    a1.plot("time", "count", data=ped_count_ts, color="black", label="number ped")
    a1.set_xlabel("time in seconds")
    a1.set_ylabel("number of pedestrians")
    a1.set_title("Pedestrian count over time in Simulation area")
    ax_apply["A"](a1)

    a2 = axes["B"]
    a2.hist(route_dist_df["route_length"])
    a2.set_title("Trajectory length of pedestrians")
    a2.set_xlabel("distance in meter")
    a2.set_ylabel("count")
    ax_apply["B"](a2)

    a3 = axes["C"]
    a3.hist(route_dist_df["start_end_dist"])
    a3.set_title("Tajectory start/end euclidean distance")
    a3.set_xlabel("distance meter")
    a3.set_ylabel("count")
    ax_apply["C"](a3)

    a4 = axes["D"]
    plot_trace(a4, traces, enb_pos, inner_r)
    ax_apply["D"](a4)

    fig.tight_layout()
    fig_path = os.path.join(output_path, f"trace_ts_{sim_id:03.0f}.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path)


def nearest_enb(trace, enb):

    t =  np.tile(trace, enb.shape[0]).reshape((-1,3))

    min_idx = np.linalg.norm(
        t[:, 1:] - enb,
        axis=1
    ).argmin()

    return np.concatenate([ trace, [min_idx]])


def plot_trace(ax: plt.Axes, traces, enb, inner_r):
    # create line segments for traces
    traces = traces.sort_values(["id", "time"])
    segments = pd.concat([traces, traces.shift(-1)], axis=1, ignore_index=True)
    segments = segments.set_axis(
        [*traces.add_prefix("p1_").columns, *traces.add_prefix("p2_").columns], axis=1
    )
    segments = segments[segments["p1_id"] == segments["p2_id"]].copy()

    # indexed_segment_start = segments.reset_index()[["index", "p1_x", "p2_x"]].values
    # args_iter = zip(indexed_segment_start, repeat(enb[["base_x", "base_y"]].values))
        
    # out = run_args_map(nearest_enb, list(args_iter), pool_size=8)

    lc = segments[["p1_x", "p1_y", "p2_x", "p2_y"]].values.reshape((-1, 2, 2))
    ax.add_collection(LineCollection(lc))
    patches = [enb_with_hex(p, inner_r=inner_r, scale=200) for p in enb[["base_x", "base_y"]].values]
    ax.add_collection(
        PatchCollection(patches=[p[0] for p in patches], facecolors="black")
    )
    ax.add_collection(
        PatchCollection(
            patches=[p[1] for p in patches], facecolors="none", edgecolors="black"
        )
    )
    ax.set_aspect("equal")


def _apply_ax(
    ax: plt.Axes,
    xlim=None,
    ylim=None,
    x_major_mult=None,
    x_minor_mult=None,
    y_major_mult=None,
    y_minor_mult=None,
):
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if x_major_mult is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_major_mult))
    if x_minor_mult is not None:
        ax.xaxis.set_minor_locator(MultipleLocator(x_minor_mult))

    if y_major_mult is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major_mult))
    if y_minor_mult is not None:
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor_mult))


def process_sumo_sim(sim_dir: SimDir, enb_position: str):
    enb = pd.read_csv(enb_position, comment="#")

    ax_apply = {
        "A": partial(
            _apply_ax, xlim=(0, 3600), ylim=(0, 850), y_major_mult=100, x_major_mult=500
        ),
        "B": partial(_apply_ax, ylim=(0, 350), y_major_mult=50),
        "C": partial(_apply_ax, ylim=(0, 350), y_major_mult=50),
        "D": partial(
            _apply_ax,
            xlim=(-500, 6000),
            ylim=(-500, 4000),
            y_major_mult=1000,
            x_major_mult=1000,
        ),
    }
    bm_files = glob.glob(sim_dir.bm("*__muc.bonnmotion.gz"))
    items = []
    for idx, bm in enumerate(bm_files):
        i = ExecutionItem(fn=frame_from_bm, args=(bm,))
        i.add_post_function(
            save_descriptive_stats,
            output_path=sim_dir.out("traj_check"),
            sim_id=idx,
            ax_apply=ax_apply,
            enb_pos=enb[["base_x", "base_y"]].values,
            inner_r=650,
        )
        items.append(i)

    run_items(items, pool_size=20)
