import os
import sys

import matplotlib.pyplot as plt

import seaborn as sns
from drawnow import drawnow, figure
from roveranalyzer import vadereanalyzer as v
from roveranalyzer.uitls.path import PathHelper
from roveranalyzer.vadereanalyzer.plots.plots import NumPedTimeSeries

sys.path.append(
    os.path.abspath(".")
)  # in case tutorial is called from the root directory
sys.path.append(os.path.abspath(".."))  # in tutorial directly


def xxx(t, df):
    df_30 = df.loc[df["timeStep"] == t, ("x", "y", "gridCount-PID8")]
    df_30 = df_30.pivot("y", "x", "gridCount-PID8")
    ax = sns.heatmap(df_30, cmap="RdBu")
    ax.invert_yaxis()


def density():
    p_helper = PathHelper.from_env("ROVER_MAIN")
    p = p_helper.join(
        "rover/simulations/simple_detoure/vadereanalyzer/output/simple_detour_100x177_2020-03-11_17-06-35.88/"
    )
    output = v.ScenarioOutput.create_output_from_project_output(p)

    df = output.files["gridDensity.csv"]()

    figure(figsize=(7, 7 / 2))
    for t in range(1, 751, 8):
        kwargs = {"t": float(t), "df": df}
        drawnow(xxx, show_once=False, confirm=False, stop_on_close=False, **kwargs)


def fig_num_peds_series():
    p_helper = PathHelper.from_env("ROVER_MAIN")
    trajectories = p_helper.glob(
        "simulation-campaigns", "simpleDetour.sh-results_20200*_mia*/**/postvis.traj"
    )
    # trajectories = p_helper.glob('simulation-campaigns', 'simple_detour_100x177_long*/**/postvis.traj')
    output_dirs = [os.path.split(p)[0] for p in trajectories]
    outputs = [
        v.ScenarioOutput.create_output_from_project_output(p) for p in output_dirs
    ]

    ratio = 16 / 9
    size_x_ax = 10
    size_y_ax = size_x_ax / ratio
    fig, axes = plt.subplots(
        len(outputs), 1, figsize=(size_x_ax, len(outputs) * size_y_ax)
    )

    for idx, o in enumerate(outputs):
        df = o.files["startEndtime.csv"]()
        ax = (
            NumPedTimeSeries.create(ax=axes[idx])
            .build(
                df,
                c_start="startTime-PID7",
                c_end="endTime-PID5",
                c_count="pedestrianId",
                title=o.path("name"),
            )
            .ax
        )
        info_txt = (
            f"inter arrival times: \n"
            f"{o.path('scenario/topography/sources[*]/distributionParameters')}"
        )
        ax.text(0.75, 0.2, info_txt, ha="left", transform=ax.transAxes)

    return fig


if __name__ == "__main__":
    fig_num_peds_series()
