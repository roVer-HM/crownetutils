import os

import numpy as np
from oppanalyzer.utils import RoverBuilder

import roveranalyzer.oppanalyzer.wlan80211 as w80211
from uitls.path import PathHelper
from vadereanalyzer.plots.plots import DensityPlots, NumPedTimeSeries

if __name__ == "__main__":
    builder_20 = RoverBuilder(
        path=PathHelper.from_env(
            "ROVER_MAIN",
            "rover/simulations/simple_detoure/results/final_20200402-15:32:25/",
        ),
        analysis_name="mac",
        analysis_dir="analysis.d",
        hdf_store_name="analysis.h5",
    )
    builder_20.set_scave_filter('module("*.hostMobile[*].*.mac")')
    builder_20.set_scave_input_path("vars_rep_0.vec")

    # drop ration graphs
    # w80211.create_mac_pkt_drop_figures(
    #     builder=builder_20,
    #     # log_file=builder_20.root.join("vars_p1Rate0.2_p2Rate0.8_rep_0.out"),
    #     figure_title="Mac package drop ratio (20% module penetration)",
    #     hdf_key="/df/mac_pkt_drop_ts",
    #     show_fig=True,
    # )

    # Number of Pedestrians
    vout = builder_20.vadere_output_from("vars_rep_0")
    df = vout.files["startEndtime.csv"].df(
        set_index=False, column_names=["pedId", "endTime", "startTime"]
    )

    p01 = NumPedTimeSeries.create().build(
        df,
        c_count="pedId",
        c_start="startTime",
        c_end="endTime",
        title="Number of Pedestrian in Simulation",
    )
    p01.fig.show()

    # mesh
    # frame = 200  # time = frame* 0.4
    # mesh_path = os.path.join(vout.output_dir, "Mesh_trias1.txt")
    # df_density = vout.files["Density_trias1.csv"].df(
    #     set_index=False,
    #     column_names=["timeStep", "faceId", "all_peds", "informed_peds"],
    # )
    # density_plots_all = DensityPlots(
    #     mesh_path, df_density.loc[:, ("timeStep", "faceId", "all_peds")].copy()
    # )
    # density_plots_informend = DensityPlots(
    #     mesh_path, df_density.loc[:, ("timeStep", "faceId", "informed_peds")].copy()
    # )
    #
    # density_plots_all.plot_density(frame, "density", max_density=0.8)
    # density_plots_informend.plot_density(frame, "density", max_density=0.8)
    #
    # frames = np.arange(1, 620, 1)  # frames, time = 0.4* frame, min = 1!
    # density_plots_all.animate_density(
    #     frames, "density", "density_movie_all", max_density=0.8
    # )
    # density_plots_informend.animate_density(
    #     frames, "density", "density_movie_informend", max_density=0.8
    # )
