import os
from multiprocessing import Pool, TimeoutError
from os.path import join

import numpy as np
from oppanalyzer.utils import RoverBuilder

import roveranalyzer.oppanalyzer.wlan80211 as w80211
from uitls import Timer
from uitls.mesh import SimpleMesh
from uitls.path import PathHelper
from vadereanalyzer.plots.plots import (DensityPlots, NumPedTimeSeries,
                                        PlotOptions, mono_cmap, t_cmap)

if __name__ == "__main__":
    builder = RoverBuilder(
        path=PathHelper.from_env(
            "ROVER_MAIN", "rover/simulations/simple_detoure/results/",
        ),
        analysis_name="mac",
        analysis_dir="analysis.d",
        hdf_store_name="analysis.h5",
    )
    builder.set_scave_filter('module("*.hostMobile[*].*.mac")')
    builder.set_scave_input_path("vars_rep_0.vec")

    # p = builder.root.glob("final_2020-04-03*/**/*.scenario")
    # print("\n".join(p))

    # drop ration graphs
    # w80211.create_mac_pkt_drop_figures(
    #     builder=builder_20,
    #     # log_file=builder_20.root.join("vars_p1Rate0.2_p2Rate0.8_rep_0.out"),
    #     figure_title="Mac package drop ratio (20% module penetration)",
    #     hdf_key="/df/mac_pkt_drop_ts",
    #     show_fig=True,
    # )

    # Number of Pedestrians
    # vout = builder_20.vadere_output_from("vars_rep_0")
    # df = vout.files["startEndtime.csv"].df(
    #     set_index=False, column_names=["pedId", "endTime", "startTime"]
    # )
    #
    # p01 = NumPedTimeSeries.create().build(
    #     df,
    #     c_count="pedId",
    #     c_start="startTime",
    #     c_end="endTime",
    #     title="Number of Pedestrian in Simulation",
    # )
    # p01.fig.show()

    outpaths = builder.root.glob("final_2020-04-03*/**/*.scenario")
    outpaths = [outpaths[0]]
    for p in outpaths:
        # print("\n".join(p))
        vout = builder.vadere_output_from(os.path.split(p)[0], is_abs=True)
        print(vout.output_dir)

        # mesh
        frame = 1200  # time = frame* 0.4
        mesh_str = vout.files["Mesh_trias1.txt"].as_string(remove_meta=True)
        mesh = SimpleMesh.from_string(mesh_str)
        df_density = vout.files["Density_trias1.csv"].df(
            set_index=True, column_names=["all_peds", "informed_peds"],
        )
        cmap_dict = {
            "informed_peds": t_cmap(cmap_name="Reds", replace_index=(0, 1, 0.0)),
            "all_peds": t_cmap(
                cmap_name="Blues",
                zero_color=(1.0, 1.0, 1.0, 1.0),
                replace_index=(0, 1, 1.0),
            ),
        }
        density_plots_all = DensityPlots(
            mesh, df_density, cmap_dict, slow_motion=[(250.0, 257.0, 24)]
        )
        # density_plots_informend = DensityPlots(mesh, df_density.loc[:, "informed_peds"])

        t = Timer.create_and_start("build video", label="detour_analysis_test")
        density_plots_all.animate_density(
            PlotOptions.DENSITY,
            join(vout.output_dir, "mapped_density_0.8"),
            animate_time=(1.0, 498.0),
            max_density=1.2,
            norm=0.8,
            plot_data=("all_peds", "informed_peds"),
            color_bar_from=(0, 1),
            cbar_lbl=("DensityAll [#/m^2]", "DensityInformed [#/m^2]"),
            title="Mapped density (80% communication)",
            # multi_pool=pool,
        )
        t.stop()
