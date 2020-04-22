import os
from os.path import join

from oppanalyzer.utils import RoverBuilder

from uitls import Timer
from uitls.path import PathHelper
from vadereanalyzer.plots.plots import DensityPlots, PlotOptions, t_cmap

if __name__ == "__main__":
    builder = RoverBuilder(
        path=PathHelper.from_env(
            "ROVER_MAIN", "rover/simulations/simple_detoure/results/",
        ),
        analysis_name="stream_video",
        analysis_dir="analysis.d",
        hdf_store_name="analysis.h5",
    )

    outpaths = builder.root.glob("final_2020-04-03*/**/*.scenario")
    outpaths = [outpaths[1]]
    for p in outpaths:
        vout = builder.vadere_output_from(os.path.split(p)[0], is_abs=True)
        print(vout.output_dir)
        cmap_dict = {
            "informed_peds": t_cmap(cmap_name="Reds", replace_index=(0, 1, 0.0)),
            "all_peds": t_cmap(
                cmap_name="Blues",
                zero_color=(1.0, 1.0, 1.0, 1.0),
                replace_index=(0, 1, 1.0),
            ),
        }
        density_plot = DensityPlots.from_mesh_processor(
            vadere_output=vout,
            mesh_out_file="Mesh_trias1.txt",
            density_out_file="Density_trias1.csv",
            cmap_dict=cmap_dict,
            data_cols_rename=["all_peds", "informed_peds"],
        )
        t = Timer.create_and_start("build video", label="detour_analysis_test")
        density_plot.set_slow_motion_intervals([(250.0, 257.0, 24)])
        density_plot.animate_density(
            PlotOptions.DENSITY,
            join(vout.output_dir, "mapped_density_0.2.mp4"),
            animate_time=(245.0, 247.0),
            max_density=1.2,
            norm=(1.0, 0.2),
            plot_data=("all_peds", "informed_peds"),
            color_bar_from=(0, 1),
            cbar_lbl=("DensityAll [#/m^2]", "DensityInformed [#/m^2]"),
            title="Mapped density (20% communication)",
        )
        t.stop()
