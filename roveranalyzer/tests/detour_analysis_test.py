import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vadereanalyzer.scenario_output import LazyDataFrameWrapper

if __name__ == "__main__":
    path = (
        "/home/sts/repos/rover-main/rover/simulations/"
        "mucFreiNetdLTE2dMulticast/"
        "results/vadere00_geo_20200903-10:26:55/grid_9:1:268.csv"
    )
    # load csv with metdata as first line.
    w = LazyDataFrameWrapper(path)
    meta = w.read_meta_data()
    # bound of szenario
    bound = [float(meta["XSIZE"]), float(meta["YSIZE"])]
    # cell size. First cell with [0, 0] is lower left cell
    cell_size = int(meta["CELLSIZE"])
    cell_count = [int(bound[0] / cell_size + 1), int(bound[1] / cell_size + 1)]
    df_raw = w.df(set_index=True, column_names=["count", "measure_t", "received_t"],)

    # create full index: time * numXCells * numYCells
    _idx = [
        df_raw.index.levels[0].to_numpy(),
        np.arange(cell_count[0]),
        np.arange(cell_count[1]),
    ]
    idx = pd.MultiIndex.from_product(_idx, names=("simtime", "x", "y"))
    # create zero filled data frame with index
    df = pd.DataFrame(
        data=np.zeros((len(idx), 3)), columns=["count", "measure_t", "received_t"]
    )
    # set index and update with raw measures. (most will stay at zero)
    df = df.set_index(idx)
    df.update(df_raw)
    dfs = df.loc[(1.0), ("count")].unstack()
    dfs2 = df.loc[(26.0), ("count")].unstack()
    dfs3 = df.loc[(50.0), ("count")].unstack()
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    ax[0, 0].pcolor(dfs, cmap="RdBu")
    ax[0, 0].set_title("t=1.0 (ID=9:1:268)")
    ax[0, 1].pcolor(dfs2, cmap="RdBu")
    ax[0, 1].set_title("t=26.0 (ID=9:1:268)")
    ax[1, 1].pcolor(dfs3, cmap="RdBu")
    ax[1, 1].set_title("t=50.0 (ID=9:1:268)")
    for a in ax:
        for aa in a:
            aa.set_aspect("equal")
            aa.set_xlim([0, 80])
            aa.set_ylim([0, cell_count[1]])
    fig.show()

    # # Make the plot
    # plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
    # plt.show()
    #
    # # Change color palette
    # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greens_r)
    # plt.show()

    # target df: key ["simtime", "x", "y"] value ["count", "measure_t", "received_t"]
    # x: [0, XSIZE / CELLSIZE ]
    # y: [0, YSIZE / CELLSIZE


# builder = RoverBuilder(
#     path=PathHelper.from_env(
#         "ROVER_MAIN", "rover/simulations/simple_detoure/results/",
#     ),
#     analysis_name="stream_video",
#     analysis_dir="analysis.d",
#     hdf_store_name="analysis.h5",
# )
#
# outpaths = builder.root.glob("final_2020-04-03*/**/*.scenario")
# outpaths = [outpaths[1]]
# for p in outpaths:
#     vout = builder.vadere_output_from(os.path.split(p)[0], is_abs=True)
#     print(vout.output_dir)
#     cmap_dict = {
#         "informed_peds": t_cmap(cmap_name="Reds", replace_index=(0, 1, 0.0)),
#         "all_peds": t_cmap(
#             cmap_name="Blues",
#             zero_color=(1.0, 1.0, 1.0, 1.0),
#             replace_index=(0, 1, 1.0),
#         ),
#     }
#     density_plot = DensityPlots.from_mesh_processor(
#         vadere_output=vout,
#         mesh_out_file="Mesh_trias1.txt",
#         density_out_file="Density_trias1.csv",
#         cmap_dict=cmap_dict,
#         data_cols_rename=["all_peds", "informed_peds"],
#     )
#     t = Timer.create_and_start("build video", label="detour_analysis_test")
#     density_plot.set_slow_motion_intervals([(250.0, 257.0, 24)])
#     density_plot.animate_density(
#         PlotOptions.DENSITY,
#         join(vout.output_dir, "mapped_density_0.2.mp4"),
#         animate_time=(245.0, 247.0),
#         max_density=1.2,
#         norm=(1.0, 0.2),
#         plot_data=("all_peds", "informed_peds"),
#         color_bar_from=(0, 1),
#         cbar_lbl=("DensityAll [#/m^2]", "DensityInformed [#/m^2]"),
#         title="Mapped density (20% communication)",
#     )
#     t.stop()
