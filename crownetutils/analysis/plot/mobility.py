import matplotlib.pyplot as plt
import numpy as np
from pandas import IndexSlice as _i

from crownetutils.analysis.common import RunMap
from crownetutils.analysis.omnetpp import CellOccupancyInfo
from crownetutils.utils.dataframe import format_frame, siunitx
from crownetutils.utils.plot import PlotUtil_, plt_rc_same


class PlotCellOccupancy_(PlotUtil_):
    def plot_cell_occupation_info(
        self, info: CellOccupancyInfo, run_map: RunMap, fig_path
    ):
        with run_map.pdf_page(fig_path) as pdf:
            m_seeds = run_map.get_mobility_seed_set()
            for seed in m_seeds:
                with plt.rc_context(plt_rc_same(size="xx-large")):
                    sub_plt = "12;63;44;55"
                    fig, axes = plt.subplot_mosaic(sub_plt, figsize=(16, 3 * 9))
                    ahist = axes["1"]
                    ahist2: plt.Axes = axes["6"]
                    astat = axes["2"]
                    astat2 = axes["3"]
                    agrid = axes["4"]
                    abox = axes["5"]
                    # fig, (ahist, astat, abox) = plt.subplots(3, 1, figsize=(16, 3*9))
                    # info.occup_sim_by_cell
                    ahist.hist(info.occup_sim_by_cell)
                    ahist.set_xlabel("cell occupancy (time) percentage")
                    ahist.set_ylabel("count")
                    ahist.set_title(
                        "Percentage of time a cell is occupied by at least one agent"
                    )

                    zz = (
                        info.occup_sim_by_cell_grid.loc[_i[:, :, seed]]
                        .groupby(["x", "y"])
                        .mean()
                    )
                    z = (
                        # info.occup_sim_by_cell_grid.loc[_i[:, :, 0 ]]
                        # .reset_index("data", drop=True)
                        info.occup_sim_by_cell_grid.loc[_i[:, :, seed]]
                        .groupby(["x", "y"])
                        .mean()
                        .unstack("y")
                        .to_numpy()
                        .T
                    )
                    y_min = zz.index.get_level_values("y").min()
                    y_max = zz.index.get_level_values("y").max()
                    x_min = zz.index.get_level_values("x").min()
                    x_max = zz.index.get_level_values("x").max()
                    extent = (x_min, x_max, y_min, y_max)
                    im = agrid.imshow(z, origin="lower", extent=extent, cmap="Reds")
                    agrid.set_title("Cell occupancy in percentage")
                    agrid.set_ylabel("y in meter")
                    agrid.set_xlabel("x in meter")
                    cb = self.add_colorbar(im, aspect=10, pad_fraction=0.5)

                    box_df = (
                        info.occup_interval_by_cell.loc[_i[:, :, :, seed]]
                        .groupby(["x", "y", "bins"])
                        .mean()
                    )
                    _ = (
                        box_df.reset_index()
                        .loc[:, ["bins", "occupation_time_delta"]]
                        .boxplot(
                            column=["occupation_time_delta"],
                            by=["bins"],
                            rot=90,
                            meanline=True,
                            showmeans=True,
                            widths=0.25,
                            ax=abox,
                        )
                    )
                    abox.set_xlabel("Simulation time intervals in [s]")
                    abox.set_ylabel("Cell (time) occupation in percentage")
                    abox.set_title(
                        "Interval grouped: Percentage of time a cell is occupied by at least one agent"
                    )
                    _d = box_df.groupby(["bins"]).mean()
                    # _d = info.occup_interval_describe.loc[_i[:, :, "mean"]]
                    abox.plot(
                        np.arange(1, _d.shape[0] + 1, 1),
                        _d,
                        linewidth=2,
                        label="mean occupation",
                    )

                    astat.axis("off")
                    s = (
                        info.occup_sim_by_cell.loc[_i[:, :, seed]]
                        .groupby(["x", "y"])
                        .mean()
                        .describe()
                        .reset_index()
                    )
                    s.columns = ["stat", "value"]
                    s = format_frame(
                        s, col_list=["value"], si_func=siunitx(precision=4)
                    )
                    # s = s.T
                    tbl = astat.table(
                        cellText=s.values, colLabels=s.columns, loc="center"
                    )
                    tbl.set_fontsize(14)
                    tbl.scale(1, 2)
                    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

                    #
                    ahist2.hist(
                        info.occup_interval_length.loc[
                            _i[:, :, False, seed], ["delta"]
                        ],
                        bins=100,
                        label="Empty",
                    )
                    ahist2.set_xlabel("Interval length in second")
                    ahist2.set_ylabel("count")
                    ahist2.set_title("Interval length distribution for empty periods")

                    astat2.axis("off")
                    s = (
                        info.occup_interval_length.loc[_i[:, :, :, seed]]
                        .reset_index()
                        .groupby(["cell_occupied"])["delta"]
                        .describe()
                        .T.reset_index()
                        .rename(
                            columns={"index": "stat", True: "Occupied", False: "Empty"}
                        )
                    )
                    # s = info.occup_sim_describe.reset_index().iloc[:, [0, -1]]
                    # s.columns = ["stat", "value"]
                    # s = s.T
                    s = format_frame(
                        s, col_list=s.columns[1:], si_func=siunitx(precision=4)
                    )
                    tbl = astat2.table(
                        cellText=s.values, colLabels=s.columns, loc="center"
                    )
                    tbl.set_fontsize(14)
                    tbl.scale(1, 2)
                    # fix super title
                    fig.suptitle(f"Cell occupation info for mobility seed {seed}")
                    print(f"create figure: {fig._suptitle.get_text()}")
                    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
                    pdf.savefig(fig)
                    plt.close(fig)


PlotCellOccupancy = PlotCellOccupancy_()
