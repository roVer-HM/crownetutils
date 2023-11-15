from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
from numpy.typing import NDArray

import crownetutils.omnetpp.scave as Scave
from crownetutils.analysis.hdf_providers.node_position import (
    CoordinateType,
    NodePositionWithRsdHdf,
)
from crownetutils.analysis.omnetpp import OppAnalysis, Simulation
from crownetutils.utils.logging import logger, timing
from crownetutils.utils.plot import (
    FigureSaver,
    PlotUtil_,
    enb_patch_annotate,
    enb_with_hex,
    with_axis,
)


class _PlotEnb(PlotUtil_):
    @timing
    def cmp_plot_served_blocks_ul_all(
        self,
        sims: List[Simulation],
        enb_index: int = 0,
        saver: FigureSaver = FigureSaver.FIG,
    ):
        _, ax_ts = self.check_ax()
        _, ax_hist = self.check_ax()
        tbl_count = []
        tbl_info = []
        for sim in sims:
            data = OppAnalysis.get_avgServedBlocksUl(sim.sql, enb_index=enb_index)
            bins = int(sim.sql.get_run_config("**.numBands"))
            _, ax_ts = self.plot_ts_enb_served_rb(
                data, time_bucket_length=1.0, ax=ax_ts, label=sim.label
            )
            _, ax_hist = self.plot_hist_enb_served_rb(
                data,
                bins,
                ax=ax_hist,
                density=True,
                label=sim.label,
                alpha=0.5,
                histtype="stepfilled",
            )
            tbl_info.append(
                data.loc[:, ["value"]].describe().set_axis([sim.label], axis=1)
            )
            tbl_count.append(
                data.groupby(["value"])
                .count()
                .reset_index()
                .set_axis(["RB", sim.label], axis=1)
                .set_index(["RB"])
            )

        tbl_count = (
            pd.concat(tbl_count, axis=1).applymap("{:1_.0f}".format).reset_index()
        )
        tbl_count["RB"] = tbl_count["RB"].astype(int)
        fig, ax = self.df_to_table(tbl_count)
        ax.set_title("")
        saver(fig, "Rb_count.pdf")

        tbl_info = pd.concat(tbl_info, axis=1).applymap("{:.4f}".format).reset_index()
        fig, ax = self.df_to_table(tbl_info)
        saver(fig, "Rb_describe.pdf")

        ax_ts.legend()
        saver(ax_ts.get_figure(), "Rb_utilization_ts.pdf")
        ax_hist.legend()
        ax_hist.set_ylabel("density")
        saver(ax_hist.get_figure(), "Rb_utilization_hist.pdf")

    @timing
    def plot_served_blocks_ul_all(
        self,
        data_root: str,
        sql: Scave.CrownetSql,
        saver: FigureSaver = FigureSaver.FIG,
    ):
        num_enb = int(sql.get_run_config("*.numEnb"))
        bins = int(sql.get_run_config("**.numBands"))
        for n in range(num_enb):
            data = OppAnalysis.get_avgServedBlocksUl(sql, enb_index=n)
            fig, ax = self.plot_ts_enb_served_rb(data, time_bucket_length=1.0)
            self.append_title(ax, prefix=f"Enb-{n}:")
            saver(fig, os.path.join(data_root, f"rb_utilization_ts_{n}.pdf"))
            fig, ax = self.plot_hist_enb_served_rb(data, bins, n)
            self.append_title(ax, prefix=f"Enb-{n}:")
            saver(fig, os.path.join(data_root, f"rb_utilization_hist_{n}.pdf"))
            fig, ax = self.plot_ecdf_enb_served_rb(data, bins, n)
            self.append_title(ax, prefix=f"Enb-{n}:")
            saver(fig, os.path.join(data_root, f"rb_utilization_ecdf_{n}.pdf"))
            fig, ax = self.plot_tbl_enb_serverd_rb_count(data)
            self.append_title(ax, prefix=f"Enb-{n}:")
            saver(fig, os.path.join(data_root, f"rb_count_{n}.pdf"))
            fig, ax, _ = self.df_to_table(
                data.describe().applymap("{:1.4f}".format).reset_index(),
                title="Served UL blocks",
            )
            self.append_title(ax, prefix=f"Enb-{n}:")
            saver(fig, os.path.join(data_root, f"rb_stat_{n}.pdf"))

    @with_axis
    def plot_ts_enb_served_rb(
        self,
        data: pd.DataFrame,
        time_bucket_length=1.0,
        *,
        ax: plt.Axes | None = None,
        **kwargs,
    ):
        interval = pd.interval_range(
            start=0.0, end=np.ceil(data.index.max()), freq=time_bucket_length
        )
        data = data.groupby(pd.cut(data.index, interval)).mean()
        data.index = interval.left
        data.index.name = "time"
        data = data.reset_index()
        ax.plot("time", "value", data=data, **kwargs)
        ax.set_title("Average Resource Block (RB) usage over time. (time bin size 1s)")
        ax.set_xlabel("time in [s]")
        ax.set_ylabel("Resource blocks")
        return ax.get_figure(), ax

    @with_axis
    def plot_hist_enb_served_rb(
        self,
        data: pd.DataFrame,
        bins=25,
        enb=0,
        *,
        ax: plt.Axes | None = None,
        **hist_args,
    ):
        data = data["value"]
        d = 1
        left_of_first_bin = 0 - float(d) / 2
        right_of_last_bin = bins + float(d) / 2
        ax.hist(
            data,
            np.arange(left_of_first_bin, right_of_last_bin + d, d),
            align="mid",
            **hist_args,
        )
        ax.set_xlim(-1, bins + 1)
        ax.set_xticks(np.arange(0, bins + 1, 1))
        ax.set_title(f"Resource block utilization of eNB {enb}")
        ax.set_xlabel("Resource Blocks (RB's)")
        ax.set_ylabel("Count")
        return ax.get_figure(), ax

    def plot_ecdf_enb_served_rb(self, data, bins=25, enb=0):
        _x = data["value"].sort_values().values
        _y = np.arange(len(_x)) / float(len(_x))
        fig, ax = self.check_ax()
        ax.plot(_x, _y)
        ax.set_title("ECDF of resource block utilization of eNB {enb}")
        ax.set_xlabel("Resource Blocks (RB's)")
        ax.set_ylabel("ECDF")
        ax.set_xlim(-1, bins + 1)
        ax.set_xticks(np.arange(0, bins + 1, 1))
        return fig, ax

    def plot_tbl_enb_serverd_rb_count(self, data):
        df = (
            data.drop(columns=["vectorId"])
            .reset_index()
            .set_axis(["count", "RB"], axis=1)
            .groupby(["RB"])
            .count()
            .T
        )
        df.columns = [int(c) for c in df.columns]
        df.columns.name = "RB"
        df = df.applymap("{:1_.0f}".format).reset_index()
        fig, ax, _ = self.df_to_table(df)
        ax.set_title("UL scheduling RB's count ")
        return fig, ax

    @with_axis
    def plot_node_enb_association_map(
        self,
        rsd: NodePositionWithRsdHdf,
        coord: CoordinateType = CoordinateType.xy,
        base_cmap: str | colors.Colormap = "tab20",
        inner_r: float = 650,
        ax: plt.Axes = None,
    ):
        trace, segments, segment_colors, rsd_color_map, cmap, norm = rsd.get_ue_traces(
            coord, cmap=base_cmap
        )

        enb = rsd.enb.frame()
        enb["color"] = enb["rsd_id"].apply(lambda x: rsd_color_map[x])

        # append enb hex
        enb_patches, hex_patches = enb_with_hex(
            origin=enb[coord.cols].values, inner_r=inner_r, scale=100
        )
        ax.add_collection(hex_patches)
        lc = LineCollection(segments=segments, colors=segment_colors)
        cbar = ax.get_figure().colorbar(
            ScalarMappable(norm=norm, cmap=cmap), ticks=MultipleLocator(1), ax=ax
        )
        ax.add_collection(lc)

        # for _host, pos in trace.groupby("hostId"):
        #     ax.scatter(pos.iloc[-1][coord.cols[0]], pos.iloc[-1][coord.cols[1]], color="red", marker=".")

        ax.add_collection(enb_patches)
        for _, row in enb.iterrows():
            ax.annotate(
                f"enb {int(row['hostId'])+1}",
                [row[coord.x], row[coord.y]],
                xytext=[row[coord.x] - 150, row[coord.y] - 190],
                textcoords=ax.transData,
            )

        if coord.is_cartesian:
            ax.xaxis.set_major_locator(MultipleLocator(500))
            ax.yaxis.set_major_locator(MultipleLocator(500))
            ax.xaxis.set_minor_locator(MultipleLocator(100))
            ax.yaxis.set_minor_locator(MultipleLocator(100))

            _min = (trace[coord.cols].min() - 500).values
            _max = (trace[coord.cols].max() + 500).values
            ax.set_xlim(_min[0], _max[0])
            ax.set_ylim(_min[1], _max[1])
            ax.set_aspect("equal")

        ax.set_title("Pedestrian traces with cell association (black: no association)")
        ax.set_ylabel("North in meter")
        ax.set_xlabel("East in meter")
        ax.get_figure().tight_layout()
        return ax.get_figure(), ax

    @with_axis
    def plot_mac_cell_throughput_ul(
        self,
        sim: Simulation,
        pos: NodePositionWithRsdHdf,
        ax: plt.Axes = None,
        saver: FigureSaver = None,
    ):
        saver = FigureSaver.FIG(saver)
        fig, ax = self.check_ax(ax)
        enbs = pos.enb.frame()
        colors = pos.enb_colors()
        for e in range(enbs.shape[0]):
            df = sim.sql.vec_data(
                f"World.eNB[{e}].cellularNic.mac",
                "macCellThroughputUl:vector",
            ).reset_index()
            ax.scatter(df["time"], df["value"], marker=".", color=colors[e], alpha=0.5)

        ax.set_ylabel("Bps")
        ax.set_xlabel("time")
        self.auto_major_minor_locator(ax)
        ax.legend()
        fig.tight_layout()
        saver(fig, "mac_cell_throughput_ul.png")


PlotEnb = _PlotEnb()
