from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.collections import LineCollection

import crownetutils.omnetpp.scave as Scave
from crownetutils.analysis.omnetpp import OppAnalysis, Simulation
from crownetutils.utils.logging import logger, timing
from crownetutils.utils.plot import FigureSaver, PlotUtil_, with_axis


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
    def plot_node_enb_association_ts(
        self,
        data: pd.DataFrame | Scave.CrownetSql,
        time_resolution: float = 1.0,
        start_time: float = 0.0,
        ax: plt.Axes = None,
        **kwargs,
    ):
        if isinstance(data, Scave.CrownetSql):
            data = OppAnalysis.get_serving_enb_interval(sql=data, **kwargs)

        node_ids = data.index.get_level_values("hostId").unique()
        min_time = start_time
        max_time = data["end"].max()
        time_index = pd.Index(
            np.arange(min_time, max_time, step=time_resolution), name="time"
        )
        index = pd.MultiIndex.from_product((node_ids, time_index))
        enb = pd.DataFrame(-1, columns=["interval"], index=index)
        for host_id, df in enb.groupby("hostId"):
            iindex = pd.IntervalIndex(data.loc[host_id].index)
            bins = pd.cut(time_index, bins=iindex)
            enb.loc[host_id, ["interval"]] = bins
        enb = enb.reset_index().merge(
            data["servingEnb"],
            how="left",
            left_on=["hostId", "interval"],
            right_on=["hostId", "interval"],
        )
        enb = enb.drop(columns=["interval"]).sort_values(["hostId", "time"])

        # create 2d-matrix for imshow
        enb_grid = (
            enb.set_index(["hostId", "time"])
            .unstack(["time"])
            .fillna(-1)
            .to_numpy()
            .astype(int)
        )

        # colormap
        max_enb = int(enb["servingEnb"].max())
        # ensure colors are not to faint, thus 1.5*....
        enb_c = plt.get_cmap("Reds")(np.linspace(0, 1, int(1.5 * max_enb)))
        cmap = colors.ListedColormap(["white", "black", *enb_c[-max_enb:]])

        img = ax.imshow(
            enb_grid, interpolation="nearest", origin="lower", cmap=cmap, aspect="equal"
        )
        ax.set_ylabel("Node indices")
        ax.set_xlabel("Time in seconds")
        ax.set_title(
            "Serving cell for nodes over time. \n (Black: NOT associated to any!)"
        )

        plt.colorbar(img, cmap=cmap, ticks=[-1, 0, max_enb])
        ax.get_figure().tight_layout()

        return ax.get_figure(), ax

    @with_axis
    def plot_node_enb_association_map(
        self, ue_position: pd.DataFrame, enb_position: pd.DataFrame, ax: plt.Axes = None
    ):
        # colormap
        max_enb = int(ue_position["servingEnb"].max())
        # ensure colors are not to faint, thus 1.5*....
        enb_c = plt.get_cmap("Reds")(np.linspace(0, 1, int(1.5 * max_enb)))
        cmap = colors.ListedColormap(
            [
                np.array([1.0, 1.0, 1.0, 1.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
                *enb_c[-max_enb:],
            ]
        )
        color_ar = np.array(cmap.colors)

        enb_color_index = ue_position["servingEnb"].to_numpy().astype(int) + 1

        _colors = color_ar[enb_color_index]

        ax.scatter(enb_position["x"], enb_position["y"], label="eNB", marker="s")
        ax.legend()
        lc = LineCollection(ue_position["segment"], cmap=cmap)
        lc.set_array(enb_color_index)
        line = ax.add_collection(lc)
        ax.get_figure().colorbar(line, ax=ax)

        for _host, pos in ue_position.groupby("hostId"):
            ax.scatter(pos.iloc[-1]["x"], pos.iloc[-1]["y"], color="red", marker=".")

        ax.set_title("Node traces by cell association (black: no association)")
        ax.set_ylabel("North in meter")
        ax.set_xlabel("East in meter")
        ax.get_figure().tight_layout()
        return ax.get_figure(), ax


PlotEnb = _PlotEnb()
