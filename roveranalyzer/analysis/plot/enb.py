from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import roveranalyzer.simulators.opp.scave as Scave
from roveranalyzer.analysis.omnetpp import OppAnalysis, Simulation
from roveranalyzer.utils.logging import logger, timing
from roveranalyzer.utils.plot import FigureSaver, _PlotUtil, with_axis


class _PlotEnb(_PlotUtil):
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
            self.append_title(ax, prefix="Enb-{n}:")
            saver(fig, os.path.join(data_root, f"rb_utilization_ts_{n}.pdf"))
            fig, ax = self.plot_hist_enb_served_rb(data, bins, n)
            self.append_title(ax, prefix="Enb-{n}:")
            saver(fig, os.path.join(data_root, f"rb_utilization_hist_{n}.pdf"))
            fig, ax = self.plot_ecdf_enb_served_rb(data, bins, n)
            self.append_title(ax, prefix="Enb-{n}:")
            saver(fig, os.path.join(data_root, f"rb_utilization_ecdf_{n}.pdf"))
            fig, ax = self.plot_tbl_enb_serverd_rb_count(data)
            self.append_title(ax, prefix="Enb-{n}:")
            saver(fig, os.path.join(data_root, f"rb_count_{n}.pdf"))
            fig, ax = self.df_to_table(
                data.describe().applymap("{:1.4f}".format).reset_index(),
                title="Served UL blocks",
            )
            self.append_title(ax, prefix="Enb-{n}:")
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
        fig, ax = self.df_to_table(df)
        ax.set_title("UL scheduling RB's count ")
        return fig, ax


PlotEnb = _PlotEnb()
