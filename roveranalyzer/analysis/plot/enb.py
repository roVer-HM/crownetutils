import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import roveranalyzer.simulators.opp.scave as Scave
from roveranalyzer.analysis.omnetpp import OppAnalysis
from roveranalyzer.utils.logging import logger, timing
from roveranalyzer.utils.plot import FigureSaver, _PlotUtil


class _PlotEnb(_PlotUtil):
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
            fig, _ = self.plot_ts_enb_served_rb(data, time_bucket_length=1.0)
            saver(fig, os.path.join(data_root, f"rb_utilization_ts_{n}.pdf"))
            fig, _ = self.plot_hist_enb_served_rb(data, bins, n)
            saver(fig, os.path.join(data_root, f"rb_utilization_hist_{n}.pdf"))
            fig, _ = self.plot_ecdf_enb_served_rb(data, bins, n)
            saver(fig, os.path.join(data_root, f"rb_utilization_ecdf_{n}.pdf"))
            fig, _ = self.plot_tbl_enb_serverd_rb_count(data)
            saver(fig, os.path.join(data_root, f"rb_count_{n}.pdf"))
            fig, _ = self.df_to_table(
                data.describe().applymap("{:1.4f}".format).reset_index(),
                title="Enb serverd UL blocks",
            )
            saver(fig, os.path.join(data_root, f"rb_stat_{n}.pdf"))

    def plot_ts_enb_served_rb(self, data: pd.DataFrame, time_bucket_length=1.0):
        interval = pd.interval_range(
            start=0.0, end=np.ceil(data.index.max()), freq=time_bucket_length
        )
        data = data.groupby(pd.cut(data.index, interval)).mean()
        data.index = interval.left
        data.index.name = "time"
        data = data.reset_index()
        fig, ax = self.check_ax()
        ax.plot("time", "value", data=data)
        ax.set_title("Average Resource Block (RB) usage over time. (time bin size 1s)")
        ax.set_xlabel("time in [s]")
        ax.set_ylabel("Resource blocks")
        # ax.set_ylim(0, bins+1)
        # ax.set_yticks(np.arange(0, bins+1, 1))
        return fig, ax

    def plot_hist_enb_served_rb(self, data: pd.DataFrame, bins=25, enb=0):
        ax: plt.Axes
        fig, ax = self.check_ax()
        data = data["value"]
        d = 1
        left_of_first_bin = 0 - float(d) / 2
        right_of_last_bin = bins + float(d) / 2
        ax.hist(
            data, np.arange(left_of_first_bin, right_of_last_bin + d, d), align="mid"
        )
        ax.set_xlim(-1, bins + 1)
        ax.set_xticks(np.arange(0, bins + 1, 1))
        ax.set_title(f"Resource block utilization of eNB {enb}")
        ax.set_xlabel("Resource Blocks (RB's)")
        ax.set_ylabel("Count")
        return fig, ax

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
