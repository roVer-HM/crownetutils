from __future__ import annotations

import os
from ast import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import IndexSlice as _i

import roveranalyzer.simulators.opp.scave as Scave
import roveranalyzer.utils.plot as p
from roveranalyzer.analysis.common import Simulation
from roveranalyzer.analysis.flaskapp.application.layout import IdProvider
from roveranalyzer.analysis.omnetpp import OppAnalysis
from roveranalyzer.simulators.opp.provider.hdf.IHdfProvider import BaseHdfProvider
from roveranalyzer.utils.logging import logger, timing
from roveranalyzer.utils.plot import (
    FigureSaver,
    FigureSaverSimple,
    PlotUtil_,
    percentile,
    with_axis,
)


class PlotAppTxInterval_(PlotUtil_):
    """Collection of plot methods for application transmission intervals"""

    def __init__(self) -> None:
        super().__init__()

    @timing
    def plot_txinterval_all(
        self,
        data_root: str,
        sql: Scave.CrownetSql,
        app: str = "Beacon",
        saver: FigureSaver | None = None,
    ):
        saver = FigureSaver.FIG(saver, FigureSaverSimple(data_root))
        data = OppAnalysis.get_txAppInterval(sql, app_type=app)
        data = data.droplevel(["hostId", "host"]).sort_index()
        if data.empty:
            logger.info(
                "No tx interval vectors found. Did you choose the correct scheduler in the Simulation?"
            )
            return
        fig, ax = self.df_to_table(
            data.describe().applymap("{:1.4f}".format).reset_index(),
            title=f"Descriptive statistics for application",
        )
        self.append_title(ax, prefix=f"{app}: ")
        saver(fig, f"{app}_tx_AppIntervall_stat.pdf")
        plt.close(fig)

        fig, ax = self.plot_ts_txinterval(data, app_name=app, time_bucket_length=1.0)
        self.append_title(ax, prefix=f"{app}: ")
        saver(fig, f"{app}_txAppInterval_ts.pdf")
        plt.close(fig)

        fig, ax = self.plot_hist_txinterval(data)
        self.append_title(ax, prefix=f"{app}: ")
        saver(fig, f"{app}_tx_AppInterval_hist_.pdf")
        plt.close(fig)

        fig, ax = self.plot_ecdf_txinterval(data)
        self.append_title(ax, prefix=f"{app}: ")
        saver(fig, f"{app}_tx_AppInterval_ecdf.pdf")
        plt.close(fig)

    def plot_ts_txinterval(
        self, data: pd.DataFrame, app_name="", time_bucket_length=1.0
    ):
        interval = pd.interval_range(
            start=0.0, end=np.ceil(data.index.max()), freq=time_bucket_length
        )
        data = data.groupby(pd.cut(data.index, interval)).mean()
        data.index = interval.left
        data.index.name = "time"
        cols = data.columns
        data = data.reset_index()
        fig, ax = self.check_ax()
        for c in cols:
            ax.plot("time", c, data=data, label=f"{c} {app_name}")
        ax.legend(loc="upper right")
        ax.set_title(
            "Average transmission interval of all nodes over time. (time bin size 1s)"
        )
        ax.set_xlabel("Time in seconds")
        ax.set_ylabel("Transmission time interval in seconds")
        return fig, ax

    def plot_hist_txinterval(self, data: pd.DataFrame):
        # use same bins for both data sets
        fig, ax = self.check_ax()
        _range = (data["txInterval"].min(), data["txInterval"].max())
        _bin_count = np.ceil(data["txInterval"].count() ** 0.5)
        _bins = np.histogram(data, bins=int(_bin_count))[1]
        for c in data.columns:
            ax.hist(data[c], bins=_bins, range=_range, density=True, alpha=0.5, label=c)
        ax.legend()
        ax.set_title("Histogram of transmission time interval in seconds ")
        ax.set_ylabel("Density")
        ax.set_xlabel("Transmission time interval in seconds")
        return fig, ax

    def plot_ecdf_txinterval(self, data: pd.DataFrame):
        fig, ax = self.check_ax()
        _x = data["txInterval"].sort_values().values
        _y = np.arange(len(_x)) / float(len(_x))
        ax.plot(_x, _y, label="txInterval")
        _x = data["txDetInterval"].sort_values().values
        _y = np.arange(len(_x)) / float(len(_x))
        ax.plot(_x, _y, label="txDetInterval")
        ax.set_title("ECDF of transmission interval time")
        ax.set_xlabel("Time in seconds")
        ax.set_ylabel("ECDF")
        ax.legend()
        return fig, ax


PlotAppTxInterval = PlotAppTxInterval_()


class PlotAppMisc_(PlotUtil_):
    @staticmethod
    def ts_mean(data: pd.DataFrame, time_bin=1.0, index="time", col="value"):
        if index in data.columns:
            data = data.reset_index().set_index(index)
        time_int = pd.interval_range(
            0.0, data.index.max(), freq=time_bin, closed="left"
        )
        data = data.loc[:, col].groupby(pd.cut(data.index, bins=time_int)).describe()
        data.reset_index().rename(columns={"index": "interval"})
        data["time"] = time_int.left
        return data

    @with_axis
    def plot_packet_size_ts(
        self, data: pd.DataFrame, *, ax: plt.Axes | None = None, **plot_args
    ):
        _i = pd.IndexSlice
        d = data.loc[_i[:, "b"], "value"]
        ax.scatter(
            d.index.get_level_values(0), d, marker="x", label="Beacon", **plot_args
        )

        d = data.loc[_i[:, "m"], "value"]
        ax.scatter(d.index.get_level_values(0), d, marker="+", label="Map", **plot_args)

        ax.set_ylabel("Packet size in bytes")
        ax.set_xlabel("Time in seconds")
        ax.set_title(
            "Packet size over time for all agents and all applications (Beacon and Map)"
        )
        ax.legend()
        return ax.get_figure(), ax

    @with_axis
    def plot_tx_throughput(
        self, data: pd.DataFrame, sql: Scave.CrownetSql, *, ax: plt.Axes | None = None
    ):
        marker = [".", "x"]
        for i, c in enumerate(data.columns):
            ax.scatter(data.index, data[c], marker=marker[i], label=c)

        map_max_bw = sql.get_run_parameter(
            sql.m_map(app_mod="scheduler"), name="maxApplicationBandwidth"
        )
        if not map_max_bw.empty:
            val = map_max_bw["paramValue"].unique()[0]
            if val.endswith("bps"):
                val = int(val[:-3]) / 8000
                ax.hlines(
                    val,
                    data.index.min(),
                    data.index.max(),
                    color="red",
                    label="Map max bandwidth",
                )

        beacon_max_bw = sql.get_run_parameter(
            sql.m_beacon(app_mod="scheduler"), name="maxApplicationBandwidth"
        )
        if not beacon_max_bw.empty:
            val = beacon_max_bw["paramValue"].unique()[0]
            if val.endswith("bps"):
                val = int(val[:-3]) / 8000
                ax.hlines(
                    val,
                    data.index.min(),
                    data.index.max(),
                    color="red",
                    linestyles="-.",
                    label="Beacon max bandwidth",
                )

        ax.legend()
        ax.set_title("Data rate based on sent packets in all from all nodes")
        ax.set_ylabel("Data rate in kBps")
        ax.set_xlabel("Time in seconds")
        return ax.get_figure(), ax

    def cmp_system_level_tx_rate_based_on_application_layer_data(
        self, sims: List[Simulation], *, saver: FigureSaver | None = None
    ):
        _, ax_pkt_size = self.check_ax()
        _, ax_app_t = self.check_ax()
        lbl_set = set()
        legend_dict = dict()
        for sim in sims:
            data = OppAnalysis.get_sent_packet_bytes_by_app(sim.sql)
            self.plot_packet_size_ts(data, ax=ax_pkt_size, alpha=0.7)
            _h = []
            _l = []
            for h, l in list(zip(*ax_pkt_size.get_legend_handles_labels())):
                if h not in legend_dict:
                    legend_dict[h] = f"{sim.label}: {l}"

            data = OppAnalysis.get_sent_packet_throughput_by_app(sim.sql, cache=data)
            self.plot_tx_throughput(data, sim.sql, ax=ax_app_t)
            for h, l in list(zip(*ax_app_t.get_legend_handles_labels())):
                if h not in legend_dict:
                    legend_dict[h] = f"{sim.label}: {l}"

        for a in [ax_pkt_size, ax_app_t]:
            _h = []
            _l = []
            for h, l in list(zip(*a.get_legend_handles_labels())):
                _h.append(h)
                _l.append(legend_dict[h])
            a.legend(_h, _l)

        saver(ax_pkt_size.get_figure(), "Packet_size_ts.pdf")
        saver(ax_app_t.get_figure(), "System_tx_data_rate.pdf")

    def plot_system_level_tx_rate_based_on_application_layer_data(
        self,
        sim: Simulation,
        *,
        create_hdf_cache: bool = True,
        saver: FigureSaver | None = None,
    ):
        """
        How many packets, at which size, and at what rate are produced by each application
        and in total.
        To get this data collect all 'packetSent:vector(packetBytes)' from all nodes and all applications.
        (1) Show a time series of packet size for each application
        (2) Show the throughput over time for each application and in total.
        (3) What are the number of neighbors/members used in the tx interval algorithm?
        """
        saver = FigureSaver.FIG(saver, FigureSaverSimple(sim.data_root))
        if create_hdf_cache:
            _hdf = sim.get_base_provider("tx_pkt_bytes", sim.path("tx_pkt_bytes.h5"))
        tx_pkt = OppAnalysis.get_sent_packet_bytes_by_app(
            sim.sql, hdf=_hdf, hdf_group="tx_pkt_bytes"
        )

        # (1) packet size t
        fig, _ = self.plot_packet_size_ts(data=tx_pkt)
        saver(fig, "Packet_size_ts.pdf")

        # (2) throughput
        tx_rate = OppAnalysis.get_sent_packet_throughput_by_app(sim.sql, cache=tx_pkt)
        fig, _ = self.plot_tx_throughput(tx_rate, sim.sql)
        saver(fig, "System_tx_data_rate.pdf")
        plt.close(fig)

    def plot_number_of_agents(
        self, sim: Simulation, *, saver: FigureSaver | None = None
    ):
        saver = FigureSaver.FIG(saver, FigureSaverSimple(sim.data_root))
        sql = sim.sql
        dmap = sim.builder.build_dcdMap()
        fig, ax = self.check_ax()

        nt_count = sql.vec_data(sql.m_table(), "tableSize:vector", drop="vectorId")
        node_count = (
            dmap.glb_map.groupby("simtime")
            .sum()
            .reset_index()
            .set_axis(["time", "value"], axis=1)
        )
        map_count = (
            dmap.map_count_measure()
            .loc[:, ["map_mean_count"]]
            .reset_index()
            .set_axis(["time", "value"], axis=1)
        )

        ax.scatter(
            "time",
            "value",
            data=nt_count,
            label="Neighborhood table node count",
            marker=".",
            s=2,
            color="r",
            alpha=0.3,
        )
        ax.scatter(
            "time",
            "value",
            data=node_count,
            label="Agent count (ground truth)",
            s=5,
            marker="v",
            color="k",
        )
        ax.scatter(
            "time",
            "value",
            data=map_count,
            s=5,
            marker="<",
            color="g",
            label="Mean map count",
        )

        l = ax.legend()
        for h in l.legendHandles:
            h.set_sizes([10])
        ax.set_ylabel("Number of members")
        ax.set_xlabel("Simulation time in seconds")
        ax.set_title("Number of members used in the tx interval algorithm over time")
        saver(fig, "Node_count_ts.png")
        plt.close(fig)

        fig, ax = self.check_ax()
        nt_count_d = self.ts_mean(nt_count.set_index("time")).dropna()

        self.fill_between(
            nt_count_d,
            x="time",
            val="mean",
            fill_val=["25%", "75%"],
            fill_alpha=0.35,
            line_args=dict(
                label="Neighborhood table node count", marker="x", color="r"
            ),
            fill_args=dict(label="Q1;Q3 neighborhood table node count"),
            ax=ax,
        )

        ax.scatter(
            "time",
            "value",
            data=node_count,
            label="Agent count (ground truth)",
            zorder=999,
            s=8,
            marker="v",
            color="k",
        )
        ax.scatter(
            "time",
            "value",
            data=map_count,
            s=8,
            marker="<",
            color="g",
            zorder=999,
            label="Mean map count",
        )
        l = ax.legend()
        for h in l.legendHandles:
            try:
                h.set_sizes([10])
            except AttributeError as e:
                pass
        ax.set_ylabel("Number of members")
        ax.set_xlabel("Simulation time in seconds")
        ax.set_title("Number of members used in the tx interval algorithm over time")
        saver(fig, "Mode_count_ts_mean.png")
        plt.close(fig)

    def get_jitter_delay_cached(
        self, sim: Simulation, hdf_path: str | None = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if hdf_path is not None and os.path.exists(hdf_path):
            delay = pd.read_hdf(hdf_path, key="delay")
            jitter = pd.read_hdf(hdf_path, key="jitter")
        else:
            delay = OppAnalysis.get_received_packet_delay(
                sql=sim.sql,
                drop_self_message=True,
                module_name=sim.sql.m_map(),
            )
            jitter = OppAnalysis.get_received_packet_jitter(
                sql=sim.sql, drop_self_message=True, module_name=sim.sql.m_map()
            )

            if hdf_path is not None:
                delay.to_hdf(hdf_path, mode="a", key="delay", format="table")
                jitter.to_hdf(hdf_path, mode="a", key="jitter", format="table")

        return delay, jitter

    def plot_application_delay_jitter(
        self, sim: Simulation, *, saver: FigureSaver | None = None
    ):
        saver = FigureSaver.FIG(saver)

        hdf = BaseHdfProvider(sim.path("rcvd_stats.h5"), "rcvd_stats")
        with hdf.ctx() as c:
            df = c.select(key="rcvd_stats", where="app=m", columns=["delay", "jitter"])

        df = df.reset_index()
        fig, ax = self.check_ax()
        ax.scatter(
            "time",
            "delay",
            data=df,
            marker="x",
            alpha=0.5,
            label="delay",
        )
        ax.scatter(
            "time",
            "jitter",
            data=df,
            marker="+",
            alpha=0.5,
            label="jitter",
        )
        ax.legend()
        ax.set_ylabel("Delay/Jitter in seconds")
        ax.set_xlabel("Simulation time in seconds")
        ax.set_title("Map: Delay and Jitter over time")
        saver(fig, "Map_delay_and_jitter.png")
        plt.close(fig)

        fig, ax = self.check_ax()
        self.ecdf(df["delay"], ax=ax, label="Delay")
        self.ecdf(df["jitter"], ax=ax, label="Jitter")
        ax.legend()
        ax.set_xlabel("Time in seconds")
        ax.set_title("CDF of jitter and delay")
        saver(fig, "Map_delay_and_jitter_ecdf.png")
        plt.close(fig)

        fig, ax = self.df_to_table(df[["delay", "jitter"]].describe().reset_index())
        saver(fig, "Map_delay_and_jitter_describe_tbl.png")

    def plot_pkt_loss(
        self,
        sim: Simulation | List[Simulation],
        saver: FigureSaver,
        app: str = "Both",  # Map, Beacon, Both
    ):
        sim = sim if isinstance(sim, list) else [sim]
        _, ax_ts = self.check_ax()
        _, ax_ecdf = self.check_ax()

        def lbl(_s, lbl):
            if len(sim) > 1:
                return f"{_s.label}: {lbl}"
            else:
                return lbl

        for s in sim:
            data = OppAnalysis.build_received_packet_loss_cache(
                s.sql, s.path("pkt_loss.h5"), return_group=app
            )
            if app == "Both":
                m = data.loc[_i[:, :, :, :, "Map"], "pkt_loss"]
                c = ax_ts.scatter(
                    m.index.get_level_values("time"), y=m, s=1, marker="x", alpha=0.35
                )
                c.set_label(lbl(s, "Map pkt loss"))
                self.ecdf(m, ax_ecdf, label=lbl(s, "Map pkt loss"))

                b = data.loc[_i[:, :, :, :, "Beacon"], ["pkt_loss"]]
                c = ax_ts.scatter(
                    b.index.get_level_values("time"), y=b, s=1, marker="<", alpha=0.35
                )
                c.set_label(lbl(s, "Beacon pkt loss"))
                self.ecdf(b, ax_ecdf, label=lbl(s, "Beacon pkt loss"), linestyle="--")
            else:
                c = ax_ts.scatter(
                    data.index.get_level_values("time"),
                    y=data["pkt_loss"],
                    s=1,
                    marker="x",
                    alpha=0.35,
                )
                if len(sim) > 1:
                    c.set_label(f"{s.label}: {app} pkt loss")
                else:
                    c.set_label(f"{app} pkt loss")

        ax_ts.legend(markerscale=6)
        ax_ts.set_xlabel("Simulation time in seconds")
        ax_ts.set_ylabel("Pkt loss encountered since last reception")
        ax_ts.set_title("Packet loss over time based on received packets")
        saver(ax_ts.get_figure(), "Pkt_loss_ts.pdf")
        ax_ecdf.legend()
        ax_ecdf.set_ylabel("Packet loss encountered sind last reception")
        ax_ecdf.set_title("Packet loss ECDF")
        saver(ax_ecdf.get_figure(), "Pkt_loss_ecdf.pdf")


PlotAppMisc = PlotAppMisc_()
