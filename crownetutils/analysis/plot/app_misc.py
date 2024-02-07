"""Application layer metrics (throughput, delay, tx interval)"""

from __future__ import annotations

import itertools
import os
from ast import List
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from pandas import IndexSlice as _i

import crownetutils.omnetpp.scave as Scave
from crownetutils.analysis.common import Simulation
from crownetutils.analysis.dpmm.dpmm_cfg import DpmmCfgDb
from crownetutils.analysis.hdf.provider import BaseHdfProvider, HdfSelector
from crownetutils.analysis.hdf_providers.node_position import NodePositionWithRsdHdf
from crownetutils.analysis.hdf_providers.node_rx_data import NodeRxData
from crownetutils.analysis.hdf_providers.node_tx_data import NodeTxData
from crownetutils.analysis.hdf_providers.sql_app_proxy import SqlAppProxy
from crownetutils.analysis.omnetpp import OppAnalysis
from crownetutils.omnetpp.sql import SqlOp
from crownetutils.utils.dataframe import (
    append_interval,
    assert_frame_structure,
    combine_stats,
    index_or_col,
)
from crownetutils.utils.logging import logger, timing
from crownetutils.utils.plot import FigureSaver, FigureSaverSimple, PlotUtil_, with_axis


class PlotAppTxInterval_(PlotUtil_):
    """Collection of plot methods for application transmission intervals"""

    def __init__(self) -> None:
        super().__init__()

    def plot_txinterval_all_beacon(
        self,
        data_root: str,
        sql: Scave.CrownetSql,
        saver: FigureSaver | None = None,
    ):
        return self.plot_txinterval_all(
            data_root=data_root,
            sql=sql,
            module_name=sql.m_beacon(path="scheduler"),
            app_name="Beacon",
            saver=saver,
        )

    def plot_txinterval_all_map(
        self,
        data_root: str,
        sql: Scave.CrownetSql,
        saver: FigureSaver | None = None,
    ):
        return self.plot_txinterval_all(
            data_root=data_root,
            sql=sql,
            module_name=sql.m_map(path="scheduler"),
            app_name="Map",
            saver=saver,
        )

    def plot_txinterval_from_hdf_all(
        self,
        data: NodeTxData,
        app_name: str,
        data_root: str,
        saver: FigureSaver | None = None,
    ):
        saver = FigureSaver.FIG(saver, FigureSaverSimple(data_root))
        df: pd.DataFrame = data.tx_interval(app=app_name).frame()
        df = df.reset_index().set_index(["servingEnb", "hostId", "time"]).sort_index()
        df = df.rename(
            columns={"tx_interval": "txInterval", "tx_interval_det": "txDetInterval"}
        )
        rsds = df.index.get_level_values("servingEnb").unique().to_list()
        rsds.sort()
        for rsd in rsds:
            _cm = 1 / 2.54
            fig, axes = plt.subplots(2, 2, figsize=(29.7 * _cm, 21.0 * _cm))
            axes = axes.flatten()
            _df = df.loc[rsd, ["txDetInterval", "txInterval"]].copy()

            # 1/4
            self.df_to_table(
                df=_df.describe().applymap("{:1.4f}".format).reset_index(),
                title=f"Descriptive statistics for application",
                ax=axes[0],
            )
            # 2/4
            self.plot_ts_txinterval(
                data=_df, app_name=app_name, time_bucket_length=1.0, ax=axes[1]
            )
            self.append_title(axes[1], prefix=f"{app_name}: ")
            # 3/4
            self.plot_hist_txinterval(data=_df, ax=axes[2])
            self.append_title(axes[2], prefix=f"{app_name}: ")
            # 4/4
            self.plot_ecdf_txinterval(data=_df, ax=axes[3])
            self.append_title(axes[3], prefix=f"{app_name}: ")

            fig.tight_layout()
            saver(fig, f"{app_name}_tx_AppIntervall_rsd_{rsd}.png", dpi=300)
            plt.close(fig)

    @timing
    def plot_txinterval_all(
        self,
        data_root: str,
        sql: Scave.CrownetSql,
        module_name: SqlOp,
        app_name: str,
        saver: FigureSaver | None = None,
    ):
        saver = FigureSaver.FIG(saver, FigureSaverSimple(data_root))
        data = OppAnalysis.get_txAppInterval(sql, module_name=module_name)
        data = data.droplevel("hostId").sort_index()
        if "host" in data.columns:
            data = data.drop(columns=["host"])
        if data.empty:
            logger.info(
                "No tx interval vectors found. Did you choose the correct scheduler in the Simulation?"
            )
            return
        fig, ax, _ = self.df_to_table(
            data.describe().applymap("{:1.4f}".format).reset_index(),
            title=f"Descriptive statistics for application",
        )
        self.append_title(ax, prefix=f"{app_name}: ")
        saver(fig, f"{app_name}_tx_AppIntervall_stat.pdf")
        plt.close(fig)

        fig, ax = self.plot_ts_txinterval(
            data, app_name=app_name, time_bucket_length=1.0
        )
        self.append_title(ax, prefix=f"{app_name}: ")
        saver(fig, f"{app_name}_txAppInterval_ts.pdf")
        plt.close(fig)

        fig, ax = self.plot_hist_txinterval(data)
        self.append_title(ax, prefix=f"{app_name}: ")
        saver(fig, f"{app_name}_tx_AppInterval_hist_.pdf")
        plt.close(fig)

        fig, ax = self.plot_ecdf_txinterval(data)
        self.append_title(ax, prefix=f"{app_name}: ")
        saver(fig, f"{app_name}_tx_AppInterval_ecdf.pdf")
        plt.close(fig)

    def plot_ts_txinterval(
        self,
        data: pd.DataFrame,
        app_name="",
        time_bucket_length=1.0,
        ax: plt.Axes = None,
    ):
        data = append_interval(frame=data.copy(), interval_range=time_bucket_length)
        data = data.groupby("time_bin").mean()
        time = [i.left for i in data.index]
        # interval = pd.interval_range(
        #     start=0.0, end=np.ceil(data.index.max()), freq=time_bucket_length
        # )
        # data = data.groupby(pd.cut(data.index, interval)).mean()
        # data.index = interval.left
        # data.index.name = "time"
        cols = data.columns
        # data = data.reset_index()
        fig, ax = self.check_ax(ax)
        for c in cols:
            ax.plot(time, data[c], label=f"{c} {app_name}")
        ax.legend(loc="upper right")
        ax.set_title(
            "Average transmission interval of all nodes over time. (time bin size 1s)"
        )
        ax.set_xlabel("Time in seconds")
        ax.set_ylabel("Transmission time interval in seconds")
        self.auto_major_minor_locator(ax)
        return fig, ax

    def plot_hist_txinterval(self, data: pd.DataFrame, ax: plt.Axes = None):
        # use same bins for both data sets
        fig, ax = self.check_ax(ax)
        _range = (data["txInterval"].min(), data["txInterval"].max())
        _bin_count = np.ceil(data["txInterval"].count() ** 0.5)
        _bins = np.histogram(data, bins=int(_bin_count))[1]
        for c in data.columns:
            ax.hist(data[c], bins=_bins, range=_range, density=True, alpha=0.5, label=c)
        ax.legend()
        ax.set_title("Histogram of transmission time interval in seconds ")
        ax.set_ylabel("Density")
        ax.set_xlabel("Transmission time interval in seconds")
        self.auto_major_minor_locator(ax)
        return fig, ax

    def plot_ecdf_txinterval(self, data: pd.DataFrame, ax: plt.Axes = None):
        fig, ax = self.check_ax(ax)
        _x = data["txInterval"].sort_values().values
        _y = np.arange(len(_x)) / float(len(_x))
        ax.plot(_x, _y, label="txInterval")
        _x = data["txDetInterval"].sort_values().values
        _y = np.arange(len(_x)) / float(len(_x))
        ax.plot(_x, _y, label="txDetInterval")
        ax.set_title("ECDF of transmission interval time")
        ax.set_xlabel("Time in seconds")
        ax.set_ylabel("ECDF")
        self.auto_major_minor_locator(ax)
        ax.legend()
        return fig, ax

    def plot_application_tx_time_hist_ecdf(
        self,
        tx_data: NodeTxData,
        saver: FigureSaver | None = None,
    ):
        data = tx_data.frame_by_app("tx_bytes", columns=["time"])

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
        ax = axes[0]
        ax, ecdf_x, _ = self.plot_ecdf(data, column="time", ax=ax, return_data=True)
        ax.set_xlabel("Simulation time in seconds")
        ax.set_ylabel("ecdf")
        ax.set_title(
            "ECDF of transmision times over time (all nodes and appllications)"
        )
        ax = axes[1]
        ax.hist(ecdf_x)
        ax.set_title("Hisogram of tx times (all nodes and applications)")
        ax.set_ylabel("count")
        ax.set_xlabel("Simulation time in seconds")
        self.auto_major_minor_locator(axes)
        FigureSaver.FIG(saver)(fig, "application_tx_time_hist_ecdf.png")

    def plot_app_tx_throughput(
        self,
        hdf: NodeTxData,
        rsd_colors: dict,
        target_rates_in_Bps: dict,
        bin_size: float = 10.0,
        saver: FigureSaver | None = None,
    ):
        saver = FigureSaver.FIG(saver)

        fig, axes = plt.subplots(
            nrows=len(target_rates_in_Bps.keys()),
            ncols=1,
            figsize=(16, 9 * len(target_rates_in_Bps.keys())),
        )

        t_max = 0
        if 0 in rsd_colors:
            del rsd_colors[0]
        for rsd, color in rsd_colors.items():
            data = hdf.get_tx_throughput_diff_by_app(
                target_rates=target_rates_in_Bps,
                bin_size=bin_size,
                throughput_unit=1000,  # in kilo bytes (applied to target rates and data)
                serving_enb=rsd,
            ).reset_index()
            for ax, (app, target_rate) in zip(axes, target_rates_in_Bps.items()):
                ax.plot("time", app, data=data, label=f"rsd {rsd}", color=color)

            _time_max = data["time"].max()
            if _time_max > t_max:
                t_max = _time_max

        for ax, (app, target_rate) in zip(axes, target_rates_in_Bps.items()):
            ax: plt.Axes
            ax.hlines(
                target_rate / (1000),
                0,
                t_max,
                color="red",
                label=f"target rate {app}",
            )
            ax.set_ylabel("throughput in kB/s")
            ax.set_title(app)
            ax.set_xlabel("Simulation time in seconds")
            self.auto_major_minor_locator(ax)
            ax.legend(ncol=2)

        ax: plt.axes = axes[-1]
        fig.tight_layout()
        saver(fig, "application_throughput_by_rsd.png")


PlotAppTxInterval = PlotAppTxInterval_()


class PlotAppMisc_(PlotUtil_):
    @staticmethod
    def ts_mean(data: pd.DataFrame, time_bin=1.0, index="time", col="value"):
        if index in data.columns:
            data = data.reset_index().set_index(index)
        time_int = pd.interval_range(
            start=0.0,
            end=max(time_bin, data.index.max()),  # ensure at least one interval range.
            freq=time_bin,
            closed="left",
        )
        data = data.loc[:, col].groupby(pd.cut(data.index, bins=time_int)).describe()
        data.reset_index().rename(columns={"index": "interval"})
        data["time"] = time_int.left
        return data

    @with_axis
    def plot_packet_size_ts(
        self, data: pd.DataFrame, *, ax: plt.Axes | None = None, **plot_args
    ):
        assert_frame_structure(data, ["app", "time"], column_names=["value"])
        _i = pd.IndexSlice
        d = data.loc[_i["b", :], "value"]
        ax.scatter(
            d.index.get_level_values(1), d, marker="x", label="Beacon", **plot_args
        )

        d = data.loc[_i["m", :], "value"]
        ax.scatter(d.index.get_level_values(1), d, marker="+", label="Map", **plot_args)

        ax.set_ylabel("Packet size in bytes")
        ax.set_xlabel("Time in seconds")
        ax.set_title(
            "Packet size over time for all agents and all applications (Beacon and Map)"
        )
        self.auto_major_minor_locator(ax)
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
            sql.m_map(path="scheduler"), name="maxApplicationBandwidth"
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
            sql.m_beacon(path="scheduler"), name="maxApplicationBandwidth"
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
        self.auto_major_minor_locator(ax)
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
        tx_rate = OppAnalysis.get_sent_packet_throughput_by_app(
            sim.sql, tx_byte_data=tx_pkt
        )
        fig, _ = self.plot_tx_throughput(tx_rate, sim.sql)
        saver(fig, "System_tx_data_rate.pdf")
        plt.close(fig)

    @with_axis
    def plot_packet_source_distribution(
        self,
        data: pd.DataFrame,
        hatch_patterns: List[str] = PlotUtil_._hatch_patterns,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot packet source distribution

        Args:
            data (pd.DataFrame):     See OppAnalysis.get_packet_source_distribution(...)
            ax (plt.Axes, optional): Axes to use. If missing a new axes will be injected by
                                     PlotUtil.with_axis decorator.

        Returns:
            plt.Axes:
        """
        patterns = itertools.cycle(hatch_patterns)

        ax = data.plot.barh(stacked=True, width=0.5, ax=ax)
        ax.set_title("Packets received from")
        ax.set_xlabel("percentage")
        bars = [i for i in ax.containers if isinstance(i, mpl.container.BarContainer)]
        for bar in bars:
            _h = next(patterns)
            for patch in bar:
                patch.set_hatch(_h)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        return ax.get_figure(), ax

    @with_axis
    def plot_neighborhood_table_size_over_time(
        self, tbl: pd.DataFrame, tbl_idx: np.ndarray, ax: plt.Axes = None
    ) -> plt.Axes:
        """Plot neighborhood table size for each node over time.
        x-axis: time
        y-axis: number of entries in neighborhood table

        Args:
            tbl (pd.DataFrame): Data see OppAnalysis.get_neighborhood_table_size_over_time()
            tbl_idx (np.ndarray): see OppAnalysis.get_neighborhood_table_size_over_time()
            ax (plt.Axes, optional): Axes to use. If missing a new axes will be injected by
                                     PlotUtil.with_axis decorator.

        Returns:
            plt.Axes:
        """
        _c = self.color_lines(line_type=None)
        for i in tbl_idx:
            _df = tbl.loc[tbl["host"] == i]
            ax.plot(_df["time"], _df["value"], next(_c), label=i)

        ax.set_ylabel("Neighboor count")
        ax.set_xlabel("time [s]")
        ax.set_title("Size of neighborhood table over time for each node")
        return ax.get_figure(), ax

    @with_axis
    def _plot_nt_map_comparison_scatter(
        self,
        node_count: pd.DataFrame,
        nt_count: pd.DataFrame,
        map_count: pd.DataFrame,
        ax: plt.Axes = None,
    ):
        fig, ax = self.check_ax(ax)

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
        ax.set_ylabel("Number of nodes")
        ax.set_xlabel("Simulation time in seconds")
        ax.set_title(
            "Number of nodes seen by neighborhood table and density map over time"
        )
        self.auto_major_minor_locator(ax)
        return fig, ax

    @with_axis
    def _plot_nt_map_comparison_mean_fill(
        self,
        node_count: pd.DataFrame,
        nt_count_d: pd.DataFrame,
        map_count: pd.DataFrame,
        ax: plt.Axes = None,
    ):
        fig, ax = self.check_ax(ax)

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
        ax.set_ylabel("Number of nodes")
        ax.set_xlabel("Simulation time in seconds")
        ax.set_title(
            "Number of nodes seen by neighborhood table and density map over time"
        )
        self.auto_major_minor_locator(ax)
        return fig, ax

    def plot_nt_map_comparision_by_rsd(
        self,
        sim: Simulation,
        pos: NodePositionWithRsdHdf,
        *,
        saver: FigureSaver | None = None,
    ):
        """Plots time series of neighborhood table count with map based count for each RSD.
        only use data of nodes which are located in the RSD during measurement time.

        Args:
            sim (Simulation): _description_
            pos (NodePositionWithRsdHdf): _description_
            saver (FigureSaver | None, optional): _description_. Defaults to None.
        """

        saver: FigureSaverSimple = FigureSaver.FIG(
            saver, FigureSaverSimple(sim.data_root)
        )
        sql = sim.sql
        dmap = sim.builder.build_dcdMap()

        nt_count = sql.vector_ids_to_host(
            module_name=sql.m_table(),
            vector_name="tableSize:vector",
            name_columns=["hostId"],
            pull_data=True,
            drop="vectorId",
        )
        nt_count = (
            pos.merge_rsd_id_on_host_time_interval(nt_count)
            .drop(columns=["hostId"])
            .set_index(["servingEnb", "time"])
            .sort_index()
        )

        # node_count = (
        #     dmap.glb_map.groupby("simtime")
        #     .sum()
        #     .reset_index()
        #     .set_axis(["time", "value"], axis=1)
        # )
        map_count = dmap.map_count_measure_by_rsd(local_data_only=True).loc[
            :, ["map_mean_count", "map_glb_count"]
        ]
        for rsd in pos.enb.frame()["rsd_id"].sort_values().to_list():
            _nt_count = nt_count.loc[rsd].copy()
            _map_count = (
                map_count.loc[pd.IndexSlice[:, rsd], ["map_mean_count"]]
                .droplevel(1)
                .reset_index()
                .set_axis(["time", "value"], axis=1)
                .copy()
            )
            _node_count = (
                map_count.loc[pd.IndexSlice[:, rsd], ["map_glb_count"]]
                .droplevel(1)
                .reset_index()
                .set_axis(["time", "value"], axis=1)
                .copy()
            )

            fig, ax = self._plot_nt_map_comparison_scatter(
                _node_count.reset_index(),
                _nt_count.reset_index(),
                _map_count.reset_index(),
            )
            ax.set_title(f"{ax.get_title()} rsd {rsd}")
            saver.with_suffix(f"_rsd{rsd}")(fig, "nt_map_node_count_ts.png")

            _nt_count_d = self.ts_mean(_nt_count).dropna()

            fig, _ = self._plot_nt_map_comparison_mean_fill(
                _node_count.reset_index(),
                _nt_count_d.reset_index(),
                _map_count.reset_index(),
            )
            saver.with_suffix(f"_rsd{rsd}")(fig, "Node_count_ts_mean.png")
            ax.set_title(f"{ax.get_title()} rsd {rsd}")

    def plot_nt_map_comparison(
        self, sim: Simulation, *, saver: FigureSaver | None = None
    ):
        """Plots timeseries of neighorhood table count with map based count.

        Args:
            sim (Simulation): _description_
            saver (FigureSaver | None, optional): _description_. Defaults to None.
        """
        saver = FigureSaver.FIG(saver, FigureSaverSimple(sim.data_root))
        sql = sim.sql
        dmap = sim.builder.build_dcdMap()

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

        fig, _ = self._plot_nt_map_comparison_scatter(node_count, nt_count, map_count)
        saver(fig, "nt_map_node_count_ts.png")

        nt_count_d = self.ts_mean(nt_count.set_index("time")).dropna()

        fig, _ = self._plot_nt_map_comparison_mean_fill(
            node_count, nt_count_d, map_count
        )
        saver(fig, "nt_map_node_ts_mean.png")

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
        self,
        sim: Simulation,
        *,
        hdf_selector: HdfSelector = None,
        saver: FigureSaver | None = None,
    ):
        saver = FigureSaver.FIG(saver)

        if hdf_selector is None:
            hdf_selector = HdfSelector.from_path(
                sim.path("rcvd_stats.h5"),
                "rcvd_stats",
                where="app=m",
                columns=["delay", "jitter"],
            )
        # hdf = BaseHdfProvider(sim.path("rcvd_stats.h5"), "rcvd_stats")
        with hdf_selector.hdf.ctx() as c:
            # df = c.select(key="rcvd_stats", where="app=m", columns=["delay", "jitter"])
            df = c.select(**hdf_selector.select_args())

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
        ax.set_title("Delay and Jitter over time")
        self.auto_major_minor_locator(ax)
        saver(fig, "delay_and_jitter.png")
        plt.close(fig)

        fig, ax = self.check_ax()
        self.plot_ecdf(df["delay"], ax=ax, label="Delay")
        self.plot_ecdf(df["jitter"], ax=ax, label="Jitter")
        ax.legend()
        ax.set_xlabel("Time in seconds")
        ax.set_title("CDF of jitter and delay")
        self.auto_major_minor_locator(ax)
        saver(fig, "delay_and_jitter_ecdf.png")
        plt.close(fig)

        fig, ax, tbl = self.df_to_table(
            df[["delay", "jitter"]].describe().reset_index()
        )
        self.auto_major_minor_locator(ax)
        saver(fig, "delay_and_jitter_describe_tbl.png")
        plt.close(fig)

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
                self.plot_ecdf(m, ax_ecdf, label=lbl(s, "Map pkt loss"))

                b = data.loc[_i[:, :, :, :, "Beacon"], ["pkt_loss"]]
                c = ax_ts.scatter(
                    b.index.get_level_values("time"), y=b, s=1, marker="<", alpha=0.35
                )
                c.set_label(lbl(s, "Beacon pkt loss"))
                self.plot_ecdf(
                    b, ax_ecdf, label=lbl(s, "Beacon pkt loss"), linestyle="--"
                )
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

    def plot_by_app_overview(
        self,
        saver: FigureSaverSimple,
        sim_id: str,
        node_rx: NodeTxData,
        node_tx: NodeRxData,
        pos: NodePositionWithRsdHdf,
        d_sim: Simulation,
        apps: List[SqlAppProxy],
        rsd_filter: List[int],
    ):
        """Generates a matrix of plots for debugging application delays for *one* RSD.
        Each row depicts one application provided by `apps`. The generated figures in each
        column are:
          1. Transmission intervals over time (raw data, scatter)
          2. Number of nodes used to calculated tx interval over time(raw data, scatter and ground truth)
          3. Size of transmitted messages in bytes (i.e. burst of one or more packets) ove time (raw data, scatter)
          4. Size of burst (i.e. number of packets in one burst) over time (raw data, scatter)
          5. Size of mean message size, calculated as exponential moving average in each node over time.
             This message size is used in the calculation of the tx interval in figure 1
          6. Received data at application layer in one second over time. todo: why is this so low??
          7. Number of different data sources over one second over time.


        Args:
            saver (FigureSaverSimple): Object to determine location and type (png/pdf) of saved figure
            sim_id (str): Simulation label
            node_rx (NodeTxData): TX data / HDF
            node_tx (NodeRxData): RX data / HDF
            pos (NodePositionWithRsdHdf): Position data / HDF
            d_sim (Simulation): Simulation object with access to density map and SQLite db's
            apps (List[SqlAppProxy]): List of appications and sqlite module functions.
            rsd_filter (List[int]): List of RSD for which a figure should be crated.
        """
        rsd_colors = pos.enb_colors()
        for rsd in rsd_filter:
            num_plots = 8
            fig, axes = plt.subplots(
                len(apps), num_plots, figsize=((16 / 2) * num_plots, 16), sharex=True
            )

            app: SqlAppProxy
            for i, app in enumerate(apps):
                a1: plt.Axes = axes[i][0]
                a2: plt.Axes = axes[i][1]
                a3: plt.Axes = axes[i][2]
                a4: plt.Axes = axes[i][3]
                a4_max: plt.Axes = axes[i][4]
                a5: plt.Axes = axes[i][5]
                a6: plt.Axes = axes[i][6]
                a7: plt.Axes = axes[i][7]
                color = rsd_colors[rsd]
                glb_count = d_sim.get_dcdMap()._map_p.select(
                    key="map_measure_by_rsd",
                    where=f"rsd_id={rsd}",
                    columns=["map_glb_count"],
                )

                tx_data = (
                    node_tx.tx_interval(app)
                    .select(where=f"servingEnb={rsd}")
                    .reset_index()
                    .sort_values(["time"])
                )

                det_interval = (
                    tx_data[["time", "tx_interval_det"]]
                    .groupby(["time"])
                    .mean()
                    .reset_index()
                )
                a1.scatter(
                    index_or_col(tx_data, "time"),
                    tx_data["tx_interval"],
                    marker="1",
                    color=color,
                    alpha=0.3,
                    label="interval",
                )
                a1.scatter(
                    det_interval["time"],
                    det_interval["tx_interval_det"],
                    color="red",
                    alpha=0.3,
                    label="det_interval",
                    marker="3",
                )
                a1.set_ylabel("Tx interval in seconds")

                a2.scatter(
                    index_or_col(tx_data, "time"),
                    tx_data["member_count"],
                    marker="3",
                    color=color,
                    alpha=0.3,
                    label="member",
                )
                a2.plot(
                    index_or_col(glb_count, "simtime"),
                    glb_count["map_glb_count"],
                    color="red",
                    label="ground truth",
                )
                a2.set_ylabel("node count used in interval calc.")

                burst_data = (
                    node_tx.tx_burst(app)
                    .select(where=f"servingEnb={rsd}")
                    .reset_index()
                    .sort_values(["time"])
                )
                a3.scatter(
                    burst_data["time"],
                    burst_data["burst_size"],
                    marker="3",
                    color=color,
                    alpha=0.3,
                )
                a3.set_ylabel("TX burst size in bytes")

                a4.scatter(
                    burst_data["time"],
                    burst_data["burst_num"],
                    marker="3",
                    color=color,
                    alpha=0.3,
                )
                a4.set_ylabel("TX pkt count in burst")

                data = app.call_cb("map_size", default=None)
                if data is not None:
                    a4_max.scatter(
                        data["simtime"],
                        np.ceil(
                            data["numberOfCells"] / 114
                        ),  # at most 114 cells per packet MTU 1400 Bytes
                        marker="3",
                        color=color,
                        alpha=0.3,
                    )
                    a4_max.set_ylabel("TX pkt count in burst\n(114 cells per pkt.)")
                else:
                    a4_max.set_axis_off()
                    a4_max.text(0.0, 0.4, "No data.", transform=a4_max.transAxes)

                # RX
                data_rx_by_app = node_rx.rcvd_by_app(app).select(
                    where=f"servingEnb={rsd}"
                )
                a5.scatter(
                    index_or_col(data_rx_by_app, "time"),
                    data_rx_by_app["avg_pkt_size"] / 8,
                    color=rsd_colors[rsd],
                    label=rsd,
                    marker="3",
                    alpha=0.3,
                )
                a5.set_ylabel("RX exp. moving average \nburst size in bytes")

                data_rx: pd.DataFrame = node_rx.rcvd_data(app).select(
                    where=f"servingEnb={rsd}", columns=["pkt_bytes"]
                )
                if not data_rx.empty:
                    data_rx_by_src = data_rx.reset_index().drop(
                        columns=["srcHostId", "eventNumber"]
                    )
                    data_rx_by_src = self.append_bin(
                        data_rx_by_src, idx_name="time", bin_size=1.0, start=0.0
                    )
                    # received sum of packets per app per 1 second
                    data_rx_by_src = (
                        data_rx_by_src.reset_index(drop=True)
                        .groupby(["hostId", "bin"])
                        .sum()
                    )
                    data_rx_by_src["time"] = [
                        x.left for x in data_rx_by_src.index.get_level_values("bin")
                    ]
                    a6.scatter(
                        data_rx_by_src["time"],
                        data_rx_by_src["pkt_bytes"],
                        color=rsd_colors[rsd],
                        label=rsd,
                        marker="3",
                        alpha=0.3,
                    )
                    a6.set_ylabel("Rx bytes in 1 second ")

                    rx_different_sorces_over_time = self.append_bin(
                        data_rx.reset_index()[["time", "hostId", "srcHostId"]],
                        idx_name="time",
                        bin_size=1.0,
                        start=0.0,
                    ).reset_index(drop=True)
                    rx_different_sorces_over_time[
                        "bin"
                    ] = rx_different_sorces_over_time["bin"].apply(lambda x: x.left)
                    rx_different_sorces_over_time = (
                        rx_different_sorces_over_time.drop_duplicates()
                    )
                    rx_different_sorces_over_time = (
                        rx_different_sorces_over_time.groupby(["bin", "hostId"])
                        .count()
                        .reset_index()
                        .set_axis(["time", "hostId", "rx_count"], axis=1)
                    )
                    a7.scatter(
                        rx_different_sorces_over_time["time"],
                        rx_different_sorces_over_time["rx_count"],
                        color=color,
                        marker="3",
                        alpha=0.3,
                    )
                    a7.set_ylabel("Number of different rx sources \nin 1 second")
                else:
                    a6.text(0.5, 0.5, "No Data.", transform=a6.transAxes)
                    a6.set_ylabel("Rx bytes in 1 second ")
                    a7.text(0.5, 0.5, "No Data.", transform=a6.transAxes)
                    a7.set_ylabel("Number of different rx sources \nin 1 second")

                self.add_eng_formatter(
                    [
                        a3,
                        a5,
                        a6,
                    ],
                    unit="B",
                    places=1,
                )
                for a in axes[i]:
                    a.set_title(app.name)
                    self.auto_major_minor_locator(a)
                    a.set_xlabel("time in seconds")
                    a.legend()

            fig.suptitle(f"Debug for rsd {rsd} for sim {sim_id}")
            fig.tight_layout()
            saver.with_suffix(f"_rsd{rsd}_sim_{sim_id}")(fig, "dbg_interval.png")

    def plot_planed_throughput_in_rsd(
        self,
        saver: FigureSaverSimple,
        sim_id: str,
        node_tx: NodeRxData,
        apps: List[SqlAppProxy],
        rsd_filter: List[int],
    ):
        """Plot planed throughput of each application and the application and basestation thoughput limits.
           The *planed* throughput is the amount of data send from the application layer down to the stack,
           not knowing if it will be queued or removed in case of full queues or delayed deue to overload
           situation.

        Args:
            saver (FigureSaverSimple): Object to determine location and type (png/pdf) of saved figure
            sim_id (str): Simulation label
            node_tx (NodeRxData): RX data / HDF
            apps (List[SqlAppProxy]): List of appications and sqlite module functions.
            rsd_filter (List[int]): List of RSD for which a figure should be crated.
        """
        for rsd in rsd_filter:
            fig, axes = plt.subplots(1, 1, figsize=(16, 9), sharex=True)

            sum_burst = []
            t_rate = node_tx.get_target_rates(
                bps_to_multiplier=1 / 8
            )  # in Bytes per Second
            for i, app in enumerate(apps):
                a1: plt.Axes = axes

                burst_data = (
                    node_tx.tx_burst(app)
                    .select(where=f"servingEnb={rsd}")
                    .reset_index()
                    .sort_values(["time"])
                )
                app_burst = self.append_bin(
                    burst_data, idx_name="time", bin_size=1.0, start=0.0, agg=["sum"]
                )[["bin_left", "burst_size_sum"]].reset_index(drop=True)
                l = a1.plot(
                    app_burst["bin_left"], app_burst["burst_size_sum"], label=app.name
                )
                a1.hlines(
                    t_rate[app.name],
                    0,
                    app_burst["bin_left"].max(),
                    linestyles="dashed",
                    color=l[0].get_color(),
                    label=f"limit {app.name}",
                )
                sum_burst.append(app_burst)

            sum_burst = (
                pd.concat(sum_burst, axis=0)
                .groupby("bin_left")
                .sum()
                .reset_index()
                .set_axis(["time", "value"], axis=1)
            )
            l = a1.plot(
                sum_burst["time"],
                sum_burst["value"],
                label="cummulative application load",
            )
            max_enb_tp = 25 * 208 * 1000 / 8
            a1.hlines(
                max_enb_tp,
                0,
                sum_burst["time"].max(),
                linestyles="dashed",
                color=l[0].get_color(),
                label=f"Enb max throughput at CQI 7",
            )

            a1.set_xlabel("simulation time in seconds")
            a1.set_ylabel("Planned thoughput per second in bytes.")
            a1.set_title(
                f"Total planed thoughput based on appliction TX data for RSD {rsd} of Simulation {sim_id}"
            )
            a1.legend()
            self.add_eng_formatter(a1, unit="B", places=2)
            self.auto_major_minor_locator(a1)
            fig.tight_layout()
            saver.with_suffix(f"_rsd{rsd}_sim_{sim_id}")(
                fig, "dbg_total_planed_trasmitted_data.png"
            )

    def plot_received_bursts(
        self,
        node_rx: NodeRxData,
        figuer_title,
        saver: FigureSaver,
        where: str | None = None,
    ):
        """Plot number of packets received from the same source at the same time. This is
        indicates that the sender queues messages for at least as long as the caculated
        transmission intervall. For the beaocn/map data each new packet will contain newer
        information which will invalidate older one, leading to old data to be transmitted
        before fresher data. This not wanted and is not good.

        Args:
            node_rx (NodeRxData): _description_
            node_tx (NodeRxData): RX data / HDF
            saver (FigureSaverSimple): Object to determine location and type (png/pdf) of saved figure
            where (str | None, optional): Query string to filter data (i.e. resource sharing id or time frame). Defaults to None.
        """
        apps = node_rx.hdf.get_groups()
        fig, ax = plt.subplots(len(apps), 4, figsize=(16 / 3 * 4, 16))
        count_max = 3
        delay_max = 0
        for app_idx, app in enumerate(apps):
            if len(apps) == 1:
                ahist, acdf, adesc, adelay = ax
            else:
                ahist, acdf, adesc, adelay = ax[app_idx]

            if where is None:
                b = node_rx.rcvd_data(app=app).frame()
            else:
                b = node_rx.rcvd_data(app=app).select(where=where)

            same_time_delivery = b.groupby(["hostId", "srcHostId", "time"])[
                "delay"
            ].count()
            if same_time_delivery.max() > count_max:
                count_max = same_time_delivery.max()
            if b["delay"].max() > delay_max:
                delay_max = b["delay"].max()
            ahist.hist(same_time_delivery.values)
            ahist.set_xlabel("number of pkt at same time")
            ahist.set_ylabel("count")
            ahist.set_title(app)
            ahist.xaxis.set_major_locator(MultipleLocator(1))
            self.plot_ecdf(same_time_delivery, ax=acdf)
            acdf.set_xlabel("number of pkt at same time")
            acdf.set_title(app)
            acdf.xaxis.set_major_locator(MultipleLocator(1))

            _desc = combine_stats(
                ["pkt_count", "delay"], same_time_delivery, b["delay"]
            )
            _desc = _desc.to_string(float_format=lambda x: f"{x:.3e}")

            adesc.text(
                -0.20,
                0.5,
                f"Describe:\n num pkt at same time and delay\n ({app}):\n{_desc}",
                transform=adesc.transAxes,
                horizontalalignment="left",
                verticalalignment="center",
                fontfamily="monospace",
                fontsize="large",
            )
            adesc.set_axis_off()

            self.plot_ecdf(b, column="delay", ax=adelay)
            adelay.set_xlabel("delay in seconds")
            adelay.set_title(app)
            adelay.set_title(f"Delay {app} CDF")
        if len(apps) > 1:
            for axes in ax:
                axes[0].set_xlim(0, count_max)
                axes[1].set_xlim(0, count_max)
                axes[3].set_xlim(-0.5, delay_max)
                axes[3].xaxis.set_major_locator(MultipleLocator(10))

        fig.suptitle(figuer_title)
        saver(fig, "same_time_delivery.png")


PlotAppMisc = PlotAppMisc_()
