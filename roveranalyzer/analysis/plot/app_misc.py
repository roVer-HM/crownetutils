import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import roveranalyzer.simulators.opp.scave as Scave
import roveranalyzer.utils.plot as p
from roveranalyzer.analysis.common import Simulation
from roveranalyzer.analysis.omnetpp import OppAnalysis
from roveranalyzer.utils.logging import logger, timing
from roveranalyzer.utils.plot import FigureSaver, _PlotUtil


class _PlotAppTxInterval(_PlotUtil):
    @timing
    def plot_txinterval_all(
        self,
        data_root: str,
        sql: Scave.CrownetSql,
        app: str = "Beacon",
        saver: FigureSaver = FigureSaver.FIG,
    ):
        data = OppAnalysis.get_txAppInterval(sql, app_type=app)
        data = data.droplevel(["hostId", "host"]).sort_index()
        fig, _ = self.df_to_table(
            data.describe().applymap("{:1.4f}".format).reset_index(),
            title=f"Descriptive statistics for application {app}",
        )
        saver(fig, os.path.join(data_root, f"tx_AppIntervall_stat.pdf"))

        fig, _ = self.plot_ts_txinterval(data, app_name=app, time_bucket_length=1.0)
        saver(fig, os.path.join(data_root, f"txAppInterval_ts.pdf"))

        fig, _ = self.plot_hist_txinterval(data)
        saver(fig, os.path.join(data_root, f"tx_AppInterval_hist_.pdf"))

        fig, _ = self.plot_ecdf_txinterval(data)
        saver(fig, os.path.join(data_root, f"tx_AppInterval_ecdf.pdf"))

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
        return fig, ax


PlotAppTxInterval = _PlotAppTxInterval()


class _PlotAppMisc(_PlotUtil):
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

    def plot_system_level_tx_rate_based_on_application_layer_data(
        self, sim: Simulation, *, saver: FigureSaver = FigureSaver.FIG
    ):
        """
        How many packets, at which size, and at what rate are produced by each application
        and in total.
        To get this data collect all 'packetSent:vector(packetBytes)' from all nodes and all applications.
        (1) Show a time series of packet size for each application
        (2) Show the throughput over time for each application and in total.
        (3) What are the number of neighbors/members used in the tx interval algorithm?
        """
        sql = sim.sql
        tx_pkt_beacon = sql.vec_data(
            sql.m_beacon(), "packetSent:vector(packetBytes)"
        ).drop(columns=["vectorId"])
        tx_pkt_beacon["app"] = "Beacon"
        tx_pkt_map = sql.vec_data(sql.m_map(), "packetSent:vector(packetBytes)").drop(
            columns=["vectorId"]
        )
        tx_pkt_map["app"] = "Map"
        # (1) packet size t
        fig, ax = self.check_ax()
        ax.scatter(
            "time", "value", data=tx_pkt_beacon, marker="x", color="red", label="Beacon"
        )
        ax.scatter(
            "time", "value", data=tx_pkt_map, marker="+", color="blue", label="Map"
        )
        ax.set_ylabel("Packet size in bytes")
        ax.set_xlabel("Time in seconds")
        ax.set_title(
            "Packet size over time for all agents and all applications (Beacon and Map)"
        )
        ax.legend()
        saver(fig, sim.path("packet_size_ts.pdf"))
        # (2) throughput
        tx_pkt = pd.concat(
            [tx_pkt_beacon, tx_pkt_map], axis=0, ignore_index=True
        ).set_index(["time"])
        t_delta = 1.0
        bins = pd.interval_range(
            start=0.0, end=tx_pkt.index.max(), freq=t_delta, closed="left"
        )
        tx_rate_sum = (
            tx_pkt.loc[:, ["value"]].groupby(pd.cut(tx_pkt.index, bins=bins)).sum()
            / t_delta
            / 1000
        )
        tx_rate_app = (
            (
                tx_pkt.groupby([pd.cut(tx_pkt.index, bins=bins), "app"]).sum()
                / t_delta
                / 1000
            )
            .unstack("app")
            .droplevel(0, axis=1)
        )
        tx_rate = pd.concat(
            [tx_rate_sum, tx_rate_app], axis=1, ignore_index=False
        ).rename(columns={"value": "Total"})
        tx_rate.index = bins.left
        tx_rate.index.name = "time"
        fig2, ax2 = self.check_ax()
        marker = [".", "x", "+"]
        for i, c in enumerate(tx_rate.columns):
            ax2.scatter(tx_rate.index, tx_rate[c], marker=marker[i], label=c)

        ax2.legend()
        ax2.set_title("Data rate based on sent packets in all from all nodes")
        ax2.set_ylabel("Data rate in kBps")
        ax2.set_xlabel("Time in seconds")
        saver(fig2, sim.path("System_tx_data_rate.pdf"))

    def plot_number_of_agents(
        self, sim: Simulation, *, saver: FigureSaver = FigureSaver.FIG
    ):
        sql = sim.sql
        dmap = sim.builder.build_dcdMap()
        fig, ax = self.check_ax()
        # (1) Cumulative number of nodes in the simulation over time
        id_set = set()
        id_count = []
        df_id = dmap.position_df.reset_index("node_id")
        times = df_id.sort_index().index.unique()

        for t in times:
            try:
                id_set = id_set.union(df_id.loc[t, "node_id"])
            except TypeError as e:
                id_set = id_set.union([df_id.loc[t, "node_id"]])

            id_count.append(len(id_set))
        ax.scatter(times, id_count, s=5, marker="+", color="k", label="Cum. node count")

        tx_member_count = sql.vec_data(
            sql.m_map(), "rcvdSrcCount:vector", drop="vectorId"
        )
        nt_count = sql.vec_data(sql.m_table(), "tableSize:vector", drop="vectorId")
        node_count = (
            dmap.glb_map.groupby("simtime")
            .count()
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
            data=tx_member_count,
            label="Tx interval alg. member count",
            s=2,
            marker="x",
            color="b",
            alpha=0.6,
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
        saver(fig, sim.path("Node_count_ts.png"))

        fig, ax = self.check_ax()
        tx_member_count_d = self.ts_mean(tx_member_count.set_index("time")).dropna()
        nt_count_d = self.ts_mean(nt_count.set_index("time")).dropna()

        ax.plot(
            "time",
            "mean",
            data=tx_member_count_d,
            label="mean tx interval alg. member count",
            marker="x",
            color="b",
        )
        ax.fill_between(
            "time",
            "25%",
            "75%",
            data=tx_member_count_d,
            color="b",
            alpha=0.35,
            interpolate=True,
            label="Q1;Q3 tx interval alg. member count",
        )

        ax.plot(
            "time",
            "mean",
            data=nt_count_d,
            label="Neighborhood table node count",
            marker="x",
            color="r",
        )
        ax.fill_between(
            "time",
            "25%",
            "75%",
            data=nt_count_d,
            color="r",
            alpha=0.35,
            interpolate=True,
            label="Q1;Q3 neighborhood table node count",
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
        saver(fig, sim.path("Mean_node_count_ts.png"))

    def plot_application_jitter(
        self, sim: Simulation, *, saver: FigureSaver = FigureSaver.FIG
    ):
        def f(df, host_id, rx_id):
            _m = (df["hostId"] == host_id) & (df["tx_host_id"] == rx_id)
            return df[_m]

        map_delay = OppAnalysis.get_received_packet_delay(
            sql=sim.sql,
            drop_self_message=True,
            module_name=sim.sql.m_map(),
        ).reset_index()
        map_jitter = OppAnalysis.get_received_packet_jitter(
            sql=sim.sql, drop_self_message=True, module_name=sim.sql.m_map()
        ).reset_index()
        fig, ax = self.check_ax()
        ax.scatter(
            "time",
            "delay",
            data=f(map_delay, 264, 1008),
            marker="x",
            alpha=0.5,
            label="delay",
        )
        ax.scatter(
            "time",
            "jitter",
            data=f(map_jitter, 264, 1008),
            marker="+",
            alpha=0.5,
            label="jitter",
        )
        ax.legend()
        print("hi")


PlotAppMisc = _PlotAppMisc()
