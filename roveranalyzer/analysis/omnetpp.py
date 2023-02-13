from __future__ import annotations

import itertools
import os
from functools import partial
from os.path import join
from typing import IO, Any, Dict, List, Protocol, TextIO, Tuple
from xmlrpc.client import ProtocolError

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import IndexSlice as _i

from roveranalyzer.utils import dataframe

sns.set(font_scale=1.0, rc={"text.usetex": True})
from matplotlib.backends.backend_pdf import PdfPages
from omnetinireader.config_parser import ObjectValue
from scipy.stats import kstest, mannwhitneyu

import roveranalyzer.simulators.crownet.dcd as Dcd
import roveranalyzer.simulators.opp.scave as Scave
import roveranalyzer.utils.plot as _Plot
from roveranalyzer.analysis.base import AnalysisBase
from roveranalyzer.analysis.common import (
    RunMap,
    SimGroupFilter,
    Simulation,
    SimulationGroup,
)
from roveranalyzer.simulators.crownet.dcd.dcd_map import percentile
from roveranalyzer.simulators.opp.provider.hdf.IHdfProvider import BaseHdfProvider
from roveranalyzer.simulators.opp.scave import CrownetSql, SqlEmptyResult, SqlOp
from roveranalyzer.utils.dataframe import (
    FrameConsumer,
    FrameConsumerList,
    append_index,
    format_frame,
    siunitx,
)
from roveranalyzer.utils.general import DataSource
from roveranalyzer.utils.logging import logger, timing
from roveranalyzer.utils.parallel import run_args_map, run_kwargs_map

PlotUtil = _Plot.PlotUtil


def make_run_series(
    df: pd.DataFrame, run_id: int, lvl_name: str = "rep", stack_index=None
) -> pd.Series:
    df.columns = pd.MultiIndex.from_tuples(
        [(run_id, c) for c in df.columns], names=[lvl_name, "data"]
    )
    if stack_index is None:
        df = df.stack([lvl_name, "data"])
    else:
        df = df.stack(stack_index)
    return df


class _hdf_Extractor(AnalysisBase):
    def __init__(self) -> None:
        pass

    @classmethod
    def extract_trajectories(cls, hdf_file: str, sql: CrownetSql):
        _hdf = BaseHdfProvider(hdf_file, "trajectories")
        pos = sql.host_position()
        _hdf.write_frame(group="trajectories", frame=pos)

    @classmethod
    def extract_packet_loss(
        cls, hdf_file: str, group_suffix: str, sql: CrownetSql, app: SqlOp
    ):
        try:
            pkt_loss_g = f"pkt_loss_{group_suffix}"
            raw_g = f"pkt_loss_raw_{group_suffix}"
            hdf_store = BaseHdfProvider(hdf_file)
            if (
                hdf_store.hdf_file_exists
                and hdf_store.contains_group(pkt_loss_g)
                and hdf_store.contains_group(raw_g)
            ):
                logger.info(
                    f"hdf store and groups: '{pkt_loss_g},{raw_g}' already exist skip."
                )
            else:
                pkt_loss, raw = OppAnalysis.get_received_packet_loss(sql, app)
                hdf_store.write_frame(pkt_loss_g, pkt_loss)
                hdf_store.write_frame(raw_g, raw)
        except SqlEmptyResult as e:
            logger.error("No packets found")
            logger.error(e)


class _OppAnalysis(AnalysisBase):
    def __init__(self) -> None:
        pass

    def get_packet_age(
        self,
        sql: Scave.CrownetSql,
        app_path: str,
        host_name: SqlOp | str | None = None,
    ) -> pd.DataFrame:
        """
        Deprecated use get_received_packet_delay
        Get packet ages for any stationary and moving node x
        Packet age: (time x received packet i) - (time packet i created)
                |hostId (x)    |  time_recv |  packet_age  |
            0   |  12          |  1.2  |  0.3  |
            1   |  12          |  1.3  |  0.4  |
            ... |  55          |  8.3  |  4.6  |
        """

        if not app_path.startswith("."):
            app_path = f".{app_path}"
        module_name = sql.m_append_suffix(app_path, modules=host_name)
        df, _ = self.get_received_packet_delay(
            sql, module_name=module_name, describe=False
        )
        df = (
            df.reset_index()
            .rename(columns={"time": "time_recv", "rcvdPktLifetime": "packet_age"})
            .drop(columns=["srcHostId"])
        )
        return df

    def get_packet_source_distribution(
        self,
        sql: Scave.CrownetSql,
        app_path: str,
        host_name: SqlOp | str | None = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        Create square matrix of [hostId X hostId] showing the source hostId of received packets for the given application path.
        Example:
        hostId/hostId |  1  |  2  |  3  |
            1         |  0  |  4  |  8  |
            2         |  1  |  0  |  1  |
            3         |  6  |  6  |  0  |
        host_1 received 4 packets from host_2
        host_1 received 8 packets from host_3
        host_2 received 1 packet  from host_1
        host_2 received 1 packet  from host_3
        host_3 received 6 packets from host_1
        host_3 received 6 packets from host_2
        """
        id_map = sql.host_ids(host_name)
        if not app_path.startswith("."):
            app_path = f".{app_path}"

        df = None
        for _id, host in id_map.items():
            _df = sql.vec_merge_on(
                f"{host}{app_path}",
                sql.OR(["rcvdPkSeqNo:vector", "rcvdPkHostId:vector"]),
            )
            _df["hostId"] = _id
            if df is None:
                df = _df
            else:
                df = pd.concat([df, _df], axis=0)

        df = df.loc[:, ["hostId", "rcvdPkHostId:vector"]]
        df["rcvdPkHostId:vector"] = pd.to_numeric(
            df["rcvdPkHostId:vector"], downcast="integer"
        )
        df["val"] = 1
        df_rcv = df.pivot_table(
            index="hostId",
            columns=["rcvdPkHostId:vector"],
            aggfunc="count",
            values=["val"],
            fill_value=0,
        )
        df_rcv.columns = df_rcv.columns.droplevel()

        # normalize number of received packets (axis = 1 / over columns)
        if normalize:
            _sum = df_rcv.sum(axis=1)
            _sum = np.repeat(_sum.to_numpy(), _sum.shape[0]).reshape(
                (_sum.shape[0], -1)
            )
            df_rcv /= _sum
        return df_rcv

    @PlotUtil.with_axis
    def plot_packet_source_distribution(
        self,
        data: pd.DataFrame,
        hatch_patterns: List[str] = PlotUtil.hatch_patterns,
        ax: plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot packet source distribution

        Args:
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

    def get_neighborhood_table_size(
        self,
        sql: Scave.CrownetSql,
        module_name: Scave.SqlOp | str | None = None,
        **kwargs,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Extract neighborhood table size vectors from given module_name.

        Args:
            sql (Scave.CrownetSql): DB handler see Scave.CrownetSql for more information
            module_name (Scave.SqlOp, optional): Module selector for neighborhood table. See CrownetSql.m_XXX methods for examples

        Returns:
            tuple[pd.DataFrame, np.ndarray]: [description]
        """
        module_name = sql.m_table() if module_name is None else module_name

        tbl = sql.vector_ids_to_host(
            module_name=module_name,
            vector_name="tableSize:vector",
            name_columns=["host", "hostId"],
            pull_data=True,
        ).drop(columns=["vectorId"])

        tbl_idx = tbl["hostId"].unique()
        tbl_idx.sort()
        return tbl, tbl_idx

    @PlotUtil.with_axis
    def plot_neighborhood_table_size_over_time(
        self, tbl: pd.DataFrame, tbl_idx: np.ndarray, ax: plt.Axes = None
    ) -> plt.Axes:
        """Plot neighborhood table size for each node over time.
        x-axis: time
        y-axis: number of entries in neighborhood table

        Args:
            tbl (pd.DataFrame): Data see get_neighborhood_table_size_over_time()
            tbl_idx (np.ndarray): see get_neighborhood_table_size_over_time()
            ax (plt.Axes, optional): Axes to use. If missing a new axes will be injected by
                                     PlotUtil.with_axis decorator.

        Returns:
            plt.Axes:
        """
        _c = PlotUtil.color_lines(line_type=None)
        for i in tbl_idx:
            _df = tbl.loc[tbl["host"] == i]
            ax.plot(_df["time"], _df["value"], next(_c), label=i)

        ax.set_ylabel("Neighboor count")
        ax.set_xlabel("time [s]")
        ax.set_title("Size of neighborhood table over time for each node")
        return ax.get_figure(), ax

    @timing
    def get_cumulative_received_packet_delay(
        self,
        sql: Scave.CrownetSql,
        module_name: Scave.SqlOp | str,
        delay_resolution: float = 1.0,
        index: List[str] | None = ("time",),
    ) -> pd.DataFrame:
        df = sql.vec_data(
            module_name=module_name,
            vector_name="rcvdPkLifetime:vector",
            value_name="delay",
            index=index if index is None else list(index),
            index_sort=True,
        )
        df["delay"] *= delay_resolution
        return df

    @timing
    def get_avgServedBlocksUl(
        self,
        sql: Scave.CrownetSql,
        enb_index: int = -1,
        index: List[str] | None = ("time",),
    ) -> pd.DataFrame:
        df = sql.vec_data(
            module_name=sql.m_enb(enb_index, ".cellularNic.mac"),
            vector_name="avgServedBlocksUl:vector",
            index=index if index is None else list(index),
            index_sort=True,
        )
        return df

    @timing
    def get_txAppInterval(
        self,
        sql: Scave.CrownetSql,
        app_type: str = "beacon",
        index: List[str] | None = ("time",),
    ) -> pd.DataFrame:
        if app_type.lower() == "beacon":
            m = sql.m_beacon(app_mod="scheduler")
        else:
            m = sql.m_map(app_mod="scheduler")

        df1 = sql.vector_ids_to_host(
            module_name=m,
            vector_name="txInterval:vector",
            name_columns=["host", "hostId"],
            pull_data=True,
            value_name="txInterval",
            index=["time", "host", "hostId"],
        ).drop(columns=["vectorId"])
        df2 = sql.vector_ids_to_host(
            module_name=m,
            vector_name="txDetInterval:vector",
            name_columns=["host", "hostId"],
            pull_data=True,
            value_name="txDetInterval",
            index=["time", "host", "hostId"],
        ).drop(columns=["vectorId"])

        df = pd.concat([df1, df2], axis=1, ignore_index=False)
        return df

    def plot_hist_enb_served_rb(self, data: pd.DataFrame, bins=25, enb=0):
        ax: plt.Axes
        fig, ax = _Plot.check_ax()
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
        fig, ax = _Plot.check_ax()
        ax.plot(_x, _y)
        ax.set_title("ECDF of resource block utilization of eNB {enb}")
        ax.set_xlabel("Resource Blocks (RB's)")
        ax.set_ylabel("ECDF")
        ax.set_xlim(-1, bins + 1)
        ax.set_xticks(np.arange(0, bins + 1, 1))
        return fig, ax

    @timing
    def plot_txinterval_all(
        self,
        data_root: str,
        sql: Scave.CrownetSql,
        app: str = "Beacon",
        saver: _Plot.FigureSaver = _Plot.FigureSaver.FIG,
    ):
        data = self.get_txAppInterval(sql, app_type=app)
        data = data.droplevel(["hostId", "host"]).sort_index()
        fig, ax = plt.subplots()
        _Plot.PlotUtil.df_to_table(
            data.describe().applymap("{:1.4f}".format).reset_index(), ax
        )
        ax.set_title(f"Descriptive statistics for application {app}")
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
        fig, ax = _Plot.check_ax()
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
        fig, ax = _Plot.check_ax()
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
        fig, ax = _Plot.check_ax()
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

    def plot_ts_enb_served_rb(self, data: pd.DataFrame, time_bucket_length=1.0):
        interval = pd.interval_range(
            start=0.0, end=np.ceil(data.index.max()), freq=time_bucket_length
        )
        data = data.groupby(pd.cut(data.index, interval)).mean()
        data.index = interval.left
        data.index.name = "time"
        data = data.reset_index()
        fig, ax = _Plot.check_ax()
        ax.plot("time", "value", data=data)
        ax.set_title("Average Resource Block (RB) usage over time. (time bin size 1s)")
        ax.set_xlabel("time in [s]")
        ax.set_ylabel("Resource blocks")
        # ax.set_ylim(0, bins+1)
        # ax.set_yticks(np.arange(0, bins+1, 1))
        return fig, ax

    @timing
    def get_received_packet_delay(
        self,
        sql: Scave.CrownetSql,
        module_name: Scave.SqlOp | str,
        delay_resolution: float = 1.0,
        describe: bool = True,
        value_name: str = "rcvdPktLifetime",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Packet delay data based on single applications in '<Network>.<Module>[*].app[*].app'

        Args:
            sql (Scave.CrownetSql): DB handler see Scave.CrownetSql for more information
            module_name (Scave.SqlOp): Module selector for application(s). See CrownetSql.m_XXX methods for examples
            delay_resolution (float, optional): Delay resolution mutliplier. Defaults to 1.0 (seconds).
            describe (bool, optional): [description]. If true second data frame contains descriptive statistics based on hostId/srcHostId. Defaults to True.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: DataFrame of the form (index)[columns]:
            (1) raw data (hostId, srcHostId, time)[delay]
            (2) empty  or (hostId, srcHostId)[count, mean, std, min, 25%, 50%, 75%, max] if describe=True
        """
        vec_names = {
            "rcvdPkLifetime:vector": {
                "name": value_name,
                "dtype": float,
            },
            "rcvdPkHostId:vector": {"name": "srcHostId", "dtype": np.int32},
        }
        vec_data = sql.vec_data_pivot(
            module_name,
            vec_names,
            append_index=["srcHostId"],
        )
        vec_data = vec_data.sort_index()
        vec_data[value_name] *= delay_resolution

        if describe:
            return vec_data, vec_data.groupby(level=["hostId", "srcHostId"]).describe()
        else:
            return vec_data, pd.DataFrame()

    @timing
    def get_measured_sinr_d2d(
        self,
        sql: Scave.CrownetSql,
        module_name: Scave.CrownetSql | str | None = None,
        withMacId: bool = False,
    ) -> pd.DataFrame:
        """SINR measurements for D2D communication at the receveing node.

        Args:
            sql (Scave.CrownetSql): DB handler see Scave.CrownetSql for more information
            module_name (Scave.CrownetSql): Module selector. If None use all modules (misc, pNode, vNode) and all.
                                            For selector path see CrownetSql.m_XXXX methods for examples as well as the moduleName
                                            column in the simulation vector files.
            withMacId (bool, optional): In addition to hostId (OMNeT based node id) return Sim5G identifier (currently only for the source node). Defaults to False.

        Returns:
            pd.DataFrame: SINR dataframe of the form [index](columns): [hostId, srcHostId, srcMacId, time](rcvdSinrD2D)

            e.g.
                                               rcvdSinrD2D
            hostId srcHostId srcMacId time
            20     23        1028     0.213    18.003613
                                      0.446    18.164892
                                      0.550    18.164892
            "Node 20 received data from node 23 (or macNode 1028) at time 0.213 a signal with the SINR of 18.003613
        """
        module_name = sql.m_channel() if module_name is None else module_name

        vec_names = {
            "rcvdSinrD2D:vector": {"name": "rcvdSinrD2D", "dtype": np.float32},
            "rcvdSinrD2DSrcOppId:vector": {"name": "srcHostId", "dtype": np.int32},
        }
        if withMacId:
            vec_names["rcvdSinrD2DSrcMacId:vector"] = {
                "name": "srcMacId",
                "dtype": np.int32,
            }
            append_idx = ["srcHostId", "srcMacId"]
        else:
            append_idx = ["srcHostId"]
        df = sql.vec_data_pivot(module_name, vec_names, append_index=append_idx)
        return df

    @timing
    def get_measured_sinr_ul_dl(
        self,
        sql: Scave.CrownetSql,
        sinr_module: Scave.CrownetSql | str | None = None,
        enb_module: Scave.CrownetSql | str | None = None,
    ) -> pd.DataFrame:
        """Return SINR for UL and DL during feedback computation at the serving eNB for each host

        Args:
            sql (Scave.CrownetSql): DB handler see Scave.CrownetSql for more information
            sinr_module (Scave.CrownetSql, optional): Module selector to access SINR vectors (part of the channelmodel node). If None use
                                                      all module vectors (misc, pNode, vNode). Defaults to None.
            enb_module (Scave.CrownetSql, optional): Module selector for serving eNB for each host over time.
                                                     If None use all module vectors (misc, pNode, vNode). Selection must match the selection of sinr_module
                                                     Defaults to None.

        Returns:
            pd.DataFrame: DataFrame of the form [index](columns): [hostId, time](sinrDl, sinrUl, eNB)
        """

        sinr_module = sql.m_channel() if sinr_module is None else sinr_module
        enb_module = sql.m_phy() if enb_module is None else enb_module
        sinr_names = {
            "measuredSinrUl:vector": {
                "name": "mSinrUl",
                "dtype": float,
            },
            "measuredSinrDl:vector": {"name": "mSinrDl", "dtype": float},
        }
        sinr = sql.vec_data_pivot(sinr_module, sinr_names)
        sinr = sinr.sort_index()

        enbs = sql.vector_ids_to_host(
            sql.m_phy(), "servingCell:vector", name_columns=["hostId"], pull_data=True
        )
        enbs = enbs.drop(columns=["vectorId"]).rename(columns={"value": "eNB"})
        enbs = enbs.set_index(["hostId", "time"]).sort_index()

        df = pd.merge(sinr, enbs, on=["hostId", "time"], how="outer").sort_index()
        for _, _df in df.groupby(level=["hostId"]):
            df.loc[_df.index, "eNB"] = _df["eNB"].ffill()
        df = df.dropna()
        return df

    @timing
    def get_received_packet_loss(
        self, sql: Scave.CrownetSql, module_name: Scave.SqlOp | str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Packet loss data based on single applications in '<Network>.<Module>[*].app[*].app'

        Args:
            sql (Scave.CrownetSql): DB handler see Scave.CrownetSql for more information
            module_name (Scave.SqlOp, optional): Modules for which the packet loss is calculated

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:  DataFrames of the form (index)[columns]:
            (1) aggregated  packet loss of the form (hostId, srcHostId) [numPackets, packet_lost, packet_loss_ratio]
            (2) raw data (hostId, srcHostId, time)[seqNo]
        """
        logger.info("load packet loss data from *.vec")

        # statistic was renamed. Check old version fist.
        seqNo_vec = ["rcvdPkSeqNo:vector", "rcvdPktPerSrcSeqNo:vector"]
        seqNo_vec = sql.find_vector_name(module_name, seqNo_vec)
        vec_names = {
            seqNo_vec: {
                "name": "seqNo",
                "dtype": np.int32,
            },
            "rcvdPkHostId:vector": {"name": "srcHostId", "dtype": np.int32},
        }

        vec_data = sql.vec_data_pivot(
            module_name, vec_names, append_index=["srcHostId"]
        )
        logger.info("calculate packet los per host and packet source")
        grouped = vec_data.groupby(level=["hostId", "srcHostId"])
        vec_data["lost"] = 0
        for _, group in grouped:
            vec_data.loc[group.index, "lost"] = group["seqNo"].diff() - 1

        logger.info("calculate packet loss ratio per host and source")
        lost_df = (
            vec_data.groupby(level=["hostId", "srcHostId"])["seqNo"]
            .apply(lambda x: x.max() - x.min())
            .to_frame()
        )
        lost_df = lost_df.rename(columns={"seqNo": "numPackets"})
        lost_df["packet_lost"] = vec_data.groupby(level=["hostId", "srcHostId"])[
            "lost"
        ].sum()
        lost_df["packet_loss_ratio"] = lost_df["packet_lost"] / lost_df["numPackets"]
        return lost_df, vec_data

    def get_received_packet_bytes(
        self,
        sql: Scave.CrownetSql,
        module_name: Scave.SqlOp | str,
    ) -> pd.DataFrame:

        df = sql.vec_data(
            module_name=module_name,
            vector_name="packetReceived:vector(packetBytes)",
        )
        return df

    @timing
    def create_common_plots_density(
        self,
        data_root: str,
        builder: Dcd.DcdHdfBuilder,
        sql: Scave.CrownetSql,
        selection: str = "yml",
    ):
        dmap = builder.build_dcdMap(selection=selection)
        with PdfPages(join(data_root, "common_output.pdf")) as pdf:
            dmap.plot_map_count_diff(savefig=pdf)

            tmin, tmax = builder.count_p.get_time_interval()
            time = (tmax - tmin) / 4
            intervals = [slice(time * i, time * i + time) for i in range(4)]
            for _slice in intervals:
                dmap.plot_error_histogram(time_slice=_slice, savefig=pdf)

    @timing
    def plot_served_blocks_ul_all(
        self,
        data_root: str,
        builder: Dcd.DcdHdfBuilder,
        sql: Scave.CrownetSql,
        saver: _Plot.FigureSaver = _Plot.FigureSaver.FIG,
    ):
        num_enb = int(sql.get_run_config("*.numEnb"))
        bins = int(sql.get_run_config("**.numBands"))
        for n in range(num_enb):
            data = self.get_avgServedBlocksUl(sql, enb_index=n)
            fig, _ = self.plot_ts_enb_served_rb(data, time_bucket_length=1.0)
            saver(fig, os.path.join(data_root, f"rb_utilization_ts_{n}.pdf"))
            fig, _ = self.plot_hist_enb_served_rb(data, bins, n)
            saver(fig, os.path.join(data_root, f"rb_utilization_hist_{n}.pdf"))
            fig, _ = self.plot_ecdf_enb_served_rb(data, bins, n)
            saver(fig, os.path.join(data_root, f"rb_utilization_ecdf_{n}.pdf"))
            fig, ax = plt.subplots()
            _Plot.PlotUtil.df_to_table(
                data.describe().applymap("{:1.4f}".format).reset_index(), ax
            )
            saver(fig, os.path.join(data_root, f"rb_stat_{n}.pdf"))

    @timing
    def append_count_diff_to_hdf(
        self,
        sim: Simulation,
    ):
        group_name = "count_diff"
        _hdf = sim.get_base_provider(group_name, path=sim.builder.count_p._hdf_path)
        if not _hdf.contains_group(group_name):
            print(f"group '{group_name}' not found. Append to {_hdf._hdf_path}")
            df = sim.get_dcdMap().count_diff()
            _hdf.write_frame(group=group_name, frame=df)
        else:
            print(f"group '{group_name}' found. Nothing to do for {_hdf._hdf_path}")

    @timing
    def append_err_measures_to_hdf(
        self,
        sim: Simulation,
    ):
        map = sim.get_dcdMap()
        if sim.sql.is_count_map():
            group = "map_measure"
            _hdf = sim.get_base_provider(group, path=sim.builder.count_p.hdf_path)
            if _hdf.contains_group(group):
                print(f"group 'map_measure' found. Nothing to do for {_hdf._hdf_path}")
            else:
                map_measure = map.map_count_measure(load_cached_version=False)
                _hdf.write_frame(group=group, frame=map_measure)

        if sim.sql.is_entropy_map():
            # use cell_value_measure method
            group = "cell_measures"
            _hdf = sim.get_base_provider(group, path=sim.builder.count_p.hdf_path)
            if _hdf.contains_group(group):
                print(f"group '{group}' found. Nothing to do for {_hdf._hdf_path}")
            else:
                cell_measure = map.cell_value_measure(load_cached_version=False)
                _hdf.write_frame(group=group, frame=cell_measure)
        else:
            # use cell_count_measure method
            group = "cell_measures"
            _hdf = sim.get_base_provider(group, path=sim.builder.count_p.hdf_path)
            if _hdf.contains_group(group):
                print(f"group '{group}' found. Nothing to do for {_hdf._hdf_path}")
            else:
                cell_measure = map.cell_count_measure(load_cached_version=False)
                _hdf.write_frame(group=group, frame=cell_measure)

    @timing
    def get_data_001(self, sim: Simulation):
        """
        collect data for given simulation
        """
        print(f"get data for {sim.data_root} ...")
        sel = sim.builder.map_p.get_attribute("used_selection")
        if sel is None:
            raise ValueError("selection not set!")
        dmap = sim.builder.build_dcdMap(selection=list(sel)[0])

        print("diff")
        count_diff = DataSource.provide_result("count_diff", sim, dmap.count_diff())

        print("box")
        err_box = DataSource.provide_result(
            "err_box", sim, dmap.err_box_over_time(bin_width=10)
        )

        print("hist")
        err_hist = DataSource.provide_result(
            "err_hist",
            sim,
            dmap.error_hist(),
        )

        print(f"done for {sim.data_root}")

        return count_diff, err_box, err_hist
        # return count_diff, err_hist

    @timing
    def create_plot_err_box_over_time(self, sim: Simulation, title: str, ax: plt.Axes):

        s = _Plot.Style()
        s.font_dict = {
            "title": {"fontsize": 14},
            "xlabel": {"fontsize": 10},
            "ylabel": {"fontsize": 10},
            "legend": {"size": 14},
            "tick_size": 10,
        }
        s.create_legend = False

        dmap = sim.get_dcdMap()
        dmap.style = s
        _, ax = dmap.plot_err_box_over_time(ax=ax, xtick_sep=10)
        ax.set_title(title)
        return ax

    def err_hist_plot(self, s: _Plot.Style, data: List[DataSource]):
        def title(sim: Simulation):
            cfg = sim.run_context.oppini
            map: ObjectValue = cfg["*.pNode[*].app[1].app.mapCfg"]
            run_name = sim.run_context.args.get_value("--run-name")
            if all(i in map for i in ["alpha", "stepDist", "zeroStep"]):
                return f"{run_name[0:-7]}\nalpha:{map['alpha']} distance threshold: {map['stepDist']} use zero: {map['zeroStep']}"
            else:
                return f"{run_name[0:-7]}\n Youngest measurement first"

        fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(16, 9))

        axes = [a for aa in ax for a in aa]
        for a in axes[len(data) :]:
            a.remove()

        for idx, run in enumerate(data):
            sim: Simulation = run.source
            dmap = sim.get_dcdMap()
            dmap.style = s
            # provide data from run cache
            _, a = dmap.plot_error_histogram(ax=axes[idx], data_source=run)
            a.set_title(title(sim))

        PlotUtil.equalize_axis(fig.get_axes(), "y")
        PlotUtil.equalize_axis(fig.get_axes(), "x")
        fig.suptitle("Count Error Histogram")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))

        return fig

    def box_plot(self, data: pd.DataFrame, bin_width, bin_key):

        if bin_key in data.columns:
            data = data.set_index(bin_key, verify_integrity=False)

        bins = int(np.floor(data.index.max() / bin_width))
        _cut = pd.cut(data.index, bins)
        return data.groupby(_cut), _cut

    def err_box_plot(self, s: _Plot.Style, data: List[DataSource]):
        def title(sim: Simulation):
            cfg = sim.run_context.oppini
            map: ObjectValue = cfg["*.pNode[*].app[1].app.mapCfg"]
            run_name = sim.run_context.args.get_value("--run-name")
            if all(i in map for i in ["alpha", "stepDist", "zeroStep"]):
                return f"{run_name[0:-7]}\nalpha:{map['alpha']} distance threshold: {map['stepDist']} use zero: {map['zeroStep']}"
            else:
                return f"{run_name[0:-7]}\n Youngest measurement first"

        fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(16, 9))

        axes = [a for aa in ax for a in aa]
        for a in axes[len(data) :]:
            a.remove()

        for idx, run in enumerate(data):
            sim: Simulation = run.source
            dmap = sim.get_dcdMap()
            dmap.style = s
            # provide data from run cache
            _, a = dmap.plot_err_box_over_time(
                ax=axes[idx], xtick_sep=10, data_source=run
            )
            a.set_title(title(sim))

        PlotUtil.equalize_axis(fig.get_axes(), "y")
        PlotUtil.equalize_axis(fig.get_axes(), "x")
        fig.suptitle("Count Error over time")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))

        return fig

    def diff_plot(self, s: _Plot.Style, data: List[DataSource]):
        def title(sim: Simulation):
            cfg = sim.run_context.oppini
            map: ObjectValue = cfg["*.pNode[*].app[1].app.mapCfg"]
            run_name = sim.run_context.args.get_value("--run-name")
            if all(i in map for i in ["alpha", "stepDist", "zeroStep"]):
                return f"{run_name[0:-7]}\nalpha:{map['alpha']} distance threshold: {map['stepDist']} use zero: {map['zeroStep']}"
            else:
                return f"{run_name[0:-7]}\n Youngest measurement first"

        fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(16, 9))

        axes = [a for aa in ax for a in aa]
        for a in axes[len(data) :]:
            a.remove()

        for idx, run in enumerate(data):
            sim: Simulation = run.source
            dmap = sim.get_dcdMap()
            dmap.style = s
            # provide data from run cache
            _, a = dmap.plot_map_count_diff(ax=axes[idx], data_source=run)
            a.set_title(title(sim))

        # fix legends
        x = axes[0].legend()
        axes[0].get_legend().remove()
        PlotUtil.equalize_axis(fig.get_axes(), "y")
        PlotUtil.equalize_axis(fig.get_axes(), "x")
        fig.suptitle("Comparing Map count with ground truth over time")
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
        fig.legend(
            x.legendHandles,
            [i._text for i in x.texts],
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
        )

        return fig

    def merge_position(
        self,
        sim_group: SimulationGroup,
        time_slice=slice(0.0),
        frame_consumer: FrameConsumer = FrameConsumer.EMPTY,
    ) -> pd.DataFrame:
        df = []
        for run_id, sim in sim_group.simulation_iter():
            _pos = sim.sql.host_position(
                module_name="World.misc[%]", apply_offset=False, time_slice=time_slice
            )
            _pos["run_id"] = run_id
            _pos["drop_nodes"] = _pos["vecIdx"] >= (_pos["vecIdx"].max() + 1) / 2
            df.append(_pos)

        df: pd.DataFrame = pd.concat(df, axis=0, ignore_index=True)
        df = df.set_index(["run_id", "hostId", "vecIdx", "drop_nodes"]).sort_index()
        df = frame_consumer(df)
        return df

    def sg_collect_maps(
        self,
        sim_group: SimulationGroup,
        data: List[str] | None = ("map_glb_count", "map_mean_count"),
        drop_nan: bool = True,
        frame_consumer: FrameConsumer = FrameConsumer.EMPTY,
    ) -> pd.DataFrame:
        """Collect density maps over all runs for given SimulationGroup (no aggregation)

        Returns:
            pd.DataFrame: _description_
        """
        df = []
        scenario_lbl = sim_group.group_name
        for i, _, sim in sim_group.simulation_iter(enum=True):
            _map = sim.get_dcdMap()
            if data is None:
                _df = (
                    _map.map_count_measure()
                )  # all mean, err, sqerr, ... (may be a lot!)
            else:
                _df = _map.map_count_measure().loc[:, data]
                if type(_df) == pd.Series:
                    _df = _df.to_frame()
            _df.columns = pd.MultiIndex.from_product(
                [[scenario_lbl], [i], _df.columns], names=["sim", "run", "data"]
            )
            df.append(_df)
        df = pd.concat(df, axis=1, verify_integrity=True)
        if df.isna().any(axis=1).any():
            nan_index = list(df.index[df.isna().any(axis=1)])
            print(f"found 'nan' valus for time indedx: {nan_index}")
            if drop_nan:
                print(f"dropping time index due to nan: {nan_index}")
                df = df[~(df.isna().any(axis=1))]
        df = df.unstack()
        if isinstance(df, pd.Series):
            df = df.to_frame()
        return frame_consumer(df)

    def run_collect_maps(
        self,
        run_map: RunMap,
        data: List[str] | None = ("map_glb_count", "map_mean_count"),
        frame_consumer: FrameConsumer = FrameConsumer.EMPTY,
        drop_nan: bool = True,
        hdf_path: str | None = None,
        hdf_key: str = "maps",
        pool_size=10,
    ) -> pd.DataFrame:
        """Collect all density maps in provided RunMap. No aggregation performed"""
        if hdf_path is not None and os.path.exists(run_map.path(hdf_path)):
            df = pd.read_hdf(run_map.path(hdf_path), key=hdf_key)
        else:
            df = run_kwargs_map(
                self.sg_collect_maps,
                [
                    dict(
                        sim_group=g,
                        data=data,
                        frame_consumer=frame_consumer,
                        drop_nan=drop_nan,
                    )
                    for g in run_map.values()
                ],
                pool_size=pool_size,
            )
            df = pd.concat(df, axis=0)

            if hdf_path is not None:
                df.to_hdf(run_map.path(hdf_path), mode="a", key=hdf_key, format="table")
        return df.to_frame() if isinstance(df, pd.Series) else df

    def sg_get_merge_maps(
        self,
        sim_group: SimulationGroup,
        data: List[str] | None = ("map_glb_count", "map_mean_count"),
        frame_consumer: FrameConsumer = FrameConsumer.EMPTY,
        drop_nan: bool = True,
    ) -> pd.DataFrame:
        """Get aggregated map over all repetions for given SimulationGroup.
        See sim_get_merge_maps for a single simulation.

        Args:

        Returns:
            pd.DataFrame: Index names ['simtime', ['scenario', 'data']]
        """
        df = self.sg_collect_maps(sim_group, data, drop_nan)

        df = df.groupby(level=["sim", "simtime", "data"]).agg(
            ["mean", "std", percentile(0.5)]
        )  # over multiple runs/seeds

        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(0, axis=1)
        df = frame_consumer(df)
        return df

    def run_get_merge_maps(
        self,
        run_map: RunMap,
        data: List[str] | None = ("map_glb_count", "map_mean_count"),
        frame_consumer: FrameConsumer = FrameConsumer.EMPTY,
        drop_nan: bool = True,
        hdf_path: str | None = None,
        hdf_key: str = "maps",
        pool_size=10,
    ) -> pd.DataFrame:
        """Merge all measurement maps for all simulation groups in given RunMap.
        See sg_merge_maps for simulation group function

        Returns:
            pd.DataFrame: _description_
        """
        if hdf_path is not None and os.path.exists(run_map.path(hdf_path)):
            df = pd.read_hdf(run_map.path(hdf_path), key=hdf_key)
        else:
            df = run_kwargs_map(
                self.sg_get_merge_maps,
                [
                    dict(
                        sim_group=g,
                        data=data,
                        frame_consumer=frame_consumer,
                        drop_nan=drop_nan,
                    )
                    for g in run_map.values()
                ],
                pool_size=pool_size,
            )
            df = pd.concat(df, axis=0)

            if hdf_path is not None:
                df.to_hdf(run_map.path(hdf_path), key=hdf_key, format="table")
        return df.to_frame() if isinstance(df, pd.Series) else df

    def sg_get_packet_loss(
        self,
        sim_group: SimulationGroup,
        app_name: str = "map",
        hdf_path: str = "packet_loss.h5",
        consumer: FrameConsumer = FrameConsumer.EMPTY,
    ) -> pd.DataFrame:
        """Packet loss for simulation group over time over all nodes in a single application. Losses are
        binned over time in one second intervals [[0., 1) .... [N-1, N)]

        Args:
            sim_group (SimulationGroup): _description_
            app_name (str, optional): _description_. Defaults to "map".
            hdf_path (str, optional): _description_. Defaults to "packet_loss.h5".
            consumer (FrameConsumer, optional): _description_. Defaults to FrameConsumer.EMPTY.

        Returns:
            pd.DataFrame: (time, rep)[lost, lost_cumsum, lost_relative]
        """
        df = []
        for rep, sim in sim_group.simulation_iter():
            hdf_store = BaseHdfProvider(os.path.join(sim.data_root, hdf_path))
            if not hdf_store.hdf_file_exists:
                logger.warn("no hdf file found build new one")
                # todo raise?
            raw: pd.DataFrame = hdf_store.get_dataframe(f"pkt_loss_raw_{app_name}")
            # sort lost count by time only
            raw = (
                raw["lost"]
                .fillna(0.0)
                .reset_index("time")
                .reset_index(drop=True)
                .set_index("time")
                .sort_index()
            )
            interval = pd.interval_range(
                start=0.0, end=np.ceil(raw.index.max()), freq=1.0, closed="left"
            )
            raw = (
                raw.groupby(pd.cut(raw.index, bins=interval))
                .sum()
                .reset_index(drop=True)
            )
            raw.index = interval.left
            raw.index.name = "time"
            raw["lost_cumsum"] = raw["lost"].cumsum()
            raw["lost_relative"] = raw["lost_cumsum"] / raw["lost_cumsum"].max()
            raw.columns = pd.MultiIndex.from_tuples(
                [(rep, c) for c in raw.columns], names=["rep", "data"]
            )
            df.append(raw)
        df = pd.concat(df, axis=1).stack(["rep"])
        return consumer(df)

    def run_get_packet_loss(
        self,
        run_map: RunMap,
        app_name: str = "map",
        consumer: FrameConsumer = FrameConsumer.EMPTY,
        pool_size: int = 10,
    ) -> pd.DataFrame:
        """Get packet loss for RunMap"""
        data: List[(pd.DataFrame, dict)] = run_kwargs_map(
            self.sg_get_packet_loss,
            [
                dict(sim_group=v, app_name=app_name, consumer=consumer)
                for v in run_map.get_simulation_group()
            ],
            pool_size=pool_size,
        )
        data: pd.DataFrame = pd.concat(data, axis=0, verify_integrity=True)
        data = data.sort_index()
        return data

    def sg_get_msce_data(
        self,
        sim_group: SimulationGroup,
        cell_count: int,
        cell_slice: Tuple(slice) | pd.MultiIndex = (slice(None), slice(None)),
        cell_slice_fc: FrameConsumer = FrameConsumer.EMPTY,
        consumer: FrameConsumer = FrameConsumer.EMPTY,
    ) -> pd.Series:
        """Mean squared (cell) error for all seed repetition in given SimulationGroup.
        See DcDMap class for simulation based function.

        Args:
            study (SuqcRun): _description_
            run_dict (Parameter_Variation): _description_
            cell_count (int): Number of cells used for normalization. Might differ from map shape if not reachable cells are
                            removed from the analysis. Removed cells must not have any error value.
            consumer (FrameConsumer, optional): Post changes to the collected DataFrame. Defaults to FrameConsumer.EMPTY.

        Returns:
            pd.Series: cell mean squared error over time and run_id index: [simtime, run_id]
        """
        df = []
        print(f"execut group: {sim_group.group_name}")
        if isinstance(cell_slice, pd.MultiIndex):
            if cell_count > 0 and cell_count != cell_slice.shape[0]:
                raise ValueError(
                    "cell slice is given as an index object and cell_count value do not match.  Set cell_count=-1."
                )
            else:
                cell_count = cell_slice.shape[0]

        for rep, sim in sim_group.simulation_iter():
            if sim.sql.is_count_map():
                # handle based on density map counts
                # missing values are set to a count of zero, assuming we do not count any
                # nodes in these cells.
                _df = sim.get_dcdMap().cell_count_measure(
                    columns=["cell_mse"], xy_slice=cell_slice, fc=cell_slice_fc
                )
                _df = _df.groupby(by=["simtime"]).sum() / cell_count
            else:
                # any other kind of value (produced by the entropy map)
                # missing values are removed and  not set to a reasonable estimate.
                _df = sim.get_dcdMap().cell_value_measure(
                    columns=["cell_mse"], xy_slice=cell_slice, fc=cell_slice_fc
                )
                _df = _df.groupby(
                    by=["simtime"]
                ).mean()  # mean of cell mean squared errror over all cels (i.e. MSME)
            _df.columns = [rep]
            _df.columns.name = "run_id"
            print(f"add: {sim_group.group_name}_{sim.run_context.opp_seed}")
            df.append(_df)
        df = pd.concat(df, axis=1, verify_integrity=True)
        df = consumer(df)
        df = df.stack()  # series
        df.name = "cell_mse"
        print(f"done group: {sim_group.group_name}")
        return df

    def run_get_msce_data(
        self,
        run_map: RunMap,
        hdf_path: str,
        cell_count: int,
        cell_slice: Tuple(slice) | pd.MultiIndex = (slice(None), slice(None)),
        cell_slice_fc: FrameConsumer = FrameConsumer.EMPTY,
        pool_size: int = 20,
    ) -> pd.DataFrame:
        """Mean squared (cell) error for *all* ParameterVariations present in given RunMap.
        See sg_get_msce_data for simulation group based function

        Args:
            study (SuqcRun): Suq-controller run object containing the data.
            run_map (RunMap): Map of ParameterVariations under investigation
            hdf_path (str): Path to save result. If it already exist just load it.
            cell_count (int): Number of cells used for normalization. Might differ from map shape if not reachable cells are
                            removed from the analysis. Removed cells must not have any error value.
            pool_size (int): Number of parallel processes used. Default 20.

        Returns:
            pd.DataFrame: cell mean squared error over time, run_id and parameter variation.
                        Index [simtime, run_id]. 'run_id' encodes parameter variations and different seeds.
        """
        if os.path.exists(run_map.path(hdf_path)):
            data = pd.read_hdf(run_map.path(hdf_path), key="cell_mse")
        else:
            data: List[(pd.DataFrame, dict)] = run_kwargs_map(
                self.sg_get_msce_data,
                [
                    dict(
                        sim_group=v,
                        cell_count=cell_count,
                        cell_slice=cell_slice,
                        cell_slice_fc=cell_slice_fc,
                    )
                    for v in run_map.get_simulation_group()
                ],
                pool_size=pool_size,
            )
            data: pd.DataFrame = pd.concat(data, axis=0, verify_integrity=True)
            data = data.sort_index()
            data.to_hdf(run_map.path(hdf_path), key="cell_mse", format="table")
        return data

    def calculate_equality_tests(
        self,
        data: pd.DataFrame,
        combination: list[Tuple[Any, Any]] | None = None,
        lbl_dict: dict | None = None,
        ax: plt.Axes | None = None,
        path: str | None = None,
    ):
        lbl_dict = {} if lbl_dict is None else lbl_dict
        if combination is None:
            combination = list(itertools.combinations(data.columns, 2))
        res = []
        for left, right in combination:
            _s = {}
            _mw = mannwhitneyu(data[left], data[right], method="asymptotic")
            _ks = kstest(data[left], data[right])
            _s["pair"] = f"{lbl_dict.get(left, left)} - {lbl_dict.get(right, right)}"
            _s["mw_stat"] = _mw.statistic
            _s["mw_p"] = _mw.pvalue
            _s["mw_H"] = "$H_0$ (same)" if _s["mw_p"] > 0.05 else "$H_1$ (different)"
            _s["ks_stat"] = _ks.statistic
            _s["ks_p"] = _ks.pvalue
            _s["ks_H"] = "$H_0$ (same)" if _s["ks_p"] > 0.05 else "$H_1$ (different)"
            res.append(_s)
        df = pd.DataFrame.from_records(res, index="pair")
        df.update(df[["mw_stat", "mw_p", "ks_stat", "ks_p"]].applymap("{:.6e}".format))
        df = df.reset_index()

        if path is not None:
            df.to_csv(path)

        if ax is None:
            return df
        else:
            ax.set_title("Test for similarity of distribution")
            ax.axis("off")
            tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
            tbl.scale(1, 2)
            ax.get_figure().tight_layout()
            return df, ax

    def plot_descriptive_comparison(
        self,
        data: pd.DataFrame,
        lbl_dict: dict,
        run_map: RunMap,
        out_name: str,
        stat_col_combination: List[Tuple[Any, Any]] | None = None,
        pdf_file=None,
        palette=None,
        value_axes_label: str = "value",
    ):
        """Save mulitple descriptive plots and statisitcs based on given data.
        DataFrame must be in the long format with a single index.
        """

        if pdf_file is None:
            with run_map.pdf_page(out_name) as pdf:
                self.plot_descriptive_comparison(
                    data,
                    lbl_dict,
                    run_map,
                    out_name,
                    stat_col_combination,
                    pdf,
                    palette,
                )
        else:
            if data.shape[1] <= 3:
                f, (stat_ax, descr_ax) = plt.subplots(2, 1, figsize=(16, 9))
                f = [f]
            else:
                f_1, stat_ax = _Plot.check_ax()
                f_2, descr_ax = _Plot.check_ax()
                f = [f_1, f_2]

            self.calculate_equality_tests(
                data,
                combination=stat_col_combination,
                lbl_dict=lbl_dict,
                ax=stat_ax,
                path=run_map.path(out_name.replace(".pdf", "_stats.csv")),
            )

            descr_ax.set_title("Summary Statistics")
            df = data.describe().applymap("{:.6f}".format).reset_index()
            df.to_csv(run_map.path(out_name.replace(".pdf", "_summary.csv")))
            descr_ax.axis("off")
            tbl = descr_ax.table(cellText=df.values, colLabels=df.columns, loc="center")
            tbl.scale(1, 2)

            for _f in f:
                _f.tight_layout()
                pdf_file.savefig(_f)
                plt.close(_f)

            # Line plot
            f, ax = _Plot.check_ax()
            sns.lineplot(data=data, ax=ax, palette=palette)
            ax.set_title(f"Time Series")
            ax.set_xlabel("Time in seconds")
            ax.set_ylabel(value_axes_label)
            _Plot.rename_legend(ax, rename=lbl_dict)
            pdf_file.savefig(f)
            plt.close(f)

            # ECDF plot
            f, ax = _Plot.check_ax()
            sns.ecdfplot(data, ax=ax, palette=palette)
            ax.set_title(f"ECDF pedestrian count")
            ax.set_xlabel(value_axes_label)
            ax.get_legend().set_title(None)
            sns.move_legend(ax, "upper left")
            _Plot.rename_legend(ax, rename=lbl_dict)
            pdf_file.savefig(f)
            plt.close(f)

            # Hist plot
            f, ax = _Plot.check_ax()
            sns.histplot(
                data,
                cumulative=False,
                common_norm=False,
                stat="percent",
                element="step",
                ax=ax,
                palette=palette,
            )
            ax.set_title(f"Histogram of pedestrian count")
            ax.set_xlabel(value_axes_label)
            ax.get_legend().set_title(None)
            sns.move_legend(ax, "upper left")
            _Plot.rename_legend(ax, rename=lbl_dict)
            pdf_file.savefig(f)
            plt.close(f)

            if data.shape[1] <= 3:
                f, (box, violin) = plt.subplots(1, 2, figsize=(16, 9))
                f = [f]
            else:
                f_box, box = _Plot.check_ax()
                f_violin, violin = _Plot.check_ax()
                f = [f_box, f_violin]

            # Box plot
            sns.boxplot(data=data, ax=box, palette=palette)
            box.set_title(f"Boxplot of pedestrian count")
            box.set_xlabel("Data")
            box.set_ylabel(value_axes_label)

            # Violin plot
            sns.violinplot(data=data, ax=violin, palette=palette)
            violin.set_title(f"Violin of pedestrian count")
            violin.set_xlabel("Data")
            violin.set_ylabel(value_axes_label)
            for _f in f:
                pdf_file.savefig(_f)
                plt.close(_f)


class CellOccupancyInfo:
    """Cell occupation (by time) for cells and the whole map."""

    @classmethod
    def concat(cls, infos: List[CellOccupancyInfo]) -> CellOccupancyInfo:
        ret = {}
        for info in infos:
            for key, df in info.data.items():
                _df = ret.get(key, [])
                _df.append(df)
                ret[key] = _df

        for key, dfs in ret.items():
            _df = pd.concat(dfs, axis=0)
            ret[key] = _df

        return cls(ret)

    @classmethod
    def from_hdf(cls, path, **filter):
        def fix_interval(df: pd.DataFrame):
            bins = pd.IntervalIndex.from_arrays(
                df.index.get_level_values("bin_left"),
                df.index.get_level_values("bin_right"),
                closed="left",
            )
            df["bins"] = bins
            idx = list(df.index.names)
            idx.insert(idx.index("bin_left"), "bins")
            idx.remove("bin_left")
            idx.remove("bin_right")
            df = df.reset_index().drop(columns=["bin_right", "bin_left"]).set_index(idx)
            return df

        ret = {}
        for k in cls._keys():
            if filter == {}:
                df = pd.read_hdf(path, key=k)
                if k == "occup_interval_by_cell":
                    df = fix_interval(df)
                ret[k] = df
            else:
                h5 = BaseHdfProvider(path)
                with h5.query as ctx:
                    data = ctx.select(key=k, **filter)
                    if k == "occup_interval_by_cell":
                        data = fix_interval(data)
                    ret[k] = data
        return cls(ret)

    def __init__(self, data: dict) -> None:
        if not all([k in self._keys() for k in data.keys()]):
            raise ValueError(f"expceted keys: {self._keys()}")
        self.data = data

    def __iter__(self):
        return iter(self.data)

    @property
    def occup_sim_by_cell(self) -> pd.Series:
        return self.data["occup_sim_by_cell"]

    @property
    def occup_sim_by_cell_grid(self) -> pd.Series:
        return self.data["occup_sim_by_cell_grid"]

    @property
    def occup_interval_length(self) -> pd.DataFrame:
        return self.data["occup_interval_length"]

    @property
    def occup_interval_by_cell(self) -> pd.Series:
        return self.data["occup_interval_by_cell"]

    def to_hdf(self, path):
        for key, df in self.data.items():
            _df = df
            if key == "occup_interval_by_cell":
                _df = df.copy()
                bins = _df.index.get_level_values("bins")
                bin_left = [b.left for b in bins]
                bin_right = [b.right for b in bins]
                _df["bin_left"] = bin_left
                _df["bin_right"] = bin_right
                idx = list(_df.index.names)
                idx.insert(idx.index("bins"), "bin_left")
                idx.insert(idx.index("bin_left"), "bin_right")
                idx.remove("bins")
                _df = _df.reset_index().drop(columns=["bins"]).set_index(idx)
            _df.sort_index().to_hdf(path, mode="a", key=key, format="table")

    @staticmethod
    def _keys():
        return [
            "occup_sim_by_cell",
            "occup_sim_by_cell_grid",
            "occup_interval_by_cell",
            "occup_interval_length",
        ]


class _CellOccupancy:
    def sim_create_cell_knowledge_ratio(
        self, sim: Simulation, frame_c: FrameConsumer = FrameConsumer.EMPTY
    ):
        """Callcualte the ratio of agents which have knowledge about a cell measruement at
        each given time, either by sensing or by receiving the measruement value. A ratio a_c
        close to 1 means that most of the agents in the simulaiton have acces to that cell.

        Args:
            sim (Simulation): _description_

        Raises:
            ValueError: (simtime, x, y)[present_count, missing_count, count, a_c]
        """
        c = sim.get_dcdMap().count_p[_i[:, :, :, 1:], ["missing_value"]]
        c_count = c.groupby(["simtime", "x", "y"]).count().set_axis(["count"], axis=1)
        c["count"] = 1
        c_known_ration = (
            c.groupby(["simtime", "x", "y", "missing_value"])
            .count()
            .unstack(["missing_value"])
            .fillna(0.0)  # no missing or all missing
            .droplevel(0, axis=1)
            .rename(columns={False: "present_count", True: "missing_count"})
        )
        c_known_ration = pd.concat(
            [c_known_ration, c_count], axis=1, verify_integrity=True
        )
        if c_known_ration.isnull().any(1).any():
            raise ValueError("NAN found. should not be")
        c_known_ration["a_c"] = (
            c_known_ration["present_count"] / c_known_ration["count"]
        )
        print(c_known_ration.shape)
        return frame_c(c_known_ration)

    def sg_create_cell_knwoledge_ratio(
        self, sim_group: SimulationGroup, frame_c: FrameConsumer = FrameConsumer.EMPTY
    ):
        df = []
        for glb_id, sim in sim_group.simulation_iter():
            print(f"cell knowledge_ratio: {sim_group.group_name}-{glb_id}")
            _df = self.sim_create_cell_knowledge_ratio(sim, frame_c)
            _df = make_run_series(_df, glb_id, lvl_name="rep", stack_index=["rep"])
            _df["m_seed"] = sim.run_context.mobility_seed
            idx = _df.index.names
            _df = _df.reset_index().set_index([*idx, "m_seed"])
            df.append(_df)
        df = pd.concat(df, axis=0)
        return df

    @timing
    def run_create_cell_knowledge_ratio(
        self,
        run_map: RunMap,
        hdf_path: str,
        hdf_key: str = "knowledge_ratio",
        pool_size: int = 20,
        sim_group_filter: SimGroupFilter = SimGroupFilter.EMPTY,
        frame_c: FrameConsumer = FrameConsumer.EMPTY,
    ) -> pd.DataFrame:
        if hdf_path is not None and os.path.exists(run_map.path(hdf_path)):
            logger.info("found H5-file. Read from file")
            df = pd.read_hdf(run_map.path(hdf_path), key=hdf_key)
        else:
            logger.info(
                "file not found creat from scratch with pool size {}", pool_size
            )
            sim_groups = [g for g in run_map.values() if sim_group_filter(g)]
            df = run_kwargs_map(
                self.sg_create_cell_knwoledge_ratio,
                [dict(sim_group=g, frame_c=frame_c) for g in sim_groups],
                pool_size=pool_size,
            )
            df = pd.concat(df, axis=0)

            if hdf_path is not None:
                logger.info("Save data to file {}", hdf_path)
                df.to_hdf(run_map.path(hdf_path), mode="a", key=hdf_key, format="table")
        return df.to_frame() if isinstance(df, pd.Series) else df

    def sim_create_cell_occupation_info(
        self,
        sim: Simulation,
        interval_bin_size: float = 100.0,
        frame_c: FrameConsumer = FrameConsumer.EMPTY,
    ) -> CellOccupancyInfo:
        """Creates occupation statistics on how long and when a cell is occupied by at
        least one agent. This does not mean that the agent did any measurments in this
        cell. It only tracks occupation over time.

        Args:
            sim (Simulation):

        Returns:
            CellOccupancyInfo:
        """
        time_index = sim.get_dcdMap().count_p[_i[:, :, :, 0], ["count"]]
        time_index = pd.Index(
            time_index.index.get_level_values("simtime").unique().sort_values(),
            name="simtime",
        )
        other = time_index.to_frame().reset_index(drop=True)
        d = sim.get_dcdMap().position_df.reset_index().set_index(["simtime", "x", "y"])
        d = d.groupby(d.index.names).count().reset_index(["simtime"])
        _df = []
        _df_intervals = []
        for g, df in d.groupby(["x", "y"]):
            # index (x, y, simtime). With number of nodes in the cell (x, y) at t.
            # note that simtime will not contain all time steps if ther is no agent in
            # that cell at this time
            _d = df.reset_index().set_index(["x", "y", "simtime"]).copy()
            _d.columns = ["cell_occupied"]
            # prepair index of missing cell/time idenices which are not part of _d
            other["x"] = g[0]
            other["y"] = g[1]
            other = other[["x", "y", "simtime"]]
            idx = pd.MultiIndex.from_frame(other, names=["x", "y", "simtime"])
            idx = idx.difference(_d.index)
            # append missing cell/time indecies with cell_occupied value of 0.0
            _d = pd.concat(
                [_d, pd.DataFrame(0, columns=["cell_occupied"], index=idx)],
                axis=0,
                verify_integrity=False,
            ).sort_index()
            # make cell_occupied value to boolean column
            _d["cell_occupied"] = _d["cell_occupied"] >= 1
            # append time diff column to get the 'interval' length of the cell_occupied flag
            # assume something for the first value. Will be removed anyway..
            _time_diff = (
                _d.index.get_level_values("simtime")
                .to_frame()
                .diff()
                .fillna(1.0)
                .values
            )
            _d["occupation_time_delta"] = _time_diff
            # create occupied/empty intervals for current cell
            intervals = []
            _start = None
            _interval_type = None
            changes = _d.index[_d["cell_occupied"].diff().fillna(True).values]
            for c in changes:
                if _start is None:
                    # handle first loop
                    _start = c
                    _interval_type = _d.loc[c, "cell_occupied"]
                    _start = c[-1]
                    continue
                _delta_t = c[-1] - _start
                if len(intervals) == 0 and _interval_type == False:
                    # first tracked intervall must be ouccupied
                    # because the cell only get's occupied at the end of the interval
                    # for the first time. Thus during the current interval we do not have any knowledge.
                    pass
                else:
                    # save interval
                    intervals.append((_interval_type, _start, c[-1], _delta_t))
                _interval_type = _d.loc[c, "cell_occupied"]
                _start = c[-1]
            _idx = pd.MultiIndex.from_tuples(
                [g for _ in range(len(intervals))], names=["x", "y"]
            )
            _df_intervals.append(
                pd.DataFrame(
                    intervals,
                    columns=["cell_occupied", "start", "end", "delta"],
                    index=_idx,
                )
            )
            intervals.clear()
            _df.append(_d)

        # concat data and apply consumer
        _df = pd.concat(_df, axis=0, verify_integrity=True).sort_index()
        _df = frame_c(_df)

        _df_intervals = (
            pd.concat(_df_intervals, axis=0, verify_integrity=False)
            .reset_index()
            .set_index(["x", "y", "cell_occupied"])
            .sort_index()
        )
        _df_intervals = frame_c(_df_intervals)

        occup = _df[_df["cell_occupied"]].drop(columns=["cell_occupied"])
        occup_sim_by_cell = (
            occup.groupby(["x", "y"]).sum()
            / _df.index.get_level_values("simtime").max()
        )

        _t = _df.index.get_level_values("simtime").unique()
        time_interval = pd.interval_range(
            0.0, _t.max(), freq=interval_bin_size, closed="left"
        )
        occup["bins"] = pd.cut(occup.index.get_level_values(-1), time_interval)
        occup_interval_by_cell = (
            occup.reset_index()
            .groupby(["x", "y", "bins"])["occupation_time_delta"]
            .sum()
            / interval_bin_size
        )
        occup_interval_by_cell = occup_interval_by_cell.to_frame()

        idx = sim.get_dcdMap().metadata.create_min_grid_index(
            occup_sim_by_cell.index, difference_only=True
        )
        occup_grid = pd.concat(
            [
                occup_sim_by_cell,
                pd.DataFrame(0.0, columns=["occupation_time_delta"], index=idx),
            ],
            axis=0,
        ).sort_index()

        ret = CellOccupancyInfo(
            {
                "occup_sim_by_cell": occup_sim_by_cell,
                "occup_sim_by_cell_grid": occup_grid,
                "occup_interval_by_cell": occup_interval_by_cell,
                "occup_interval_length": _df_intervals,
            }
        )

        m_seed = sim.run_context.mobility_seed
        for k in ret.data.keys():
            ret.data[k] = append_index(ret.data[k], "seed", m_seed)
        return ret

    def sg_create_cell_occupation_info(
        self,
        sim_group: SimulationGroup,
        interval_bin_size: float = 100.0,
        same_mobility_seed: bool = True,
        frame_c: FrameConsumer = FrameConsumer.EMPTY,
    ) -> CellOccupancyInfo:
        """Creates occupation statistics on how long and when a cell is occupied for each simualtion in the given simulation group

        Args:
            sim_group (SimulationGroup): _description_
            interval_bin_size (float, optional): _description_. Defaults to 100.0.
            same_mobility_seed (bool, optional): Only create CellOccupancyInfo for the first simulation as the mobiliy is the same for all. Defaults to True.

        Returns:
            CellOccupancyInfo: _description_
        """
        ret = {
            "occup_sim_by_cell": [],
            "occup_sim_by_cell_grid": [],
            "occup_interval_by_cell": [],
        }

        if same_mobility_seed:
            # just run for first seed because all mobility seeds are identical
            _ret = self.sim_create_cell_occupation_info(
                sim_group[0], interval_bin_size=interval_bin_size, frame_c=frame_c
            )
            for k, df in _ret.data.items():
                df.columns = pd.MultiIndex.from_tuples(
                    [(0, c) for c in df.columns], names=["rep", "data"]
                )
                s = df.stack(["rep"])
                _df = ret.get(k, [])
                _df.append(s)
                ret[k] = _df
        else:
            for glb_id, sim in sim_group.simulation_iter():
                print(f"{sim_group.group_name}-{glb_id}: create_cell_occupation_info")
                _ret = self.sim_create_cell_occupation_info(
                    sim, interval_bin_size=interval_bin_size, frame_c=frame_c
                )

                for k, df in _ret.data.items():
                    df.columns = pd.MultiIndex.from_tuples(
                        [(glb_id, c) for c in df.columns], names=["rep", "data"]
                    )
                    s = df.stack(["rep"])
                    _df = ret.get(k, [])
                    _df.append(s)
                    ret[k] = _df

        for k, df in ret.items():
            df = pd.concat(df, axis=0)
            ret[k] = df
        ret = CellOccupancyInfo(ret)
        return ret

    def run_create_cell_occupation_info(
        self,
        run_map: RunMap,
        hdf_path: str,
        interval_bin_size: float = 100.0,
        frame_c: FrameConsumer = FrameConsumer.EMPTY,
        pool_size: int = 20,
    ) -> CellOccupancyInfo:
        """Cell occupation info for run_map.

        The cell occupation only depends on the mobility seed. Thus collect occupation for
        each mobility seed only once.
        """

        if hdf_path is not None and os.path.exists(run_map.path(hdf_path)):
            logger.info("found H5-file. Read from file")
            info = CellOccupancyInfo.from_hdf(run_map.path(hdf_path))
        else:
            # search for all mobility seeds
            seed_set = {}
            for sg in run_map.values():
                for sim_id, sim in sg.simulation_iter():
                    if sim.run_context.mobility_seed not in seed_set:
                        seed_set[sim.run_context.mobility_seed] = sim_id

            # collect data from seeds
            infos = run_kwargs_map(
                self.sim_create_cell_occupation_info,
                [
                    dict(
                        sim=run_map.get_sim_by_id(sim_id),
                        interval_bin_size=interval_bin_size,
                        frame_c=frame_c,
                    )
                    for sim_id in seed_set.values()
                ],
                pool_size=pool_size,
            )
            info = CellOccupancyInfo.concat(infos)
            info.to_hdf(run_map.path(hdf_path))
        return info

    def plot_cell_occupation_info(
        self, info: CellOccupancyInfo, run_map: RunMap, fig_path
    ):
        with run_map.pdf_page(fig_path) as pdf:
            m_seeds = run_map.get_mobility_seed_set()
            for seed in m_seeds:
                with plt.rc_context(_Plot.plt_rc_same(size="xx-large")):
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
                    cb = _Plot.add_colorbar(im, aspect=10, pad_fraction=0.5)

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


CellOccupancy = _CellOccupancy()
OppAnalysis = _OppAnalysis()
HdfExtractor = _hdf_Extractor()
