from __future__ import annotations

import itertools
from timeit import default_timer
from typing import List, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

import roveranalyzer.simulators.opp.scave as Scave
import roveranalyzer.utils.plot as _Plot
from roveranalyzer.simulators.opp.scave import SqlOp
from roveranalyzer.utils.logging import logger, timing

PlotUtil = _Plot.PlotUtil


class _OppAnalysis:
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
    def get_received_packet_delay(
        self,
        sql: Scave.CrownetSql,
        module_name: Scave.SqlOp | str,
        delay_resolution: float = 1.0,
        describe: bool = True,
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
                "name": "rcvdPktLifetime",
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
        vec_data["rcvdPktLifetime"] *= delay_resolution

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

        vec_names = {
            "rcvdPkSeqNo:vector": {
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


OppAnalysis = _OppAnalysis()
