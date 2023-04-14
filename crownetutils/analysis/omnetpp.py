from __future__ import annotations

import itertools
import os
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import IndexSlice as _i

sns.set(font_scale=1.0, rc={"text.usetex": True})
import crownetutils.omnetpp.scave as Scave
import crownetutils.utils.plot as _Plot
from crownetutils.analysis.base import AnalysisBase
from crownetutils.analysis.common import (
    RunMap,
    SimGroupFilter,
    Simulation,
    SimulationGroup,
)
from crownetutils.analysis.dpmm.dpmm import percentile
from crownetutils.analysis.hdf.provider import BaseHdfProvider
from crownetutils.omnetpp.scave import CrownetSql, SqlEmptyResult, SqlOp
from crownetutils.utils.dataframe import (
    FrameConsumer,
    append_index,
    format_frame,
    siunitx,
)
from crownetutils.utils.logging import logger, timing
from crownetutils.utils.misc import DataSource
from crownetutils.utils.parallel import run_kwargs_map
from crownetutils.utils.plot import PlotUtil, with_axis


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
        pos = sql.node_position()
        _hdf.write_frame(group="trajectories", frame=pos)

    @classmethod
    def extract_rvcd_statistics(cls, hdf_file: str, sql: Scave.CrownetSql):
        _hdf = BaseHdfProvider(hdf_file, "rcvd_stats")
        if _hdf.hdf_file_exists:
            logger.info("hdf file exists nothing to do.")
            return
        df = []
        for module_name, m_str in [(sql.m_beacon(), "b"), (sql.m_map(), "m")]:
            logger.info(f"read vector data for {module_name}")
            seqNo_vec = ["rcvdPkSeqNo:vector", "rcvdPktPerSrcSeqNo:vector"]
            seqNo_vec = sql.find_vector_name(module_name, seqNo_vec)
            vec_names = {
                seqNo_vec: {
                    "name": "seqNo",
                    "dtype": np.int32,
                },
                "rcvdPktPerSrcLossCount:vector": dict(
                    name="pkt_loss_sum", dtype=np.int32
                ),
                "rcvdPktPerSrcCount:vector": dict(
                    name="total_pkt_received", dtype=np.int32
                ),
                "rcvdPkHostId:vector": dict(name="srcHostId", dtype=np.int32),
                "rcvdPktPerSrcJitter:vector": dict(name="jitter", dtype=np.float32),
                "rcvdPkLifetime:vector": dict(name="delay", dtype=np.float32),
            }

            vec_data = sql.vec_data_pivot(
                module_name, vec_names, append_index=["srcHostId"]
            ).sort_index()

            # drop self messages, where hostId == srcHostId
            _shape = vec_data.shape
            _m = vec_data.index.get_level_values(
                "hostId"
            ) != vec_data.index.get_level_values("srcHostId")
            vec_data = vec_data[_m].copy(deep=True)
            logger.info(
                f"remove self references (hostId==srcHostId): {_shape}->{vec_data.shape} "
            )

            # packet loss
            vec_data["total_pkt_send"] = (
                vec_data["pkt_loss_sum"] + vec_data["total_pkt_received"]
            )
            vec_data["PRR"] = (
                vec_data["total_pkt_received"] / vec_data["total_pkt_send"]
            )

            # add application tag and update index
            vec_data["app"] = m_str
            vec_data = vec_data.set_index(["app"], append=True).sort_index()
            df.append(vec_data)
        df = pd.concat(df, axis=0, verify_integrity=False)
        _hdf.write_frame(group="rcvd_stats", frame=df)

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

    @with_axis
    def plot_packet_source_distribution(
        self,
        data: pd.DataFrame,
        hatch_patterns: List[str] = PlotUtil._hatch_patterns,
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

    @with_axis
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
    def get_received_host_ids(
        self,
        sql: Scave.CrownetSql,
        module_name: Scave.SqlOp | str,
    ):
        # add eventNumber to allow concat with rx host id data
        columns = ["vectorId", "simtimeRaw", "value", "eventNumber"]
        df = sql.vector_ids_to_host(
            module_name,
            "rcvdPkHostId:vector",
            name_columns=["hostId"],
            pull_data=True,
            value_name="srcHostId",
            columns=columns,
            drop=["vectorId"],
        )
        df["srcHostId"] = df["srcHostId"].astype(int)
        return df

    def get_rcvd_generic_vec_data(
        self,
        sql: Scave.CrownetSql,
        module_name: Scave.SqlOp | str,
        vector_name: str,
        value_name: str,
        with_host_id: bool = False,
        append_source_id: bool = False,
        drop_self_message: bool = False,
        drop_col: List[str] | None = ("vectorId",),
    ) -> pd.DataFrame:
        if drop_self_message:
            logger.info(
                f"drop data points where hostId (i.e. receiving node) and tx_host_id are equal."
            )
            # drop_self_message implies that these must be true!
            append_source_id = True
            with_host_id = True
        if append_source_id:
            columns = ["vectorId", "simtimeRaw", "value", "eventNumber"]
        else:
            columns = ["vectorId", "simtimeRaw", "value"]

        if with_host_id or append_source_id:
            df = sql.vector_ids_to_host(
                module_name,
                vector_name,
                pull_data=True,
                value_name=value_name,
                name_columns=["hostId"],
                columns=columns,
                drop=drop_col if drop_col is None else list(drop_col),
            )
        else:
            df = sql.vec_data(
                module_name=module_name,
                vector_name=vector_name,
                value_name=value_name,
                drop=drop_col if drop_col is None else list(drop_col),
            )

        if append_source_id:
            df = self._merge_rx_host_id(df, sql, module_name, drop_self_message)
        return df

    @timing
    def get_received_packet_jitter(
        self,
        sql: Scave.CrownetSql,
        module_name: Scave.SqlOp | str,
        with_host_id: bool = False,
        append_source_id: bool = False,
        drop_self_message: bool = True,
        drop_col: List[str] | None = ("vectorId",),
    ) -> pd.DataFrame:
        return self.get_rcvd_generic_vec_data(
            sql,
            module_name,
            vector_name="rcvdPktPerSrcJitter:vector",
            value_name="jitter",
            with_host_id=with_host_id,
            append_source_id=append_source_id,
            drop_self_message=drop_self_message,
            drop_col=drop_col,
        )

    @timing
    def get_received_packet_delay(
        self,
        sql: Scave.CrownetSql,
        module_name: Scave.SqlOp | str,
        with_host_id: bool = False,
        append_source_id: bool = False,
        drop_self_message: bool = True,
        drop_columns: List[str] | None = ("vectorId",),
    ) -> pd.DataFrame:
        return self.get_rcvd_generic_vec_data(
            sql,
            module_name,
            vector_name="rcvdPkLifetime:vector",
            value_name="delay",
            with_host_id=with_host_id,
            append_source_id=append_source_id,
            drop_self_message=drop_self_message,
            drop_col=drop_columns,
        )

    @timing
    def get_received_packet_loss(
        self,
        sql: Scave.CrownetSql,
        module_name: Scave.SqlOp | str,
        with_host_id: bool = False,
        append_source_id: bool = False,
        drop_self_message: bool = True,
        drop_columns: List[str] | None = ("vectorId",),
    ) -> pd.DataFrame:
        return self.get_rcvd_generic_vec_data(
            sql,
            module_name,
            vector_name="rcvdPktPerSrcLossCount:vector",
            value_name="pkt_loss_sum",
            with_host_id=with_host_id,
            append_source_id=append_source_id,
            drop_self_message=drop_self_message,
            drop_col=drop_columns,
        )

    @timing
    def get_received_packet_count_per_source(
        self,
        sql: Scave.CrownetSql,
        module_name: Scave.SqlOp | str,
        with_host_id: bool = False,
        append_source_id: bool = False,
        drop_self_message: bool = True,
        drop_columns: List[str] | None = ("vectorId",),
    ) -> pd.DataFrame:
        return self.get_rcvd_generic_vec_data(
            sql,
            module_name,
            vector_name="rcvdPktPerSrcCount:vector",
            value_name="total_pkt_received",
            with_host_id=with_host_id,
            append_source_id=append_source_id,
            drop_self_message=drop_self_message,
            drop_col=drop_columns,
        )

    def _merge_rx_host_id(
        self, data: pd.DataFrame, sql, module_name, drop_self_msg: bool = False
    ):
        """Merge receiving host id with data from data vector."""
        tx_host_ids = self.get_received_host_ids(sql, module_name)
        data = data.set_index(["eventNumber", "time", "hostId"]).sort_index()
        tx_host_ids = tx_host_ids.set_index(
            ["eventNumber", "time", "hostId"]
        ).sort_index()
        if not data.index.is_unique or not data.index.is_unique:
            logger.warn(f"index is not unique")
        data = pd.concat(
            [data, tx_host_ids], axis=1, ignore_index=False, verify_integrity=True
        )
        data = data.reset_index()
        if drop_self_msg:
            _mask = data["hostId"] == data["srcHostId"]
            data = data[~_mask]
        data = data.set_index(
            ["hostId", "srcHostId", "eventNumber", "time"]
        ).sort_index()
        return data

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
    def get_map_pkt_count_ts(
        self,
        sql: Scave.CrownetSql,
    ):
        df = sql.vector_ids_to_host(
            module_name=sql.m_map(),
            vector_name="packetSent:vector(packetBytes)",
            name_columns=["hostId"],
            pull_data=True,
            value_name="sentBytes",
            index=["time", "hostId"],
        ).drop(columns=["vectorId"])
        df = (
            df.groupby(["time", "hostId"])
            .agg(["count", "sum"])
            .set_axis(["pkt_count", "byte_count"], axis=1)
        )
        return df

    @timing
    def plot_map_pkt_count_all(
        self,
        data_root: str,
        sql: Scave.CrownetSql,
        saver: _Plot.FigureSaver | None = None,
    ):
        saver = _Plot.FigureSaver.FIG(saver)
        data = self.get_map_pkt_count_ts(sql)
        fig, ax = plt.subplots()
        _Plot.PlotUtil.df_to_table(
            data.describe().applymap("{:1.4f}".format).reset_index(), ax
        )
        ax.set_title(f"Descriptive statistics for map application")
        saver(fig, os.path.join(data_root, f"tx_MapPkt_stat.pdf"))

        fig, ax = PlotUtil.check_ax()
        ax.scatter("time", "pkt_count", data=data.reset_index())
        ax.set_title("Packet count over time")
        ax.set_ylabel("Number of packets")
        ax.set_xlabel("Simulation time in seconds")
        saver(fig, os.path.join(data_root, f"txMapPktCount_ts.pdf"))

    def append_run_col(self, func, run: int, **kwargs) -> pd.DataFrame:
        print(f"run func for run {run}")
        df = func(**kwargs)
        df["run"] = run
        df = df.reset_index()
        return df

    @timing
    def sg_get_txAppInterval(
        self,
        sg: SimulationGroup,
        app_type: str = "beacon",
        interval_type: str = "all",
        jobs: int = 5,
    ) -> pd.DataFrame:
        df = []
        kw_list = [
            dict(
                func=self.get_txAppInterval,
                run=run_id,
                sql=sim.sql,
                app_type=app_type,
                interval_type=interval_type,
            )
            for run_id, sim in sg.simulation_iter()
        ]
        df = run_kwargs_map(func=self.append_run_col, kwargs_iter=kw_list, pool_size=5)
        # for run_id, sim in sg.simulation_iter():
        #     print(f"read run {run_id}")
        #     _df = self.get_txAppInterval(sim.sql, app_type, interval_type)
        #     _df["run"] = run_id
        #     _df = _df.reset_index()
        #     df.append(_df)
        df = pd.concat(df, ignore_index=True, verify_integrity=False)
        return df

    @timing
    def get_txAppInterval(
        self,
        sql: Scave.CrownetSql,
        app_type: str = "beacon",
        interval_type: str = "all",
    ) -> pd.DataFrame:
        if app_type.lower() == "beacon":
            m = sql.m_beacon(app_mod="scheduler")
        else:
            m = sql.m_map(app_mod="scheduler")

        if interval_type in ["all", "real"]:
            df1 = sql.vector_ids_to_host(
                module_name=m,
                vector_name="txInterval:vector",
                name_columns=["hostId"],
                pull_data=True,
                value_name="txInterval",
                index=["time", "hostId"],
            ).drop(columns=["vectorId"])
        if interval_type in ["all", "det"]:
            df2 = sql.vector_ids_to_host(
                module_name=m,
                vector_name="txDetInterval:vector",
                name_columns=["host", "hostId"],
                pull_data=True,
                value_name="txDetInterval",
                index=["time", "hostId"],
            ).drop(columns=["vectorId"])
        if interval_type == "real":
            return df1
        elif interval_type == "det":
            return df2
        else:
            return pd.concat([df1, df2], axis=1, ignore_index=False)

    def _load_vectors(
        self,
        module_index,
        sql: Scave.CrownetSql,
        module_f,
        vector_name: SqlOp | str | None = None,
    ):
        df = sql.vector_ids_to_host(
            module_f(idx=module_index),
            vector_name,
            pull_data=True,
            columns=["vectorId", "simtimeRaw", "value", "eventNumber"],
            name_columns=["hostId"],
        )
        df = (
            df.set_index(["eventNumber", "time", "hostId", "vectorId"])
            .unstack(["vectorId"])
            .droplevel(0, axis=1)
        )
        vec_names = sql.vec_info(
            vector_ids=df.columns.to_list(), cols=["vectorId", "vectorName"]
        )
        vec_names["vectorName"] = vec_names["vectorName"].apply(
            lambda x: x[: x.index(":")]
        )
        names = dict()
        for idx, row in vec_names.reset_index().iterrows():
            names[row["vectorId"]] = row["vectorName"]
        df = df.rename(columns=names)
        df["app"] = 0
        df = df.reset_index().set_index(["app", "eventNumber", "time", "hostId"])
        return df

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

    def build_received_packet_loss_cache(
        self, sql: Scave.CrownetSql, hdf_path: str, return_group: str = "Map"
    ) -> pd.DataFrame:
        hdf = BaseHdfProvider(hdf_path=hdf_path)
        if return_group is not None and return_group not in ["Beacon", "Map", "Both"]:
            raise ValueError("Expected Map, Beacon or Both as return_group")
        if hdf.contains_group("Beacon"):
            logger.info("found Beacon group nothing to do")
        else:
            logger.info("build Beacon group...")
            df = self.get_received_packet_loss(sql, sql.m_beacon())
            hdf.put_frame_fixed("Beacon", df)

        if hdf.contains_group("Map"):
            logger.info("found Map group nothing to do")
        else:
            logger.info("build Map group...")
            df = self.get_received_packet_loss(sql, sql.m_map())
            hdf.put_frame_fixed("Map", df)

        if return_group is None:
            return None
        if return_group == "Both":
            d1 = hdf.get_dataframe("Beacon")
            idx_names = list(d1.index.names)
            d1["app"] = "Beacon"
            d2 = hdf.get_dataframe("Map")
            d2["app"] = "Map"
            df = pd.concat(
                [d1.reset_index(), d2.reset_index()],
                axis=0,
                ignore_index=True,
                verify_integrity=False,
            )
            df = df.set_index([*idx_names, "app"]).sort_index()
            return df
        return hdf.get_dataframe(return_group)

    @timing
    def get_received_packet_loss2(
        self,
        sql: Scave.CrownetSql,
        module_name: Scave.SqlOp | str,
    ) -> pd.DataFrame:

        logger.info("load packet loss data from *.vec")

        # statistic was renamed. Check old version fist.
        seqNo_vec = ["rcvdPkSeqNo:vector", "rcvdPktPerSrcSeqNo:vector"]
        seqNo_vec = sql.find_vector_name(module_name, seqNo_vec)
        vec_names = {
            seqNo_vec: {
                "name": "seqNo",
                "dtype": np.int32,
            },
            "rcvdPktPerSrcLossCount:vector": dict(name="pkt_loss_sum", dtype=np.int32),
            "rcvdPktPerSrcCount:vector": dict(
                name="total_pkt_received", dtype=np.int32
            ),
            "rcvdPkHostId:vector": dict(name="srcHostId", dtype=np.int32),
            "rcvdPktPerSrcJitter:vector": dict(name="jitter", dtype=np.float32),
            "rcvdPkLifetime:vector": dict(name="delay", dtype=np.float32),
        }

        vec_data = sql.vec_data_pivot(
            module_name, vec_names, append_index=["srcHostId"]
        ).sort_index()

        # drop self messages, where hostId == srcHostId
        _shape = vec_data.shape
        _m = vec_data.index.get_level_values(
            "hostId"
        ) != vec_data.index.get_level_values("srcHostId")
        vec_data = vec_data[_m].copy(deep=True)
        logger.info(
            f"remove self references (hostId==srcHostId): {_shape}->{vec_data.shape} "
        )
        vec_data["total_pkt_send"] = (
            vec_data["pkt_loss_sum"] + vec_data["total_pkt_received"]
        )
        vec_data["PRR"] = vec_data["total_pkt_received"] / vec_data["total_pkt_send"]

        return vec_data

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
    def get_sent_packet_bytes_by_app(
        self,
        sql: Scave.CrownetSql,
        hdf: BaseHdfProvider | None = None,
        hdf_group: str = "tx_pkt_bytes",
    ) -> pd.DataFrame:

        if hdf is not None:
            if hdf.contains_group(hdf_group):
                return hdf.get_dataframe(hdf_group)
        print("create tx_pkt_beacon")
        tx_pkt_beacon = sql.vec_data(
            sql.m_beacon(), "packetSent:vector(packetBytes)"
        ).drop(columns=["vectorId"])
        tx_pkt_beacon["app"] = "b"
        tx_pkt_map = sql.vec_data(sql.m_map(), "packetSent:vector(packetBytes)").drop(
            columns=["vectorId"]
        )
        tx_pkt_map["app"] = "m"

        tx_pkt = (
            pd.concat([tx_pkt_beacon, tx_pkt_map], axis=0, ignore_index=True)
            .set_index(["app", "time"])
            .sort_index()
        )
        if hdf is not None:
            print(f"write frame to hdf. group: {hdf_group}")
            hdf.write_frame(hdf_group, tx_pkt)
        return tx_pkt

    def get_sent_packet_throughput_diff_by_app(
        self,
        sql: Scave.CrownetSql,
        target_rate: dict,
        freq: float = 1.0,
        hdf: BaseHdfProvider | None = None,
        hdf_group_base: str = "tx_throughput",
    ):

        hdf_group_diff = f"{hdf_group_base}_diff_{str(freq).replace('.','_')}"
        if hdf is not None and hdf.contains_group(hdf_group_diff):
            return hdf.get_dataframe(hdf_group_diff)

        hdf_group = f"{hdf_group_base}{str(freq).replace('.','_')}"
        if hdf is not None and hdf.contains_group(hdf_group):
            print("found group")
            data = hdf.get_dataframe(hdf_group)
        else:
            print("create packet throughput by app")
            data = self.get_sent_packet_throughput_by_app(sql, freq=freq, hdf=hdf)
        for c in data.columns:
            _rate = target_rate[c]
            data[f"diff_{c}"] = data[c] - _rate
        if hdf is not None:
            hdf.write_frame(hdf_group_diff, data)

        return data

    def get_sent_packet_throughput_by_app(
        self,
        sql: Scave.CrownetSql,
        freq: float = 1.0,
        tx_byte_data: pd.DataFrame | None = None,
        hdf: BaseHdfProvider | None = None,
        hdf_group_base: str = "tx_throughput",
    ):
        hdf_group = f"{hdf_group_base}{str(freq).replace('.','_')}"
        if hdf is not None and hdf.contains_group(hdf_group):
            return hdf.get_dataframe(hdf_group)

        if tx_byte_data is None:
            data = self.get_sent_packet_bytes_by_app(sql, hdf=hdf)
        else:
            data = tx_byte_data

        tx_rate = data.reset_index(["app"])
        bins = pd.interval_range(
            start=0.0,
            end=tx_rate.index.get_level_values(0).max(),
            freq=freq,
            closed="right",
        )
        # rate in kilo bytes per seconds
        tx_rate = (
            (
                tx_rate.groupby([pd.cut(tx_rate.index, bins=bins), "app"]).sum()
                / freq
                / 1000
            )
            .unstack("app")
            .droplevel(0, axis=1)
        )
        tx_rate.index = bins.right
        tx_rate.index.name = "time"
        if hdf is not None:
            hdf.write_frame(hdf_group, tx_rate)
        return tx_rate

    # def get_sent_packet_bytes_for_map(
    #     self,
    #     sql: Scave.CrownetSql,
    #     freq: float = 1.0,
    # ) -> pd.DataFrame:
    #     vec_ids = sql.vec_ids(
    #         sql.m_map(),
    #         "packetSent:vector(packetBytes)",
    #     )
    #     df = (
    #         sql.vec_data(
    #             # sql.m_map(),
    #             # "packetSent:vector(packetBytes)",
    #             ids=vec_ids[0:50],
    #             time_slice=slice(0, 100.0),
    #         )
    #         .drop(columns=["vectorId"])
    #         .sort_index()
    #     )

    #     bins = pd.interval_range(
    #         start=0.0,
    #         end=df.index.get_level_values(0).max(),
    #         freq=freq,
    #         closed="right",
    #     )
    #     df = df.groupby(pd.cut(df.index, bins)).sum() / freq / 1000  # kbps

    #     return df

    # def sg_get_sent_packet_throughput_by_app(
    #     self, sim_group: SimulationGroup, freq: float = 1.0
    # ) -> pd.DataFrame:
    #     dfs = []
    #     for rep, sim in sim_group.simulation_iter():
    #         print(rep, sim)
    #         df = self.get_sent_packet_bytes_for_map(sim.sql, freq=freq)
    #         df["rep"] = rep
    #         dfs.append(df.reset_index())

    #     dfs = pd.concat(dfs, axis=1, ignore_index=True, verify_integrity=False)
    #     dfs = dfs.set_index(["time", "app", "rep"])
    #     return dfs

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

    def merge_position(
        self,
        sim_group: SimulationGroup,
        time_slice=slice(0.0),
        frame_consumer: FrameConsumer = FrameConsumer.EMPTY,
    ) -> pd.DataFrame:
        df = []
        for run_id, sim in sim_group.simulation_iter():
            _pos = sim.sql.node_position(
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
                    cb = PlotUtil.add_colorbar(im, aspect=10, pad_fraction=0.5)

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
