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
from roveranalyzer.utils.logging import logger

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
        Get packet ages for any stationary and moving node x
        Packet age: (time x received packet i) - (time packet i created)
                |hostId (x)    |  time_recv |  packet_age  |
            0   |  12          |  1.2  |  0.3  |
            1   |  12          |  1.3  |  0.4  |
            ... |  55          |  8.3  |  4.6  |
        """

        id_map = sql.host_ids(host_name)
        if not app_path.startswith("."):
            app_path = f".{app_path}"

        df = None
        for _id, host in id_map.items():

            _df = sql.vec_data(
                module_name=f"{host}{app_path}",
                vector_name="rcvdPkLifetime:vector",
            )
            _df["hostId"] = _id

            if df is None:
                df = _df
            else:
                df = pd.concat([df, _df], axis=0)

        df = df.loc[:, ["hostId", "time", "value"]]
        df.sort_values(by=["hostId"], inplace=True)
        df.rename(columns={"value": "packet_age", "time": "time_recv"}, inplace=True)
        df["hostId"] = df["hostId"].astype(int)
        df.reset_index(drop=True, inplace=True)
        df.index.name = "index"
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
        indexList: str = "host",
        **kwargs,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Extract neighborhood table size vectors from given module_name. Use the sql Operator for multiple selections.
        and None to default to all module vectors (misc, pNode, vNode)
        """
        if module_name is None:
            module_name = sql.OR(
                [f"{sql.network}.{i}[%].nTable" for i in sql.module_vectors]
            )

        tbl = sql.vec_data(
            module_name=module_name,  # "World.misc[%].nTable",
            vector_name="tableSize:vector",
        )

        ids = [str(i) for i in tbl["vectorId"].unique()]
        ids = sql.vector_ids_to_host(vector_ids=ids)
        tbl = pd.merge(tbl, ids, how="inner", on=["vectorId"]).drop(
            columns=["vectorId"]
        )
        tbl_idx = tbl[indexList].unique()
        tbl_idx.sort()
        return tbl, tbl_idx

    @PlotUtil.with_axis
    def plot_neighborhood_table_size_over_time(
        self, tbl: pd.DataFrame, tbl_idx: np.ndarray, ax: plt.Axes = None
    ) -> plt.Axes:
        """
        x-axis: time
        y-axis: number of entries in neighborhood table
        data: selected hosts
        """
        _c = PlotUtil.color_lines(line_type=None)
        for i in tbl_idx:
            _df = tbl.loc[tbl["host"] == i]
            ax.plot(_df["time"], _df["value"], next(_c), label=i)

        ax.set_ylabel("Neighboor count")
        ax.set_xlabel("time [s]")
        ax.set_title("Size of neighborhood table over time for each node")
        return ax.get_figure(), ax

    def get_received_packet_delay(
        self,
        sql: Scave.CrownetSql,
        module_name: Scave.SqlOp | str,
        delay_resolution: float = 1.0,
        describe: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Packet delay data based on single applications in '<Network>.<Module>[*].app[*].app'

        Args:
            sql (Scave.CrownetSql): Scave database handler
            module_name (Scave.SqlOp): [description]
            delay_resolution (float, optional): Delay resolution mutliplier. Defaults to 1.0 (seconds).
            describe (bool, optional): [description]. If true second data frame contains descriptive statistics based on hostId/srcHostId. Defaults to True.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: DataFrame of the form (index)[columns]:
            (1) raw data (hostId, srcHostId, time)[delay]
            (2) empty  or (hostId, srcHostId)[count, mean, std, min, 25%, 50%, 75%, max] if describe=True
        """
        vec_names = ["rcvdPkLifetime:vector", "rcvdPkHostId:vector"]
        df = sql.vector_ids_to_host(
            module_name,
            sql.OR(vec_names),
            vec_info_columns=["vectorId", "vectorName"],
            name_columns=["hostId"],
        )
        vec_data = sql.vec_data(ids=df, value_name="delay")
        vec_data = vec_data.rename(columns={"vectorName": "value_type"})
        vec_data["value_type"] = vec_data["value_type"].map(
            {"rcvdPkLifetime:vector": "delay", "rcvdPkHostId:vector": "srcHostId"}
        )
        vec_data = (
            vec_data.drop(columns=["vectorId"])
            .pivot(index=["hostId", "time"], columns=["value_type"])
            .droplevel(level=0, axis=1)
            .reset_index()
        )
        col_dtypes = sql.get_column_types(
            vec_data.columns.to_list(), time=float, delay=float, srcHostId=np.int32
        )
        vec_data = vec_data.astype(col_dtypes)
        vec_data.columns.name = ""
        vec_data = vec_data.set_index(
            keys=["hostId", "srcHostId", "time"], verify_integrity=True
        )
        vec_data = vec_data.sort_index()
        vec_data["delay"] *= delay_resolution

        if describe:
            return vec_data, vec_data.groupby(level=["hostId", "srcHostId"]).describe()
        else:
            return vec_data, pd.DataFrame()

    def get_received_packet_loss(
        self, sql: Scave.CrownetSql, module_name: Scave.SqlOp | str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Packet loss data based on single applications in '<Network>.<Module>[*].app[*].app'

        Args:
            sql (Scave.CrownetSql): Scave database handler
            module_name (Scave.SqlOp, optional): Modules for which the packet loss is calculated

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:  DataFrames of the form (index)[columns]:
            (1) aggregated  packet loss of the form (hostId, srcHostId) [numPackets, packet_lost, packet_loss_ratio]
            (2) raw data (hostId, srcHostId, time)[seqNo]
        """
        time_start = default_timer()
        logger.info("load packet loss data from *.vec")

        vec_names = ["rcvdPkSeqNo:vector", "rcvdPkHostId:vector"]
        vec_info = sql.vector_ids_to_host(
            module_name,
            sql.OR(vec_names),
            vec_info_columns=["vectorId", "vectorName"],
            name_columns=["hostId"],
        )
        vec_data = sql.vec_data(ids=vec_info)
        logger.info(f"received data with shape {vec_data.shape}")
        vec_data = vec_data.rename(columns={"vectorName": "value_type"})
        vec_data["value_type"] = vec_data["value_type"].map(
            {"rcvdPkSeqNo:vector": "seqNo", "rcvdPkHostId:vector": "srcHostId"}
        )
        vec_data = (
            vec_data.drop(columns=["vectorId"])
            .pivot(index=["hostId", "time"], columns=["value_type"])
            .droplevel(level=0, axis=1)
            .reset_index()
        )
        col_dtypes = sql.get_column_types(
            vec_data.columns.to_list(), time=float, seqNo=np.int32, srcHostId=np.int32
        )
        vec_data = vec_data.astype(col_dtypes)
        vec_data.columns.name = ""
        vec_data = vec_data.set_index(
            keys=["hostId", "srcHostId", "time"], verify_integrity=True
        )
        vec_data = vec_data.sort_index()
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
        logger.info(f"packet loss ratio done {default_timer() - time_start}")
        return lost_df, vec_data


OppAnalysis = _OppAnalysis()
