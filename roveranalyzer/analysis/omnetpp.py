import itertools
from typing import List, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

import roveranalyzer.simulators.opp.scave as Scave
import roveranalyzer.utils.plot as _Plot
from roveranalyzer.simulators.opp.scave import SqlOp

PlotUtil = _Plot.PlotUtil


class _OppAnalysis:
    def __init__(self) -> None:
        pass

    def get_packet_age(
            self,
            sql: Scave.CrownetSql,
            app_path: str,
            host_name: Union[None, str, SqlOp] = None,
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
        df.index.name = 'index'
        return df

    def get_packet_source_distribution(
        self,
        sql: Scave.CrownetSql,
        app_path: str,
        host_name: Union[None, str, SqlOp] = None,
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

    def plot_packet_source_distribution(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        hatch_patterns: List[str] = PlotUtil.hatch_patterns,
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
        return ax

    def get_neighborhood_table_size(
        self,
        sql: Scave.CrownetSql,
        module_name: Union[Scave.SqlOp, str, None] = None,
        indexList: str = "host",
        **kwargs,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
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
        ids = sql.vector_ids_to_host(vec_ids=ids)
        tbl = pd.merge(tbl, ids, how="inner", on=["vectorId"]).drop(
            columns=["vectorId"]
        )
        tbl_idx = tbl[indexList].unique()
        tbl_idx.sort()
        return tbl, tbl_idx

    def plot_neighborhood_table_size_over_time(
        self, ax, tbl: pd.DataFrame, tbl_idx: np.ndarray
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
        return ax


OppAnalysis = _OppAnalysis()
