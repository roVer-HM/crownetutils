import itertools
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import roveranalyzer.simulators.opp as OMNeT
from roveranalyzer.utils import PlotUtil


class _OppAnalysis:
    def __init__(self) -> None:
        pass

    def get_packet_source_distribution(
        self, sql: OMNeT.CrownetSql, app_path: str, normalize: bool = True
    ) -> pd.DataFrame:
        """
        Create square matrix of [hostId X hostId] showing the source hostId of received packets for the given application path.
        Example:
        hostId/hostId |  1  |  2  |  3  |
            1       |  0  |  4  |  8  |
            2       |  1  |  0  |  1  |
            3       |  6  |  6  |  0  |
        host_1 received 4 packets from host_2
        host_1 received 8 packets from host_3
        host_2 received 1 packet  from host_1
        host_2 received 1 packet  from host_3
        host_3 received 6 packets from host_1
        host_3 received 6 packets from host_2
        """
        id_map = sql.host_ids()
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


OppAnalysis = _OppAnalysis()
