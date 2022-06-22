from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from roveranalyzer.analysis.omnetpp import PlotUtil
from roveranalyzer.utils.dataframe import FrameConsumer


class _VadereAnalysis:
    def __init__(self) -> None:
        pass

    def read_time_step(
        self,
        path,
        sep=" ",
        step=0.4,
        cols: List[str] | None = None,
        frame_consumer: FrameConsumer = FrameConsumer.EMPTY,
    ) -> pd.DataFrame:
        df = pd.read_csv(
            path,
            sep=sep,
            comment="#",
        )
        df.columns = ["time", *df.columns[1:]]
        df["time"] *= step
        df = df.set_index(["time"])
        if cols is not None:
            df.columns = cols

        return frame_consumer(df)

    @PlotUtil.with_axis
    @PlotUtil.savefigure
    def plot_number_agents_over_time(
        self, data: pd.DataFrame, ax: plt.Axes = None, **kwargs
    ):
        ax.set_title("Number of Pedestrians in Simulation")
        ax.set_ylabel("Number of Pedestrians")
        ax.set_xlabel("Time in [s]")
        ax.plot(data.index, data.iloc[:, 0], color="black", label=data.columns[0])
        ax.legend(loc="lower right")

        return ax.get_figure(), ax


VadereAnalysis = _VadereAnalysis()
