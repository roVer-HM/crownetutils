from typing import Any, Protocol

import numpy as np
import pandas as pd


class MissingValueImputationStrategy(Protocol):
    """Imputation strategy to fill or remove missing values from a frame. Note that in case of removing the whole
    row will be removed."""

    def __call__(
        self, df: pd.DataFrame, data_column, *args: Any, **kwds: Any
    ) -> pd.DataFrame:
        ...


class ArbitraryValueImputation:
    """Replace missing values on the measurement and ground truth side
    with a predefined fixed value.

    Example use case:
    For DPM maps which count the number of pedestrians using beacon communication,
    and we do not  'hear' (i.e. receive data) from a cell, we  can assume that there
    is no one in that cell, and thus we replace NAN values with zero.
    Similarly if a node has measurements for a cell, but the ground truth does not
    have any value for that cell we can replace the ground truth NAN-value with zero
    """

    def __init__(
        self, fill_value=0.0, glb_prefix: str = "glb_", glb_fill_value=0.0
    ) -> None:
        self.fill_value = fill_value
        self.glb_prefix = glb_prefix
        self.glb_fill_value = glb_fill_value

    def __call__(
        self, df: pd.DataFrame, data_column, *args: Any, **kwds: Any
    ) -> pd.DataFrame:
        glb_data_column = f"{self.glb_prefix}{data_column}"
        df[data_column] = df[data_column].fillna(self.fill_value)
        df[glb_data_column] = df[glb_data_column].fillna(self.glb_fill_value)
        return df


class ArbitraryValueImputationWithRsd(ArbitraryValueImputation):
    """Replace missing values on the measurement and ground truth side
    with a predefined fixed value. In addition replace missing RSD ID's
    of measurements with the RSD that is the closest to it (euclidean distance).
    """

    def __init__(
        self,
        rsd_origin_position: pd.DataFrame,
        rsd_col="rsd_id",
        fill_value=0,
        glb_prefix: str = "glb_",
        glb_fill_value=0,
    ) -> None:
        super().__init__(fill_value, glb_prefix, glb_fill_value)
        self.rsd_col = rsd_col
        self.rsd_distance = rsd_origin_position.rename(
            columns={"x": "enb_x", "y": "enb_y"}
        )  # DataFrame[x, y, rsd_id]

    def __call__(
        self, df: pd.DataFrame, data_column, *args: Any, **kwds: Any
    ) -> pd.DataFrame:
        df = self.update_rsd_value(df, data_column)
        super().__call__(df, data_column=data_column)
        return df

    def update_rsd_value(self, df: pd.DataFrame, data_column):
        # find *unique* cells location (x, y) with nan valued rsd
        # for distance calculation for nearest rsd_origin (i.e. base station)
        null_index = df[df[self.rsd_col].isna()].index
        xy = null_index.droplevel("simtime").unique().to_frame().reset_index(drop=True)
        # create cross product of missing measurements with enb_positions
        # and calculate the 2-norm for each row (i.e. distance cell to base station)
        xy = xy.join(self.rsd_distance, how="cross")
        xy["dist"] = np.linalg.norm(
            xy.loc[:, ["enb_x", "enb_y"]].values - xy.loc[:, ["x", "y"]].values, axis=1
        )
        xy = xy.sort_values(["x", "y", "dist"])
        # select for each cell (x, y) only the nearest base station
        # not mask must be inverted as it marks due duplicates as true.
        _mask = ~xy.duplicated(subset=["x", "y"], keep="first")
        xy_rsd = (
            xy.loc[_mask, ["x", "y", "rsd_id"]]
            .copy(deep=True)
            .set_axis(["x", "y", self.rsd_col], axis=1)
        )
        xy_rsd = (
            null_index.to_frame()
            .reset_index(drop=True)
            .merge(xy_rsd, how="left")
            .set_index(["simtime", "x", "y"])
        )
        df.loc[xy_rsd.index, self.rsd_col] = xy_rsd.astype(float)

        # owner_rsd_id value for missing_values
        o = (
            df.reset_index()
            .set_index(["simtime", "missing_value", "x", "y"])
            .sort_index()  # ensure valid values if present are at top
            .loc[:, ["owner_rsd_id"]]  # only remove nan from owner_rsd_id
            .groupby(["simtime"])  # only propagate value within one time step
            .fillna(
                method="ffill"
            )  # forward fill (propagate) valid value to all cells at current time
            .droplevel("missing_value")
        )
        df.loc[o.index, "owner_rsd_id"] = o

        return df


class DeleteMissingImputation(MissingValueImputationStrategy):
    def __call__(
        self, df: pd.DataFrame, data_column, *args: Any, **kwds: Any
    ) -> pd.DataFrame:
        mask = ~df[data_column].isna()
        return df[mask].copy()
