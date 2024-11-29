from __future__ import annotations

import os
import sys
from abc import ABC
from typing import Any, List, Protocol, TextIO, runtime_checkable

import numpy as np
import pandas as pd

from crownetutils.utils.logging import logger, logging, timing
from crownetutils.utils.misc import Timer


class ImputationIncidentLogger:
    def __init__(self, writer: TextIO) -> None:
        self.writer: TextIO = writer

    def log(self, s: str, end="\n") -> ImputationIncidentLogger:
        self.writer.write(s)
        self.writer.write(end)
        return s


class NullImputationIncidentLogger(ImputationIncidentLogger):
    def __init__(
        self,
    ) -> None:
        super().__init__(os.devnull)

    def log(self, s: str, end="\n") -> ImputationIncidentLogger:
        return f"{s}{end}"


class MissingValueImputationStrategy(ABC):
    """Imputation strategy to fill or remove missing values from a frame. Note that in case of removing the whole
    row will be removed.

    The imputation will sort the provided frame by index [simtime, x, y]
    """

    def __init__(self) -> None:
        self._delay_sort = False
        self.incident_log: ImputationIncidentLogger = NullImputationIncidentLogger()
        self.csv_id = None

    def with_csv_id(self, id):
        self.csv_id = id
        return self

    def set_incident_logger(self, log: ImputationIncidentLogger):
        self.incident_log = log
        return self

    def sort_if_needed(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._delay_sort:
            df = df.sort_index()
        return df

    def delay_sort(self) -> MissingValueImputationStrategy:
        self._delay_sort = True
        return self

    def name(self) -> str:
        ...

    def apply(self, df: pd.DataFrame, *args: Any, **kwds: Any) -> pd.DataFrame:
        """Apply imputation to frame. Returned frame is index sorted."""
        with Timer(self.name()):
            return self._apply(df, *args, **kwds)

    def _apply(self, df: pd.DataFrame, *args: Any, **kwds: Any) -> pd.DataFrame:
        ...


class ImputationStream(MissingValueImputationStrategy):
    """Apply multiple imputation functions based on the append order."""

    def __init__(self) -> None:
        self.imputations: List[MissingValueImputationStrategy] = []

    def append(self, i: MissingValueImputationStrategy) -> ImputationStream:
        self.imputations.append(i)
        return self

    def name(self) -> str:
        r = ", ".join([i.name() for i in self.imputations])
        return f"Imputation stream:[{r}]"

    def apply(self, df: pd.DataFrame, *args: Any, **kwds: Any) -> pd.DataFrame:
        with Timer(name="", label="ImputationStream") as timer:
            csv_id = self.csv_id
            for i, imp_func in enumerate(self.imputations):
                # do not sort if not necessary for imputation. Will be done
                # at the end if needed.
                df = imp_func.delay_sort().with_csv_id(csv_id)._apply(df, *args, **kwds)
                timer.round(f"{i}/{len(self.imputations)}: {imp_func.name()}")

            # ensure that stream of imputations sorted the provided data
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()

        return df


class DeleteMissingArbitraryGlobalValueForImagined(MissingValueImputationStrategy):
    """Delete values where node does not have any measurements (i.e missing values) and
    set an arbitrary  global value for imagined values (i.e node 'sees' something in a
    cell but there is no global value)
    """

    def __init__(
        self,
        glb_fill_value: float = 0.0,
        glb_prefix: str = "glb_",
        data_column: str = "count",
    ) -> None:
        super().__init__()
        self.glb_prefix = glb_prefix
        self.data_column = data_column
        self.glb_fill_value = glb_fill_value

    def name(self) -> str:
        return "DeleteMissingValuesKeepErrorToGlobal"

    def _apply(self, df: pd.DataFrame, *args: Any, **kwds: Any) -> pd.DataFrame:
        glb_data_column = f"{self.glb_prefix}{self.data_column}"
        df[glb_data_column] = df[glb_data_column].fillna(self.glb_fill_value)
        df = df.dropna()
        return self.sort_if_needed(df)


class ArbitraryValueImputation(MissingValueImputationStrategy):
    """Replace missing values on the measurement and ground truth side
    with a predefined fixed value.
    The imputation will sort the provided frame by index [simtime, x, y]

    Example use case:
    For DPM maps which count the number of pedestrians using beacon communication,
    and we do not  'hear' (i.e. receive data) from a cell, we  can assume that there
    is no one in that cell, and thus we replace NAN values with zero.
    Similarly if a node has measurements for a cell, but the ground truth does not
    have any value for that cell we can replace the ground truth NAN-value with zero
    """

    def __init__(
        self,
        fill_value=0.0,
        glb_prefix: str = "glb_",
        data_column: str = "count",
        glb_fill_value=0.0,
    ) -> None:
        super().__init__()
        self.fill_value = fill_value
        self.glb_prefix = glb_prefix
        self.data_column = data_column
        self.glb_fill_value = glb_fill_value

    def name(self) -> str:
        return "ArbitraryValueImputation"

    def _apply(self, df: pd.DataFrame, *args: Any, **kwds: Any) -> pd.DataFrame:
        glb_data_column = f"{self.glb_prefix}{self.data_column}"
        df[self.data_column] = df[self.data_column].fillna(self.fill_value)
        df[glb_data_column] = df[glb_data_column].fillna(self.glb_fill_value)

        return self.sort_if_needed(df)


class FullRsdImputation(MissingValueImputationStrategy):
    """Replace missing RSD ID's
    of measurements with the RSD that is the closest to it (euclidean distance).

    The imputation will sort the provided frame by index [simtime, x, y]
    """

    def __init__(
        self,
        rsd_origin_position: pd.DataFrame,
        rsd_col="rsd_id",
    ) -> None:
        super().__init__()
        self.rsd_col = rsd_col
        self.rsd_distance = rsd_origin_position.rename(
            columns={"x": "enb_x", "y": "enb_y"}
        )  # DataFrame[x, y, rsd_id]

    def name(self) -> str:
        return "FullRsdImputation"

    def _apply(self, df: pd.DataFrame, *args: Any, **kwds: Any) -> pd.DataFrame:
        # find *unique* cells location (x, y) with nan valued rsd
        # for distance calculation for nearest rsd_origin (i.e. base station)

        nan_cols = df.isna().any(axis=0).to_dict()
        if nan_cols[self.rsd_col]:
            # found nan values in rsd_col,
            # apply imputation based on smallest distance to base station (i.e. RSD provider)

            null_index = df[df[self.rsd_col].isna()].index
            xy = (
                null_index.droplevel("simtime")
                .unique()
                .to_frame()
                .reset_index(drop=True)
            )
            # create cross product of missing measurements with enb_positions
            # and calculate the 2-norm for each row (i.e. distance cell to base station)
            xy = xy.join(self.rsd_distance, how="cross")
            xy["dist"] = np.linalg.norm(
                xy.loc[:, ["enb_x", "enb_y"]].values - xy.loc[:, ["x", "y"]].values,
                axis=1,
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
            # performance: join with dummy column faster than resetting index, merging and apply new index
            null_df = pd.DataFrame(0, columns=["dummy"], index=null_index)
            xy_rsd = (
                null_df.join(xy_rsd.set_index(["x", "y"]), on=["x", "y"], how="left")
                .drop(columns="dummy")
                .sort_index()
            )

            # performance: replacing nan with indexed series faster than using index.__setitem__ (i.e. df.loc[indexA, [col]] = seriesA)
            df[self.rsd_col] = df[self.rsd_col].fillna(xy_rsd["rsd_id"]).astype(int)

        if nan_cols["owner_rsd_id"]:
            # found nan values in owner_rsd_id,
            # apply imputation based on owner_rsd_id value present for the same time,
            # as the node is always in the same rsd for the same time

            # performance: As only one owner_rsd_id value per time step exist we can
            #               use the first occurrence for each time and repeat that for
            #               each row in that time. Faster than groupby/fillna as this
            #               requires sorting. Requires  that frame is sorted by simtime
            #
            df = (
                df.sort_index()
            )  # index: [time, x, y] needed to ensure times are sorted
            owner_rsd_for_time = df.groupby("simtime")["owner_rsd_id"].first()
            if owner_rsd_for_time.isna().any():
                times_without_data = owner_rsd_for_time[
                    owner_rsd_for_time.isna()
                ].index.to_list()
                num_data = df.loc[times_without_data].shape[0]
                logger.warning(
                    self.incident_log.log(
                        f"node {self.csv_id}: found {len(times_without_data)} time(s) without any data. Remove rows {num_data}/{df.shape[0]} ({num_data/df.shape[0]*100:0.4f}%) "
                    )
                )
                logger.warning(
                    self.incident_log.log(
                        f"node {self.csv_id}: violating time idencies: {times_without_data}"
                    )
                )
                df = df.drop(level="simtime", labels=times_without_data)
                owner_rsd_for_time = owner_rsd_for_time.dropna().values
            else:
                owner_rsd_for_time = owner_rsd_for_time.values
            time_count = df.groupby("simtime")["missing_value"].count()
            owner_rsd = np.repeat(owner_rsd_for_time, repeats=time_count)
            df["owner_rsd_id"] = owner_rsd

        return df  # is already sorted as it was needed during imputation


class DeleteMissingImputation(MissingValueImputationStrategy):
    def __init__(self, data_column: str = "count") -> None:
        super().__init__()
        self.data_column = data_column

    def name(self) -> str:
        return "DeleteMissingImputation"

    def _apply(self, df: pd.DataFrame, *args: Any, **kwds: Any) -> pd.DataFrame:
        mask = ~df[self.data_column].isna()
        return self.sort_if_needed(df[mask].copy())


class OwnerPositionImputation(MissingValueImputationStrategy):
    def __init__(self, x_owner: str = "x_owner", y_owner: str = "y_owner") -> None:
        super().__init__()
        self.x_owner = x_owner
        self.y_owner = y_owner

    def name(self) -> str:
        return "OwnerPositionImputation"

    def _apply(self, df: pd.DataFrame, *args: Any, **kwds: Any) -> pd.ddDataFrame:
        nan_cols = df.isna().any(axis=0).to_dict()
        if nan_cols[self.x_owner] or nan_cols[self.y_owner]:
            # sort if needed
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()  # needed for imputation
            try:
                col_with_no_nans = [k for k, v in nan_cols.items() if v is False][0]
            except IndexError as e:
                raise ValueError(
                    f"at least one row must not have any nan values. got {nan_cols}"
                )
            xy_owner = df.groupby(["simtime"])[["x_owner", "y_owner"]].first()
            if xy_owner.isna().any().any():
                times_without_data = xy_owner[
                    xy_owner.isna().any(axis=1)
                ].index.to_list()
                num_data = df.loc[times_without_data].shape[0]
                logger.warning(
                    self.incident_log.log(
                        f"node {self.csv_id}: found {len(times_without_data)} time(s) without any data. Remove rows {num_data}/{df.shape[0]} ({num_data/df.shape[0]*100:0.4f}%) "
                    )
                )
                logger.warning(
                    self.incident_log.log(
                        f"node {self.csv_id}: violating time idencies: {times_without_data}"
                    )
                )
                df = df.drop(level="simtime", labels=times_without_data)
                xy_owner = xy_owner.dropna().values
            else:
                xy_owner = xy_owner.values

            time_count = df.groupby(["simtime"])[col_with_no_nans].count().values

        if nan_cols[self.x_owner]:
            df["x_owner"] = np.repeat(xy_owner[:, 0], time_count)

        if nan_cols[self.y_owner]:
            df["y_owner"] = np.repeat(xy_owner[:, 1], time_count)

        return df  # already sorted
