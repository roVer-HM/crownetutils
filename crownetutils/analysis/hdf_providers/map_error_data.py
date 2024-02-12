from __future__ import annotations

import os
import sys
import timeit as it
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd

from crownetutils.analysis import RsdAssociationProvider
from crownetutils.analysis.dpmm.hdf.dpmm_count_provider import DpmmCount, DpmmCountKey
from crownetutils.analysis.dpmm.hdf.dpmm_global_positon_provider import (
    DpmmGlobalPosition,
)
from crownetutils.analysis.dpmm.hdf.dpmm_provider import DpmmProvider
from crownetutils.analysis.hdf.provider import BaseHdfProvider
from crownetutils.analysis.hdf_providers.helper import ExpectedHdfContent
from crownetutils.utils.dataframe import FrameConsumer, partial_index_match
from crownetutils.utils.logging import LogWriter, logger, timing


def percentile(n):
    def _percentil(x):
        try:
            return np.percentile(x, float(n) * 100.0)
        except:
            return np.nan

    _percentil.__name__ = f"p_{n*100:2.0f}"
    return _percentil


def check_measure_data_for_nans_and_assert(cell_base):
    is_na_col = cell_base.isna().any(axis=0)
    if any(is_na_col):
        raise ValueError(f"Found coulmns with nan values: \n {is_na_col}")

    assert_obs_v_agents = (
        cell_base["num_observations"] <= cell_base["num_active_agents"]
    )
    if not assert_obs_v_agents.all():
        count = len(assert_obs_v_agents[~assert_obs_v_agents])
        if count < 10:
            raise ValueError(
                f"found {count} rows where num_observations <= num_active_agents which should not be:\n{cell_base[~assert_obs_v_agents]}"
            )
        else:
            raise ValueError(
                f"found {count} rows where num_observations <= num_active_agents which should not be"
            )


def create_time_chunk_intervall(
    nrows, average_chunk_size_bytes, time_interval, bytes_per_row
) -> float | None:
    """create complete 'simtime' chunks based on average_chunk_size in bytes. This method assumes roughly equal
    data over time. This is obviosly woring but a good gess to keep chunksizing easy.

    Args:
        nrows (_type_): number of rows in dcd_map hdf table
        average_chunk_size_bytes (_type_): number of bytes each size shoud have (ballpark number)
        time_interval (_type_): total interval of range of rows in dcd_map (min/max values inclusive)
        bytes_per_row (_type_): byte count estimate for one row

    Returns:
        float|None: size of interval or null if only one chunk is needed
    """
    total_size = nrows * bytes_per_row
    if total_size < average_chunk_size_bytes:
        # no chunking pull all data at once
        return None

    num_chunks = np.ceil(total_size / average_chunk_size_bytes)
    # assume similar aoumnt of data over all times. This is obviouly wrong everything else is simulation specifc.
    total_time = time_interval[1] - time_interval[0]
    time_chunk = np.ceil(total_time / num_chunks)

    return time_chunk


def remove_missing_values(df: pd.DataFrame):
    mask = df["missing_value"].values
    return df[~mask].copy()


@dataclass
class CellEntropyValueErrorBuilder:
    index_slice: slice | Tuple(slice) = field(default=slice(None))
    xy_slice: Tuple(slice) | pd.MultiIndex = field(default=(slice(None), slice(None)))
    fc: FrameConsumer = field(default=FrameConsumer.EMPTY)
    columns: slice | List[str] = field(default=slice(None))


class CellEntropyValueError:
    """compare with CellCountError
    This method handles arbitrary measurements with removed missing values.
    In contrast to CellCountError this class handels arbitrary values which
    are not additive like the pedestrian count data (see CellCountError).

    This data does not use a single value imputation such as zero for unknown
    cells as the CellContError uses. Furthermore, this class does not contain
    implicit measurements.
    """

    tsc_id_idx_name = "ID"
    tsc_time_idx_name = "simtime"
    tsc_x_idx_name = "x"
    tsc_y_idx_name = "y"

    g_error_cell_all = "cell_entropy_measure"
    g_error_cell_rsd_local = "cell_entropy_measure_by_rsd_local"
    g_error_cell_rsd_all = "cell_entropy_measure_by_rsd"

    groups = [g_error_cell_all, g_error_cell_rsd_all, g_error_cell_rsd_local]

    def __init__(self, hdf_path) -> None:
        self.hdf_path = hdf_path

        self._hdf_cell_entropy_measure: BaseHdfProvider = None
        self._hdf_cell_entropy_measure_rsd: BaseHdfProvider = None
        self._hdf_cell_entropy_measure_rsd_local: BaseHdfProvider = None

    @staticmethod
    def default_hdf_name(map_type=""):
        _default_hdf_name = "entropy_map_cell_error"
        if map_type != "":
            return f"{map_type}_{_default_hdf_name}.h5"
        else:
            return f"{_default_hdf_name}.h5"

    @property
    def hdf_cell_entropy_measure(self) -> BaseHdfProvider:
        if self._hdf_cell_entropy_measure is None:
            self._hdf_cell_entropy_measure = BaseHdfProvider(
                self.hdf_path, group=self.g_error_cell_all
            )
        return self._hdf_cell_entropy_measure

    @property
    def hdf_cell_entropy_measure_rsd(self) -> BaseHdfProvider:
        if self._hdf_cell_entropy_measure_rsd is None:
            self._hdf_cell_entropy_measure_rsd = BaseHdfProvider(
                self.hdf_path, group=self.g_error_cell_rsd_all
            )
        return self._hdf_cell_entropy_measure_rsd

    @property
    def hdf_cell_entropy_measure_rsd_local(self) -> BaseHdfProvider:
        if self._hdf_cell_entropy_measure_rsd_local is None:
            self._hdf_cell_entropy_measure_rsd_local = BaseHdfProvider(
                self.hdf_path, group=self.g_error_cell_rsd_local
            )
        return self._hdf_cell_entropy_measure_rsd_local

    @classmethod
    def get_or_create(
        cls,
        hdf_path,
        count_p: DpmmCount,
        builder: CellEntropyValueErrorBuilder | None = None,
        with_rsd: bool = True,
    ) -> CellEntropyValueError:
        if builder is None:
            builder = CellCountErrorBuilder()

        obj: CellEntropyValueError = cls(hdf_path)
        if os.path.exists(hdf_path):
            time_interval = count_p.get_attribute("time_interval")
            content = ExpectedHdfContent().add_groups(
                groups=obj.groups, time_interval=time_interval
            )
            is_content_as_expected, diff = content.test_hdf(BaseHdfProvider(hdf_path))

            if is_content_as_expected:
                # hdf file exists and contains all data with same parameters
                logger.info(
                    f"found existing {cls.__name__} file with match parameter setup. No build required."
                )
            else:
                logger.info("found difference in existing hdf file.")
                diff.write_diff(writer=LogWriter.info(), header=f"{cls.__name__}:")
                logger.info(
                    f"Found {cls.__name__} file with mismatching groups. Delete file and recreate."
                )
                os.remove(hdf_path)
                logger.info(f"Create hdf {hdf_path}")
                obj._create_hdf(count_p, builder, with_rsd)
        else:
            logger.info("no existing hdf file found. Build hdf...")
            obj._create_hdf(count_p, builder, with_rsd)

        return obj

    def load_glb(
        self, count_p: DpmmCount, time_slice: slice, builder: CellCountErrorBuilder
    ) -> pd.DataFrame:
        _i = pd.IndexSlice
        # ground truth of nodes at each time where at least one agent was present at some earlier time
        if isinstance(builder.xy_slice, pd.MultiIndex):
            glb = count_p[_i[time_slice, :, :, 0], _i["count"]]  # only ground truth
            glb = partial_index_match(glb, builder.xy_slice)
        else:
            glb = count_p[
                _i[time_slice, builder.xy_slice[0], builder.xy_slice[1], 0], _i["count"]
            ]  # only ground truth
        return glb

    def load_data(
        self,
        count_p: DpmmCount,
        time_slice: slice,
        builder: CellEntropyValueErrorBuilder,
    ) -> pd.DataFrame:
        # todo: use code from DpmMap.cell_value_measure. move metric calc to extra

        # all (time, x, y, id) based count, err, squerr cell values
        # without ground truth (see slice last slice `1:`)
        # The measurements are summed over all nodes (this will drop the id index )
        _i = pd.IndexSlice
        cols = _i["count", "err", "sqerr", "rsd_id", "owner_rsd_id", "missing_value"]
        if isinstance(builder.xy_slice, pd.MultiIndex):
            nodes: pd.DataFrame = count_p[
                _i[time_slice, :, :, 1:], cols
            ]  # all but ground truth
            nodes = remove_missing_values(nodes)
            nodes = partial_index_match(nodes, builder.xy_slice)
        else:
            nodes: pd.DataFrame = count_p[
                _i[time_slice, builder.xy_slice[0], builder.xy_slice[1], 1:], cols
            ]  # all but ground truth
            nodes = remove_missing_values(nodes)

        return nodes

    def time_chunked_stream(
        self,
        count_p: DpmmCount,
        time_chunk_size,
        time_interval,
        builder: CellEntropyValueErrorBuilder,
    ):
        if time_chunk_size is None:
            time_slice = (
                slice()
            )  # empty slice, no chunking load all data (can cause OOM!)
            glb = self.load_glb(count_p, start, end, builder)
            data = self.load_data(start, end, builder)
            yield glb, data
        else:
            start = time_interval[0]
            end = start + time_chunk_size
            while start < time_interval[1]:
                ts = it.default_timer()
                time_slice = slice(start, end)
                glb = self.load_glb(count_p, time_slice, builder)
                data = self.load_data(count_p, time_slice, builder)
                s1 = sys.getsizeof(data) / 1e6
                s2 = sys.getsizeof(glb) / 1e6
                logger.info(
                    f"simtime range: ({start}, {end}] retrieving {data.shape[0]:,} rows took {it.default_timer() - ts:2.2f} seconds for {(s1 + s2):,.2f}MB (sizeof data+glb)"
                )
                start = end
                end = start + time_chunk_size
                yield glb, data

    def _aggregate_time_chunk(
        self,
        glb: pd.DataFrame,
        nodes: pd.DataFrame,
        builder: CellEntropyValueErrorBuilder,
    ) -> pd.DataFrame:
        glb = glb.droplevel("ID")
        glb.columns = ["glb_count"]

        # Number of nodes which are active at a given time, irrespective location (i.e. cells). This is
        # used by CellCountError due to implicit zero errors. Not used here, see CellCountError why this
        # was used there. It still will be used for assertion test num_observerations <= num_active_agents.
        num_active_agents = (
            nodes.index.copy()
            .droplevel(["x", "y"])
            .unique()
            .to_frame()
            .reset_index(drop=True)
            .groupby("simtime")
            .count()
            .set_axis(["num_active_agents"], axis=1)
        )
        # Number of observations is total number of agents which contributed a cell based measures for the
        # triplet (time, x, y). This would be the eqal to directly use the 'mean' aggregated above instead
        # of 'sum'. I do this separately to have the same structure as as CellCountError.
        num_observations = nodes.groupby(["simtime", "x", "y"])["count"].count()
        num_observations.name = "num_observations"

        nodes["abserr"] = np.abs(nodes["err"])
        # metric III 1/N sum^N_i[ 1/M sum^M_j (Y_ij - Y^_i)^2 ]
        # create sum: sum^M_j[*]  with [*] is nodes["sqerr"] = (Y_ij - Y^_i)^2 and nodes["count"] = (Y_ij)
        cell_base: pd.DateFrame = nodes.groupby(
            level=[self.tsc_time_idx_name, self.tsc_x_idx_name, self.tsc_y_idx_name]
        ).agg(
            ["sum"]
        )  # [time, x, y](...data-columns...)

        cell_base.columns = [f"{a}_{b}" for a, b in cell_base.columns]

        cell_base: pd.DataFrame = cell_base.join(num_active_agents, on="simtime")
        cell_base: pd.DataFrame = cell_base.join(
            num_observations, on=["simtime", "x", "y"]
        )
        # no nans expeceted due to used imputation.
        cell_base: pd.DataFrame = cell_base.join(glb, on=["simtime", "x", "y"])

        check_measure_data_for_nans_and_assert(cell_base)

        # divide by total number of nodes at each time to create mean measruements
        cell_base["cell_mean_count_est"] = (
            cell_base["count_sum"] / cell_base["num_observations"]
        )  # 1/M sum^M_j (Y_ij) -> neee for metric II
        cell_base["cell_mean_est_sqerr"] = np.power(
            cell_base["cell_mean_count_est"] - cell_base["glb_count"], 2
        )  # (1/M sum^M_j (Y_ij) - Y^_i)^2 -> needed for metric II

        cell_base["cell_mse"] = (
            cell_base["sqerr_sum"] / cell_base["num_observations"]
        )  # 1/M sum^M_j (Y_ij - Y^_i)^2 -> needed for metric III
        cell_base["cell_mean_err"] = (
            cell_base["err_sum"] / cell_base["num_observations"]
        )  # 1/M sum^M_j (Y_ij - Y^_i) -> optional
        cell_base["cell_mean_abserr"] = (
            cell_base["abserr_sum"] / cell_base["num_observations"]
        )  # 1/M sum^M_j |Y_ij - Y^_i| -> optional

        return cell_base

    def _aggregate_time_chunk_rsd(
        self,
        hdf: BaseHdfProvider,
        mask,
        glb: pd.DataFrame,
        data: pd.DataFrame,
        rsd: int,
        builder: CellEntropyValueErrorBuilder,
    ):
        # select subset of cell that belong to provided rsd.
        idx_ = (
            data[mask].index.droplevel("ID").unique().to_frame().reset_index(drop=True)
        )
        idx_["ID"] = 0
        idx_ = pd.MultiIndex.from_frame(idx_).sort_values()
        in_glb = glb.index.intersection(idx_, sort=True)
        _glb = glb.loc[in_glb].copy()

        _data = data.loc[mask, ["err", "count", "sqerr"]].copy()
        df_aggregate = self._aggregate_time_chunk(_glb, _data, builder)
        df_aggregate[DpmmCountKey.RSD_ID] = rsd
        df_aggregate = (
            df_aggregate.reset_index()
            .set_index([DpmmCountKey.RSD_ID, "simtime", "x", "y"])
            .sort_index()
        )
        # todo builder.fc not planned to work on chunks!!!!
        df_aggregate = df_aggregate.loc[:, builder.columns]
        hdf.write_frame(group=hdf.group, frame=df_aggregate)

    @timing
    def _create_hdf(
        self, count_p: DpmmCount, builder: CellEntropyValueErrorBuilder, with_rsd: bool
    ):
        count_p.print_info(fd=LogWriter.info2())
        time_interval = count_p.get_attribute("time_interval")
        time_chunk_size = create_time_chunk_intervall(
            nrows=count_p.get_attribute("NROWS"),
            average_chunk_size_bytes=2e9,  # 2GB chunks
            time_interval=time_interval,
            bytes_per_row=48,  # 4xfloat64 and 4xint32
        )

        rsd_list = []
        if with_rsd:
            try:
                rsd_list = count_p.get_rsd_ids()
            except Exception as e:
                raise ValueError(
                    f"CellEntropyValueError schould be build with RSD support but retrieving the rsd list faield with: {e}"
                )

        for glb, data in self.time_chunked_stream(
            count_p, time_chunk_size, time_interval, builder
        ):
            # all RSD combined
            aggregate_all = self._aggregate_time_chunk(
                glb, data[["err", "sqerr", "count"]].copy(), builder
            )
            aggregate_all = builder.fc(
                aggregate_all.loc[builder.index_slice, builder.columns]
            )
            self.hdf_cell_entropy_measure.write_frame(
                group=self.hdf_cell_entropy_measure.group, frame=aggregate_all
            )
            del aggregate_all
            aggregate_all = None

            for rsd in rsd_list:
                # rsd_all
                mask = data["rsd_id"] == rsd
                self._aggregate_time_chunk_rsd(
                    hdf=self.hdf_cell_entropy_measure_rsd,
                    mask=mask,
                    glb=glb,
                    data=data,
                    rsd=rsd,
                    builder=builder,
                )

                # rsd_local
                mask_local = mask & (data["rsd_id"] == data["owner_rsd_id"])
                self._aggregate_time_chunk_rsd(
                    hdf=self.hdf_cell_entropy_measure_rsd_local,
                    mask=mask_local,
                    glb=glb,
                    data=data,
                    rsd=rsd,
                    builder=builder,
                )

        # write time_interval attribute at end of processing. If this is missing the processing was not completed.
        for _hdf in [
            self.hdf_cell_entropy_measure,
            self.hdf_cell_entropy_measure_rsd,
            self.hdf_cell_entropy_measure_rsd_local,
        ]:
            _hdf.set_attribute(
                attr_key="time_interval", value=time_interval, group=_hdf.group
            )

        self.hdf_cell_entropy_measure.repack_hdf(keep_old_file=False)


@dataclass
class CellCountErrorBuilder:
    index_slice: slice | Tuple(slice) = field(default=slice(None))
    xy_slice: Tuple(slice) | pd.MultiIndex = field(default=(slice(None), slice(None)))
    fc: FrameConsumer = field(default=FrameConsumer.EMPTY)
    columns: slice | List[str] = field(default=slice(None))
    remove_missing_values: bool = field(default=False)


class CellCountError:
    """create cell based error measures over time to indicate **positional correctness**

    remove_missing_values: If false we use the values introduced by the
    imputation function during creation of the count map. Missing values are
    marked in the column 'missing_value'.

    count_p contains count, err, and sqerr values at the (time, id, x, y)
    level.  In other words the table contains these values for each node
    (id) for a given cell (x, y) for a given time (time).  The table count_p
    only contains communicated cells as well as over and underestimation
    errors.

    Assume cell x_i was occupied until t=10 . Then this cell is reported for
    each time step and node. Either with err = 0 if the node sees the
    occupant or err = -1 for nodes where the occupant is not seen and err >=
    1 in the case some nodes see more than one node. Note that negative
    values (underestimation) is bound by the real number of occupants in the
    cell. On the other hand overestimation is not bound!

    Assume now t > 100 and x_i is not occupied anymore and any TTL is
    reached, thus no node should have any values for the cell x_i.  Assume
    now that from a total of N=10 nodes one node is faulty and has a count of 1
    for cell x_i. This count will be part of the count_p table and marked
    with an error count of 1. The correct value of count=0 for all other
    nodes is not stored in count_p but are implied. Thus to calculate the
    mean absolute error of cell x_i is:

            mean_abs_err = (|1| + 9*|0|)/(N=10) = 0.1

    For this reason a simple count_p.groupby([...]).mean() will not work
    because the used number of observations will be wrong because the
    implied zero-error values are not saved in count_p. This function will
    therefore calculate the mean errors manually by utilizing the total
    number of nodes N for each time `t`. The numerator will be the same
    because only zero-counts / zero-erros are implied. Any non-zero count or
    error will be saved explicitly in count_p.

        N:= set of cells (x, y) with index i
        M:= set of agents/measuring agents (ID) with index j
        Y^_i := (Y-Hat) ground truth for cell i. This is identical for each agents thus
                Y^_ij - Y^_i(j+1) for all i and j.

    """

    tsc_id_idx_name = "ID"
    tsc_time_idx_name = "simtime"
    tsc_x_idx_name = "x"
    tsc_y_idx_name = "y"

    g_error_cell_all = "cell_measure"
    g_error_cell_rsd_local = "local_cell_measure_by_rsd"
    g_error_cell_rsd_all = "cell_measure_by_rsd"

    groups = [g_error_cell_all, g_error_cell_rsd_all, g_error_cell_rsd_local]

    def __init__(self, hdf_path) -> None:
        self.hdf_path = hdf_path

        self._hdf_cell_measure: BaseHdfProvider = None
        self._hdf_cell_measure_rsd: BaseHdfProvider = None
        self._hdf_cell_measure_rsd_local: BaseHdfProvider = None

    @staticmethod
    def default_hdf_name(map_type=""):
        _default_hdf_name = "map_cell_error"
        if map_type != "":
            return f"{map_type}_{_default_hdf_name}.h5"
        else:
            return f"{_default_hdf_name}.h5"

    @classmethod
    def get_or_create(
        cls,
        hdf_path,
        count_p: DpmmCount,
        builder: CellCountErrorBuilder | None = None,
        with_rsd: bool = True,
    ) -> CellCountError:
        obj: CellCountError = cls(hdf_path)

        if os.path.exists(hdf_path):
            time_interval = count_p.get_attribute("time_interval")
            content = ExpectedHdfContent().add_groups(
                groups=obj.groups, time_interval=time_interval
            )
            is_content_as_expected, diff = content.test_hdf(BaseHdfProvider(hdf_path))

            if is_content_as_expected:
                # hdf file exists and contains all data with same parameters
                logger.info(
                    f"found existing {cls.__name__} file with match parameter setup. No build required."
                )
            else:
                logger.info("found difference in existing hdf file.")
                diff.write_diff(writer=LogWriter.info(), header=f"{cls.__name__}:")
                logger.info(
                    f"Found {cls.__name__} file with mismatching groups. Delete file and recreate."
                )
                os.remove(hdf_path)
                logger.info(f"Create hdf {hdf_path}")
                obj._create_hdf(count_p, builder, with_rsd)
        else:
            logger.info("no existing hdf file found. Build hdf...")
            obj._create_hdf(count_p, builder, with_rsd)

    @property
    def hdf_cell_measure(self) -> BaseHdfProvider:
        if self._hdf_cell_measure is None:
            self._hdf_cell_measure = BaseHdfProvider(
                self.hdf_path, group=self.g_error_cell_all
            )
        return self._hdf_cell_measure

    @property
    def hdf_cell_measure_rsd(self) -> BaseHdfProvider:
        if self._hdf_cell_measure_rsd is None:
            self._hdf_cell_measure_rsd = BaseHdfProvider(
                self.hdf_path, group=self.g_error_cell_rsd_all
            )
        return self._hdf_cell_measure_rsd

    @property
    def hdf_cell_measure_rsd_local(self) -> BaseHdfProvider:
        if self._hdf_cell_measure_rsd_local is None:
            self._hdf_cell_measure_rsd_local = BaseHdfProvider(
                self.hdf_path, group=self.g_error_cell_rsd_local
            )
        return self._hdf_cell_measure_rsd_local

    def load_glb(
        self, count_p: DpmmCount, time_slice: slice, builder: CellCountErrorBuilder
    ) -> pd.DataFrame:
        _i = pd.IndexSlice
        if isinstance(builder.xy_slice, pd.MultiIndex):
            glb = count_p[_i[time_slice, :, :, 0], _i["count"]]  # only ground truth
            glb = partial_index_match(glb, builder.xy_slice)
        else:
            glb = count_p[
                _i[time_slice, builder.xy_slice[0], builder.xy_slice[1], 0], _i["count"]
            ]  # only ground truth
        return glb

    def load_data(
        self, count_p: DpmmCount, time_slice: slice, builder: CellCountErrorBuilder
    ) -> pd.DataFrame:
        _i = pd.IndexSlice
        # all (time, x, y, id) based count, err, squerr cell values
        # without ground truth (see slice last slice `1:`)
        # The measurements are summed over all nodes (this will drop the id index )
        if builder.remove_missing_values:
            cols = _i[
                "count", "err", "sqerr", "rsd_id", "owner_rsd_id", "missing_value"
            ]
        else:
            cols = _i["count", "err", "sqerr", "rsd_id", "owner_rsd_id"]

        if isinstance(builder.xy_slice, pd.MultiIndex):
            nodes: pd.DataFrame = count_p[
                _i[time_slice, :, :, 1:], cols
            ]  # all but ground truth
            if builder.remove_missing_values:
                nodes = remove_missing_values(nodes)
            nodes = nodes.drop(columns=["missing_value"])
            nodes = partial_index_match(nodes, builder.xy_slice)
        else:
            nodes: pd.DataFrame = count_p[
                _i[time_slice, builder.xy_slice[0], builder.xy_slice[1], 1:], cols
            ]  # all but ground truth
            if builder.remove_missing_values:
                nodes = remove_missing_values(nodes)

        return nodes

    def time_chunked_stream(
        self,
        count_p: DpmmCount,
        time_chunk_size,
        time_interval,
        builder: CellCountErrorBuilder,
    ):
        if time_chunk_size is None:
            time_slice = (
                slice()
            )  # empty slice, no chunking load all data (can cause OOM!)
            glb = self.load_glb(count_p, start, end, builder)
            data = self.load_data(start, end, builder)
            yield glb, data
        else:
            start = time_interval[0]
            end = start + time_chunk_size
            while start < time_interval[1]:
                ts = it.default_timer()
                time_slice = slice(start, end)
                glb = self.load_glb(count_p, time_slice, builder)
                data = self.load_data(count_p, time_slice, builder)
                s1 = sys.getsizeof(data) / 1e6
                s2 = sys.getsizeof(glb) / 1e6
                logger.info(
                    f"simtime range: ({start}, {end}] retriving {data.shape[0]:,} rows took {it.default_timer() - ts:2.2f} seconds for {(s1 + s2):,.2f}MB (sizeoff data+glb)"
                )
                start = end
                end = start + time_chunk_size
                yield glb, data

    def _aggregate_time_chunk(
        self, glb: pd.DataFrame, nodes: pd.DataFrame, builder: CellCountErrorBuilder
    ) -> pd.DataFrame:
        glb = glb.droplevel("ID")
        glb.columns = ["glb_count"]

        # 12.01.2023 S. Schuhbaeck
        # num_Agents represents the numeber of nodes that are able to measure the value of a
        # cell. Due to the missing implicit zero-error-values the number of observerations are not equal to
        # to that number. In a previous version that number was calcalted by the toatl number
        # of agents in the simualtion, based on the oracle (ground truth) values. This however,
        # is not completrly correct as nodes may spawn an t_0 but only start the application at t_0 + d_t.
        # The ground truth oracel sees the nodes at t_0 but real measureements only start at t_0 + d_t.
        # If the some nodes never communicate due to  some  other config the values would be considerabel
        # to high.
        #
        # I keep the num_Agents for now.
        glb_map_sum = glb.groupby("simtime").sum()  # [simtime](count) aka. M
        glb_map_sum.columns = ["num_Agents"]
        #
        # Instead of using the ground truth data I retriev the active number of agents from the nodes
        # data. I define a node as actice for time t if the nodes has at least one measurement for that
        # time t. If this measrue exists, the measure must be part of the nodes frame. Furthermore,
        num_active_agents = (
            nodes.index.copy()
            .droplevel(["x", "y"])
            .unique()
            .to_frame()
            .reset_index(drop=True)
            .groupby("simtime")
            .count()
            .set_axis(["num_active_agents"], axis=1)
        )
        # Furthermore, I add the number of observerations per (time, x, y) triblet. This would
        # be equal to the number of active agents, when implicit zero-error-values would not be
        # missing. I use the num_observerations <= num_active_agents as an assertion to ensure
        # correctnets.
        num_observations: pd.Series = nodes.groupby(
            [self.tsc_time_idx_name, self.tsc_x_idx_name, self.tsc_y_idx_name]
        )["count"].count()
        num_observations.name = "num_observations"

        nodes["abserr"] = np.abs(nodes["err"])

        # metric III 1/N sum^N_i[ 1/M sum^M_j (Y_ij - Y^_i)^2 ]
        # create sum: sum^M_j[*]  with [*] is nodes["sqerr"] = (Y_ij - Y^_i)^2 and nodes["count"] = (Y_ij)
        cell_base: pd.DateFrame = nodes.groupby(
            level=[self.tsc_time_idx_name, self.tsc_x_idx_name, self.tsc_y_idx_name]
        ).agg(
            ["sum"]
        )  # [time, x, y](...data-columns...)
        cell_base.columns = [f"{a}_{b}" for a, b in cell_base.columns]
        # join total number of agents (aka. M) with cell based measures. See function description
        cell_base: pd.DataFrame = cell_base.join(glb_map_sum, on="simtime")
        cell_base: pd.DataFrame = cell_base.join(glb, on=["simtime", "x", "y"])
        cell_base: pd.DataFrame = cell_base.join(
            num_observations, on=["simtime", "x", "y"]
        )
        cell_base: pd.DataFrame = cell_base.join(num_active_agents, on="simtime")
        cell_base["glb_count"] = cell_base["glb_count"].fillna(
            value=0
        )  # nan expected add 0
        # cell_base["num_Agents"] = cell_base["num_Agents"].fillna(value=0) # see comment above. should not be nan

        check_measure_data_for_nans_and_assert(cell_base)

        # divide by total number of nodes at each time to create mean measruements
        cell_base["cell_mean_count_est"] = (
            cell_base["count_sum"] / cell_base["num_active_agents"]
        )  # 1/M sum^M_j (Y_ij) -> neee for metric II
        cell_base["cell_mean_est_sqerr"] = np.power(
            cell_base["cell_mean_count_est"] - cell_base["glb_count"], 2
        )  # (1/M sum^M_j (Y_ij) - Y^_i)^2 -> needed for metric II

        cell_base["cell_mse"] = (
            cell_base["sqerr_sum"] / cell_base["num_active_agents"]
        )  # 1/M sum^M_j (Y_ij - Y^_i)^2 -> needed for metric III
        cell_base["cell_mean_err"] = (
            cell_base["err_sum"] / cell_base["num_active_agents"]
        )  # 1/M sum^M_j (Y_ij - Y^_i) -> optional
        cell_base["cell_mean_abserr"] = (
            cell_base["abserr_sum"] / cell_base["num_active_agents"]
        )  # 1/M sum^M_j |Y_ij - Y^_i| -> optional

        return cell_base

    def _aggregate_time_chunk_rsd(
        self,
        hdf: BaseHdfProvider,
        mask,
        glb: pd.DataFrame,
        data: pd.DataFrame,
        rsd: int,
        builder: CellCountErrorBuilder,
    ):
        # select subset of cell that belong to provided rsd.
        idx_ = (
            data[mask].index.droplevel("ID").unique().to_frame().reset_index(drop=True)
        )
        idx_["ID"] = 0
        idx_ = pd.MultiIndex.from_frame(idx_).sort_values()
        in_glb = glb.index.intersection(idx_, sort=True)
        _glb = glb.loc[in_glb].copy()

        _data = data.loc[mask, ["err", "count", "sqerr"]].copy()
        df_aggregate = self._aggregate_time_chunk(_glb, _data, builder)
        df_aggregate[DpmmCountKey.RSD_ID] = rsd
        df_aggregate = (
            df_aggregate.reset_index()
            .set_index([DpmmCountKey.RSD_ID, "simtime", "x", "y"])
            .sort_index()
        )
        # todo builder.fc not planned to work on chunks!!!!
        df_aggregate = df_aggregate.loc[:, builder.columns]
        hdf.write_frame(group=hdf.group, frame=df_aggregate)

    @timing
    def _create_hdf(
        self, count_p: DpmmCount, builder: CellCountErrorBuilder, with_rsd: bool = True
    ):
        count_p.print_info(fd=LogWriter.info2())
        time_interval = count_p.get_attribute("time_interval")
        time_chunk_size = create_time_chunk_intervall(
            nrows=count_p.get_attribute("NROWS"),
            average_chunk_size_bytes=2e9,  # 2GB chunks
            time_interval=time_interval,
            bytes_per_row=48,  # 4xfloat64 and 4xint32
        )

        rsd_list = []
        if with_rsd:
            try:
                rsd_list = count_p.get_rsd_ids()
            except Exception as e:
                raise ValueError(
                    f"CellCountError schould be build with RSD support but retrieving the rsd list faield with: {e}"
                )

        for glb, data in self.time_chunked_stream(
            count_p, time_chunk_size, time_interval, builder
        ):
            # all RSD combined
            aggregate_all = self._aggregate_time_chunk(
                glb, data[["err", "sqerr", "count"]].copy(), builder
            )
            aggregate_all = builder.fc(
                aggregate_all.loc[builder.index_slice, builder.columns]
            )
            self.hdf_cell_measure.write_frame(
                group=self.hdf_cell_measure.group, frame=aggregate_all
            )
            del aggregate_all
            aggregate_all = None

            for rsd in rsd_list:
                # rsd_all
                mask = data["rsd_id"] == rsd
                self._aggregate_time_chunk_rsd(
                    hdf=self.hdf_cell_measure_rsd,
                    mask=mask,
                    glb=glb,
                    data=data,
                    rsd=rsd,
                    builder=builder,
                )

                # rsd_local
                mask_local = mask & (data["rsd_id"] == data["owner_rsd_id"])
                self._aggregate_time_chunk_rsd(
                    hdf=self.hdf_cell_measure_rsd_local,
                    mask=mask_local,
                    glb=glb,
                    data=data,
                    rsd=rsd,
                    builder=builder,
                )

        # write time_interval attribute at end of processing. If this is missing the processing was not completed.
        for _hdf in [
            self.hdf_cell_measure,
            self.hdf_cell_measure_rsd,
            self.hdf_cell_measure_rsd_local,
        ]:
            _hdf.set_attribute(
                attr_key="time_interval", value=time_interval, group=_hdf.group
            )

        self.hdf_cell_measure.repack_hdf(
            keep_old_file=False
        )  # same file only repack once


class MapCountError:
    """create map based error measure over time to indicate **total area count correctness**

    Get map count measure that shows how good the number of agents are
    represented by the density map irrespective of there positions. Meaning
    this error measure only shows if no agents are left out or are 'produced' by
    the density map. There positional information is lost in this measure.

    """

    g_error_count_all = "map_measure"
    g_error_count_rsd_local = "local_map_measure_by_rsd"
    g_error_count_rsd_all = "map_measure_by_rsd"

    groups = [g_error_count_all, g_error_count_rsd_all, g_error_count_rsd_local]

    def __init__(self, hdf_path) -> None:
        self.hdf_path = hdf_path

        self._hdf_map_measure: BaseHdfProvider = None
        self._hdf_map_measure_rsd: BaseHdfProvider = None
        self._hdf_map_measure_rsd_local: BaseHdfProvider = None

    @classmethod
    def default_hdf_name(cls, map_type=""):
        _default_hdf_name = "map_count_error"
        if map_type != "":
            return f"{map_type}_{_default_hdf_name}.h5"
        else:
            return f"{_default_hdf_name}.h5"

    @classmethod
    def get_or_create(
        cls,
        hdf_path,
        map_p: DpmmProvider,
        glb_pos: DpmmGlobalPosition,
        rsd_p: RsdAssociationProvider | None = None,
        with_rsd: bool = True,
    ):
        obj: MapCountError = cls(hdf_path)

        if os.path.exists(hdf_path):
            time_interval = map_p.get_attribute("time_interval")
            content = ExpectedHdfContent().add_groups(
                groups=obj.groups, time_interval=time_interval
            )
            is_content_as_expected, diff = content.test_hdf(BaseHdfProvider(hdf_path))

            if is_content_as_expected:
                # hdf file exists and contains all data with same parameters
                logger.info(
                    f"found existing {cls.__name__} file with match parameter setup. No build required."
                )
            else:
                logger.info("found difference in existing hdf file.")
                diff.write_diff(writer=LogWriter.info(), header=f"{cls.__name__}:")
                logger.info(
                    f"Found {cls.__name__} file with mismatching groups. Delete file and recreate."
                )
                os.remove(hdf_path)
                logger.info(f"Create hdf {hdf_path}")
                obj._create_count_error_hdf(map_p, glb_pos, with_rsd, rsd_p)
        else:
            logger.info("no existing hdf file found. Build hdf...")
            obj._create_count_error_hdf(map_p, glb_pos, with_rsd, rsd_p)

        return obj

    @property
    def is_initialized(self):
        if os.path.exists(self.hdf_path):
            for g in [
                self.g_error_count_all,
                self.g_error_count_rsd_all,
                self.g_error_count_rsd_local,
            ]:
                h = BaseHdfProvider(self.hdf_path, group=g)
                if not h.contains_group(g):
                    return False
            return True
        else:
            return False

    @property
    def hdf_map_measure(self) -> BaseHdfProvider:
        if self._hdf_map_measure is None:
            self._hdf_map_measure = BaseHdfProvider(
                self.hdf_path, group=self.g_error_count_all, allow_lazy_loading=False
            )
        return self._hdf_map_measure

    @property
    def hdf_map_measure_rsd(self) -> BaseHdfProvider:
        if self._hdf_map_measure_rsd is None:
            self._hdf_map_measure_rsd = BaseHdfProvider(
                self.hdf_path,
                group=self.g_error_count_rsd_all,
                allow_lazy_loading=False,
            )
        return self._hdf_map_measure_rsd

    @property
    def hdf_map_measure_rsd_local(self) -> BaseHdfProvider:
        if self._hdf_map_measure_rsd_local is None:
            self._hdf_map_measure_rsd_local = BaseHdfProvider(
                self.hdf_path,
                group=self.g_error_count_rsd_local,
                allow_lazy_loading=False,
            )
        return self._hdf_map_measure_rsd_local

    def _pull_ground_truth_data(
        self, glb_pos, rsd_p: RsdAssociationProvider | None = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        glb_all = (
            glb_pos.groupby(["simtime"])[["x"]].count().set_axis(["count"], axis=1)
        )
        glb_rsd = glb_pos
        if "servingEnb" not in glb_rsd:
            if rsd_p is None:
                raise ValueError(
                    "Rsd association missing in postion data. Either provide an RsdAssociationProvider or update the global position dataframe with a servingEnb column"
                )
            else:
                glb_rsd = rsd_p.merge_rsd_id_on_host_time_interval(
                    glb_pos, "node_id", "simtime", append_interval=False
                )
        glb_rsd = (
            glb_rsd.groupby(["servingEnb", "simtime"])[["x"]]
            .count()
            .set_axis(["count"], axis=1)
        )
        return glb_all, glb_rsd

    def _append_rsd_and_write(self, hdf: BaseHdfProvider, data: pd.DataFrame, rsd: int):
        """Append ressource sharing domain (RSD) index column

        Args:
            hdf (BaseHdfProvider): file to save data
            data (pd.DataFrame): data
            rsd (int): domain identifier
        """
        data["rsd_id"] = rsd
        data = data.set_index(["rsd_id"], append=True).sort_index()
        _mask_nan = data.isna().any(axis=1)
        if any(_mask_nan):
            _nans = data[_mask_nan]
            logger.info(
                f"found {_nans.shape[0]} rows with nan values. Map application did not start yet. \nDropping rows:\n{_nans}"
            )
            data = data[~_mask_nan]
        hdf.write_frame(
            group=hdf.group, frame=data, index=True, index_data_columns=True
        )

    def time_chunked_stream(self, map_p: DpmmProvider, chunk_size, time_interval):
        if chunk_size is None:
            _i = pd.IndexSlice
            yield map_p[
                _i[:, :, :, 1:], _i["count", "rsd_id", "owner_rsd_id"]
            ]  # all but ground truth
        else:
            start = time_interval[0]
            end = start + chunk_size
            while start < time_interval[1]:
                ts = it.default_timer()
                n = map_p.select(
                    where=f"simtime >= {start} and simtime < {end}",
                    columns=["count", "rsd_id", "owner_rsd_id"],
                )
                size = sys.getsizeof(n) / 1e6
                logger.info(
                    f"simtime range: ({start}, {end}] retriving {n.shape[0]:,} rows took {it.default_timer() - ts:2.2f} seconds for {size:,.2f}MB (sizeoff)"
                )
                start = end
                end = start + chunk_size
                yield n

    def _aggregate_time_chunk(
        self,
        glb: pd.DataFrame,
        glb_rsd: pd.DataFrame,
        data: pd.DataFrame,
        rsd_list: List[int],
    ):
        # all data, no distinction for rsd
        data_all = self._aggregate_group(glb, node_data=data["count"])
        self.hdf_map_measure.write_frame(group=self.g_error_count_all, frame=data_all)

        for rsd in rsd_list:
            # seperate by rsd all data point
            mask_rsd = data["rsd_id"] == rsd
            data_rsd = self._aggregate_group(
                glb_rsd.loc[rsd].copy(), node_data=data[mask_rsd]["count"].copy()
            )
            self._append_rsd_and_write(self.hdf_map_measure_rsd, data=data_rsd, rsd=rsd)
            del data_rsd

            # seperate by rsd only rsd local data
            mask_rsd_local = mask_rsd & (data["owner_rsd_id"] == rsd)
            data_rsd_local = self._aggregate_group(
                glb_rsd.loc[rsd].copy(), node_data=data[mask_rsd_local]["count"].copy()
            )
            self._append_rsd_and_write(
                self.hdf_map_measure_rsd_local, data=data_rsd_local, rsd=rsd
            )
            del data_rsd_local

    @timing
    def _create_count_error_hdf(
        self,
        map_p: DpmmProvider,
        glb_pos: DpmmGlobalPosition,
        with_rsd: bool,
        rsd_p: RsdAssociationProvider | None = None,
    ):
        glb_all, glb_rsd = self._pull_ground_truth_data(glb_pos, rsd_p)

        map_p.print_info(fd=LogWriter.info2())
        time_interval = map_p.get_attribute("time_interval")
        time_chunk_size = create_time_chunk_intervall(
            nrows=map_p.get_attribute("NROWS"),
            average_chunk_size_bytes=2e9,  # 2GB chunks
            time_interval=time_interval,
            bytes_per_row=48,  # 4xfloat64 and 4xint32
        )

        rsd_list = []
        if with_rsd:
            try:
                rsd_list = map_p.get_rsd_ids()
            except Exception as e:
                raise ValueError(
                    f"CellEntropyValueError schould be build with RSD support but retrieving the rsd list faield with: {e}"
                )

        for data in self.time_chunked_stream(map_p, time_chunk_size, time_interval):
            if time_chunk_size is None:
                # only one chunk, thus all data. no slicing needed
                _glb_all = glb_all
                _glb_rsd = glb_rsd
            else:
                # select global data based on time_chunk returened.
                t_min = data.index.get_level_values("simtime").min()
                t_max = data.index.get_level_values("simtime").max()

                _glb_all = glb_all.loc[t_min:t_max]
                _glb_rsd = glb_rsd.loc[pd.IndexSlice[:, t_min:t_max], :]
            self._aggregate_time_chunk(_glb_all, _glb_rsd, data, rsd_list=rsd_list)

        # write time_interval attribute at end of processing. If this is missing the processing was not completed.
        for _hdf in [
            self.hdf_map_measure,
            self.hdf_map_measure_rsd,
            self.hdf_map_measure_rsd_local,
        ]:
            _hdf.set_attribute(
                attr_key="time_interval", value=time_interval, group=_hdf.group
            )

        self.hdf_map_measure.repack_hdf(keep_old_file=False)  # same file only once.

    # todo find erronoues stuff....
    # def map_count_by_rsd(self, rsd_list):
    #     # only ground truth
    #     # frame (simtime, x, y, ID)[count]
    #     glb = self.count_p[pd.IndexSlice[:, :, :, 0], ["count"]]

    #     for rsd in rsd_list:
    #         # frame (simtime, x, y, source, ID)
    #         measure = self.map_p.select(where=f"{DpmmKey.RSD_ID}={rsd} and {DpmmKey.RSD_ID_OWNER}={rsd} and count", columns=["count"]) # only local or all meausres

    #         # index of the form (simtime, x, y, ID=0), sorted unique
    #         # These are all timestamped cells assocated with the provided rsd, retrieved over map measurements.
    #         time_cell_by_m = measure.index.droplevel("source").to_frame()
    #         time_cell_by_m["ID"] = 0
    #         time_cell_by_m = pd.MultiIndex.from_frame(time_cell_by_m).sort_values().unique()
    #         cells_by_m: pd.MultiIndex = time_cell_by_m.droplevel(["simtime", "ID"]).unique()

    #         cell_size = 10
    #         # index of the form (simtime, x, y, ID=0), sorted unique
    #         # These are all timestamped cells assocated with the provided rsd, retrieved using UE's trace data.
    #         time_cell_by_ue = self.pos.ue.select(where=f"servingEnb=={rsd}", columns=["time", *CoordinateType.xy_cell.cols]).set_axis(["simtime", "x", "y"], axis=1)
    #         time_cell_by_ue["ID"] = 0
    #         time_cell_by_ue["x"] = np.floor(time_cell_by_ue["x"] / cell_size) * cell_size
    #         time_cell_by_ue["y"] = np.floor(time_cell_by_ue["y"] / cell_size) * cell_size
    #         time_cell_by_ue = pd.MultiIndex.from_frame(time_cell_by_ue).sort_values().unique()
    #         cells_by_ue: pd.MultiIndex = time_cell_by_ue.droplevel(["simtime", "ID"]).unique()

    #         _only_in_m = cells_by_m.difference(cells_by_ue)
    #         _intersection = cells_by_m.intersection(cells_by_ue)
    #         _only_in_ue = cells_by_ue.difference(cells_by_m)

    #         print(f"{rsd}: found {len(time_cell_by_m)} unique timestamped cells (and {len(cells_by_m)} cells) based on map measurements over time.")
    #         print(f"{rsd}: found {len(time_cell_by_ue)} unique timestamped cell (and {len(cells_by_ue)} cells) based associated ue traces over time.")
    #         print(f"{rsd}: out of all cells {len(_only_in_m)} are only provide by measruements, {len(_only_in_ue)} are only provided by assocateid ue traces and {len(_intersection)} cells are part of of both sets.")
    #         print("hi")

    def _aggregate_group(self, glb: pd.DataFrame, node_data: pd.DataFrame):
        nodes: pd.DataFrame = (
            node_data.groupby(level=["ID", "simtime"])
            .sum()
            .groupby(level="simtime")
            .agg(
                [
                    "mean",
                    percentile(0.5),
                    percentile(0.25),
                    percentile(0.75),
                    "min",
                    "max",
                ]
            )
        )
        nodes = nodes.rename(
            columns={
                "p_50": "map_median_count",
                "mean": "map_mean_count",
                "p_25": "map_count_p25",
                "p_75": "map_count_p75",
                "min": "map_count_min",
                "max": "map_count_max",
            }
        )
        glb.columns = ["map_glb_count"]

        df = pd.concat([glb, nodes], axis=1)
        df["map_mean_err"] = df["map_mean_count"] - df["map_glb_count"]
        df["map_mean_sqrerr"] = np.power(df["map_mean_err"], 2)
        df["map_median_err"] = df["map_median_count"] - df["map_glb_count"]
        df["map_median_sqerr"] = np.power(df["map_median_err"], 2)

        return df
