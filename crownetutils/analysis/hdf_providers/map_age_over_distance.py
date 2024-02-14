import json
import os
import timeit as it
from typing import Any, List

import numpy as np
import pandas as pd

from crownetutils.analysis.dpmm.dpmm_cfg import DpmmCfgDb
from crownetutils.analysis.dpmm.dpmm_sql import DpmmSql, RowIdChunk, TimeChunks
from crownetutils.analysis.hdf.provider import BaseHdfProvider, HdfInconsistentState
from crownetutils.analysis.hdf_providers.helper import ExpectedHdfContent
from crownetutils.utils.logging import LogWriter, logger, set_level, timing
from crownetutils.utils.plot import calc_box_stats, mult_percentile, percentile


class MapDistanceData:
    sql_global_pos = """
        select 
            d.simtime, d.x, d.y, n.node_id as hostId  
        from 
            dcd_map_glb as d 
        join 
            glb_node_id_mapping  as n
        on
            d.uid = n.glb_map_uid
        where d.simtime >= {} and d.simtime < {};
    """
    sql_map_data = """
        select  
                d.simtime, d.x, d.y, d.hostId, d.owner_rsd_id,
                d.simtime - d.measured_t as 'age_of_information', 
                d.received_t - d.measured_t as 'tx_delay',
                d.simtime - d.received_t as 'age_of_access'
        from dcd_map as d where d.uid >= {} and d.uid <= {}; 
    """

    def __init__(
        self,
        map_sql: DpmmSql,
        chunk_list: List[RowIdChunk],
        distance_interval: float = 50.0,
        cell_size: float = 10.0,
    ) -> None:
        self.map_sql: DpmmSql = map_sql
        self.chunk_list: List[RowIdChunk] = chunk_list
        self.distance_interval: float = distance_interval
        self.cell_size: float = cell_size

    def get_data_stream(self):
        chunk_len = len(self.chunk_list)
        with self.map_sql.con() as _con:
            for chunk_num, chunk in enumerate(self.chunk_list):
                ts = it.default_timer()
                data_chunk: pd.DataFrame = self._pull_chunk(
                    connection=_con,
                    chunk=chunk,
                    chunk_num=chunk_num,
                    chunk_len=chunk_len,
                )

                data_chunk = self.calc_dist(data_chunk=data_chunk)

                logger.debug(
                    f"{chunk_num}/{chunk_len} query and merge chunk took {it.default_timer() - ts:2,.2f} seconds. shape {data_chunk.shape}"
                )

                yield chunk_num, chunk, data_chunk

    def calc_dist(self, data_chunk: pd.DataFrame) -> pd.DataFrame:
        # calc dist
        pos = data_chunk[["x", "y", "x_g", "y_g"]].values.reshape((-1, 2, 2))
        dist = self.cell_size * pos[:, 0] - self.cell_size * pos[:, 1]
        data_chunk["dist"] = np.linalg.norm(dist, axis=1)

        distance_bins = pd.interval_range(
            start=0,
            end=data_chunk["dist"].max() + self.distance_interval,
            freq=self.distance_interval,
            closed="left",
        )
        data_chunk["dist_bin"] = pd.cut(data_chunk["dist"], bins=distance_bins)
        return data_chunk

    def _pull_chunk(
        self, connection, chunk: RowIdChunk, chunk_num: int, chunk_len: int
    ) -> pd.DataFrame:
        ts = it.default_timer()
        logger.debug(
            f"{chunk_num}/{chunk_len} run query with chunk values: {chunk.info_str()}"
        )
        glb_chunk = pd.read_sql_query(
            self.sql_global_pos.format(*chunk.args_time), connection
        ).add_suffix("_g")
        time_glb_query = it.default_timer() - ts
        data_chunk = pd.read_sql_query(
            self.sql_map_data.format(*chunk.args_row), connection
        )
        time_total_query = it.default_timer() - ts
        time_data_query = time_total_query - time_glb_query

        # merge
        _d = data_chunk.merge(
            glb_chunk,
            left_on=["hostId", "simtime"],
            right_on=["hostId_g", "simtime_g"],
        )
        if _d.shape[0] != data_chunk.shape[0]:
            raise ValueError(
                f"merge produced wrong number of rows. expected {data_chunk.shape[0]} got {_d.shape[0]} "
            )

        logger.debug(
            f"{chunk_num}/{chunk_len} query chunk took {time_total_query:,.2f} seconds. Glb:{glb_chunk.shape}/{time_glb_query:,.2f}s data:{data_chunk.shape}/{time_data_query:,.2f}s"
        )
        return _d


class CalcStats:
    def __init__(self, agg_function_list: List[Any], metric_map: dict) -> None:
        self.agg_func_list: List[Any] = agg_function_list
        self.metric_map: dict = metric_map
        self.cleanup_f = self._stat_cleanup

    def _stat_cleanup(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.dropna()

    def get_stat_aggregation_functions(self) -> List[Any]:
        return self.agg_func_list

    @timing
    def calc_stats(
        self,
        data: pd.DataFrame,
        group_by: List[str],
        cols: List[str],
        index_names: List[str],
        chunk: RowIdChunk,
        stat_index: List[str],
    ):
        agg_func_list = self.get_stat_aggregation_functions()
        stat = data.groupby(group_by)[cols].agg(agg_func_list)

        stat = stat.stack(0)
        stat.index.names = index_names
        stat = self.cleanup_f(stat)

        if "p_mult" in stat.columns:
            stat = pd.concat(
                [
                    stat,
                    pd.DataFrame.from_records(stat["p_mult"].values, index=stat.index),
                ],
                axis=1,
            )
            stat = stat.drop(columns="p_mult")

        if "box_stats" in stat.columns:
            stat, outliers = self.process_box_stats(stat, chunk, stat_index)
        else:
            outliers = None

        stat = self.remove_interval(stat, col="dist_bin", base_name="dist")
        stat["time_left"] = chunk.args_time[0]
        stat["time_right"] = chunk.args_time[1]
        stat["metric"] = stat["metric"].apply(lambda x: self.metric_map[x]).astype(int)
        stat = stat.set_index(stat_index).sort_index()

        if outliers is None:
            return stat
        else:
            return stat, outliers

    @timing
    def process_box_stats(
        self, stat: pd.DataFrame, chunk: RowIdChunk, stat_index: List[str]
    ) -> pd.DataFrame:
        stat = pd.concat(
            [
                stat,
                pd.DataFrame.from_records(stat["box_stats"].values, index=stat.index),
            ],
            axis=1,
        )

        o = stat["outliers"]

        o1 = o.apply(lambda x: x[0]).explode().to_frame().dropna()
        o1["type"] = 0
        o1 = o1.set_index("type", append=True)

        o2 = o.apply(lambda x: x[1]).explode().to_frame().dropna()
        o2["type"] = 1
        o2 = o2.set_index("type", append=True)

        outliers = pd.concat([o1, o2], axis=0)
        outliers = self.remove_interval(outliers, col="dist_bin", base_name="dist")
        outliers["time_left"] = chunk.args_time[0]
        outliers["time_right"] = chunk.args_time[1]
        outliers["metric"] = (
            outliers["metric"].apply(lambda x: self.metric_map[x]).astype(int)
        )
        outliers["outliers"] = outliers["outliers"].astype(float)
        outliers = outliers.set_index([*stat_index, "type"]).sort_index()
        stat = stat.drop(columns=["box_stats", "outliers"])

        return stat, outliers

    def remove_interval(self, data: pd.DataFrame, col, base_name):
        data = data.reset_index()
        data[f"{base_name}_left"] = data[col].apply(lambda x: x.left).astype(float)
        data[f"{base_name}_right"] = data[col].apply(lambda x: x.right).astype(float)
        data = data.drop(columns=col)
        return data


class MapMeasurementsAgeOverDistance:
    """Create age metrics for cell measurement, grouped only by a distance and time buckets.

    By default all cell measurements are grouped by 10.0 seconds time buckets and 50.0 meter distance
    buckets, where the distance between the cell measured and the current position of the measurement owner
    is used.

    With this aggregation the association to a specific owner is  lost. If the owner association is needed
    use the class MapAgeOverDistance.
    """

    g_age_over_dist = "age_over_dist"
    g_age_over_dist_rsd = "age_over_dist_rsd"

    g_age_over_dist_box_outl = "age_over_dist_box_plot_outliers"
    g_age_over_dist_rsd_box_outl = "age_over_dist_rsd_box_plot_outliers"

    groups = [
        g_age_over_dist,
        g_age_over_dist_rsd,
        g_age_over_dist_box_outl,
        g_age_over_dist_rsd_box_outl,
    ]
    metric_map = {"age_of_access": 0, "tx_delay": 1, "age_of_information": 2}

    def __init__(
        self,
        hdf_path,
    ) -> None:
        self.hdf_path = hdf_path
        self._hdf: BaseHdfProvider = BaseHdfProvider(
            hdf_path=self.hdf_path, group=self.g_age_over_dist
        )
        self._hdf_rsd: BaseHdfProvider = self._hdf.created_shared_provider(
            group=self.g_age_over_dist_rsd
        )
        self._hdf_outliers: BaseHdfProvider = self._hdf.created_shared_provider(
            group=self.g_age_over_dist_box_outl
        )
        self._hdf_rsd_outliers: BaseHdfProvider = self._hdf.created_shared_provider(
            group=self.g_age_over_dist
        )

    @classmethod
    def get_or_create(
        cls,
        hdf_path,
        map_sql: DpmmSql,
        distance_interval: float = 50.0,
        start_time: float = 0.0,
        time_bin=10.0,
        override_existing: bool = False,
    ):
        obj = cls(hdf_path)

        if os.path.exists(obj.hdf_path):
            content = ExpectedHdfContent().add_groups(
                groups=obj.groups, metric_map=obj.metric_map, processing_done=True
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
                if not override_existing:
                    raise ValueError(
                        f"found existing {cls.__name__} file with inconsistent parameters but override_existing is false."
                    )
                else:
                    logger.info(
                        f"Found {cls.__name__} file with mismatching groups. Delete file and recreate."
                    )
                    os.remove(hdf_path)
                    logger.info(f"Create hdf {hdf_path}")
                    obj._build(
                        map_sql=map_sql,
                        distance_interval=distance_interval,
                        start_time=start_time,
                        time_bin=time_bin,
                    )
        else:
            logger.info("no existing hdf file found. Build hdf...")
            obj._build(
                map_sql=map_sql,
                distance_interval=distance_interval,
                start_time=start_time,
            )

        return obj

    @property
    def hdf_age_over_dist(self) -> BaseHdfProvider:
        return self._hdf

    @property
    def hdf_age_over_dist_outliers(self) -> BaseHdfProvider:
        return self._hdf_outliers

    @property
    def hdf_age_over_dist_rsd(self) -> BaseHdfProvider:
        return self._hdf_rsd

    @property
    def hdf_age_over_dist_rsd_outliers(self) -> BaseHdfProvider:
        return self._hdf_rsd_outliers

    def _build(
        self,
        map_sql: DpmmSql,
        distance_interval: float = 50.0,
        start_time: float = 0.0,
        time_bin=10.0,
    ):
        time_chunks = map_sql.get_dcd_map_chunked_row_ids(
            time_bin=time_bin, start_time=start_time
        )
        chunk_len = len(time_chunks)

        dist_data_provider = MapDistanceData(
            map_sql=map_sql,
            chunk_list=time_chunks,
            distance_interval=distance_interval,
            cell_size=10.0,
        )

        stats_f = CalcStats(
            agg_function_list=[
                "count",
                "mean",
                "std",
                "min",
                "max",
                "var",
                mult_percentile(
                    1, 10, 20, 30, 40, 60, 70, 80, 90, 99
                ),  # 50 will be median provide by calc_stats
                *[percentile(x) for x in range(10, 101, 10)],
                calc_box_stats(),
            ],
            metric_map=self.metric_map,
        )

        for chunk_num, chunk, data_chunk in dist_data_provider.get_data_stream():
            # create stats
            ts = it.default_timer()

            stats_all, outliers_all = stats_f.calc_stats(
                data=data_chunk,
                group_by=["dist_bin"],
                cols=["age_of_information", "tx_delay", "age_of_access"],
                index_names=["dist_bin", "metric"],
                chunk=chunk,
                stat_index=["dist_left", "metric", "time_left"],
            )
            stats_rsd, outliers_rsd = stats_f.calc_stats(
                data=data_chunk,
                group_by=["dist_bin", "owner_rsd_id"],
                cols=["age_of_information", "tx_delay", "age_of_access"],
                index_names=["dist_bin", "rsd", "metric"],
                chunk=chunk,
                stat_index=["rsd", "dist_left", "metric", "time_left"],
            )
            logger.debug(
                f"{chunk_num}/{chunk_len} creating stats for chunk  took {it.default_timer() - ts:2,.2f} seconds."
            )

            # save stats
            ts = it.default_timer()
            self.hdf_age_over_dist.write_frame(
                frame=stats_all, group=self.hdf_age_over_dist.group
            )
            self.hdf_age_over_dist_outliers.write_frame(
                frame=outliers_all, group=self.hdf_age_over_dist_outliers.group
            )

            self.hdf_age_over_dist_rsd.write_frame(
                frame=stats_rsd, group=self.hdf_age_over_dist_rsd.group
            )
            self.hdf_age_over_dist_rsd_outliers.write_frame(
                frame=outliers_rsd, group=self.hdf_age_over_dist_rsd_outliers.group
            )

            logger.debug(
                f"{chunk_num}/{chunk_len} save stats for chunk  took {it.default_timer() - ts:2,.2f} seconds."
            )

        for g in self.groups:
            self.hdf_age_over_dist.set_attribute("processing_done", value=True, group=g)
            self.hdf_age_over_dist.set_attribute(
                "metric_map", value=self.metric_map, group=g
            )


class MapSizeAndAgeOverDistance:
    """Map size and age metrics grouped for each time step, over hostId and a distance bucket.

    With this aggregation the size of the map a node has is broken up over the distance buckets. It allows
    statements such as 'The total map of a node has X number of cells, from which Y number of cells are within
    50 meters of the owning node.'

    """

    g_map_size_over_distance = "map_size_over_dist"
    g_map_size_over_distance_rsd = "map_size_over_dist_rsd"
    groups = [g_map_size_over_distance, g_map_size_over_distance_rsd]

    metric_map = {"age_of_access": 0, "tx_delay": 1, "age_of_information": 2}

    def __init__(self, hdf_path: str) -> None:
        self._hdf_path = hdf_path
        self._hdf = BaseHdfProvider(
            hdf_path=hdf_path, group=self.g_map_size_over_distance
        )
        self._hdf_rsd = self._hdf.created_shared_provider(
            group=self.g_map_size_over_distance_rsd
        )

    @classmethod
    def get_or_create(
        cls,
        hdf_path: str,
        map_sql: DpmmSql,
        distance_interval: float = 50.0,
        start_time: float = 1.0,
        override_existing: bool = False,
        allow_append: bool = False,
    ):
        obj: MapSizeAndAgeOverDistance = MapSizeAndAgeOverDistance(hdf_path)
        if os.path.exists(hdf_path):
            content = ExpectedHdfContent().add_groups(
                groups=obj.groups, metric_map=obj.metric_map, processing_done=True
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
                if not override_existing:
                    raise ValueError(
                        f"found existing {cls.__name__} file with inconsistent parameters but override_existing is false."
                    )
                else:
                    if allow_append:
                        try:
                            start_time = obj.last_time() + 1.0
                            logger.info(
                                f"found valid state found to append to hdf. Starting with time {start_time}."
                            )
                        except Exception as e:
                            # no valid state create new one...
                            logger.info(
                                "now valid state found to append to hdf. Recreate it from scratch."
                            )
                            os.remove(hdf_path)

                        obj._build(
                            map_sql=map_sql,
                            distance_interval=distance_interval,
                            start_time=start_time,
                        )
                    else:
                        logger.info(
                            f"Found {cls.__name__} file with mismatching groups. Delete file and recreate."
                        )
                        os.remove(hdf_path)
                        logger.info(f"Create hdf {hdf_path}")
                        obj._build(
                            map_sql=map_sql,
                            distance_interval=distance_interval,
                            start_time=start_time,
                        )
        else:
            logger.info("no existing hdf file found. Build hdf...")
            obj._build(
                map_sql=map_sql,
                distance_interval=distance_interval,
                start_time=start_time,
            )

        return obj

    def last_time(self):
        last_time = self._hdf.select(start=-1).index.get_level_values("time_left")[0]
        last_time_rsd = self._hdf_rsd.select(start=-1).index.get_level_values(
            "time_left"
        )[0]
        if last_time != last_time_rsd:
            raise HdfInconsistentState(
                self._hdf_path, "Last time of groups do not match."
            )
        return last_time

    @timing
    def _build(
        self,
        map_sql: DpmmSql,
        distance_interval: float = 50.0,
        start_time: float = 1.0,
    ):
        time_chunks = map_sql.get_dcd_map_chunked_row_ids(
            time_bin=1.0, start_time=start_time
        )  # each time separately!
        chunk_len = len(time_chunks)

        dist_data_provider = MapDistanceData(
            map_sql=map_sql,
            chunk_list=time_chunks,
            distance_interval=distance_interval,
            cell_size=10.0,
        )

        stats_f = CalcStats(
            agg_function_list=[
                "count",
                "mean",
                "std",
                "min",
                "max",
                "var",
                mult_percentile(1, 25, 50, 75, 99),
            ],
            metric_map=self.metric_map,
        )
        stats_f.cleanup_f = self.stat_cleanup

        for chunk_num, chunk, data_chunk in dist_data_provider.get_data_stream():
            # create stats
            ts = it.default_timer()

            stat = stats_f.calc_stats(
                data=data_chunk,
                group_by=["hostId", "dist_bin"],
                cols=list(self.metric_map.keys()),
                index_names=["hostId", "dist_bin", "metric"],
                chunk=chunk,
                stat_index=["time_left", "dist_left", "hostId", "metric"],
            )
            stat_rsd = stats_f.calc_stats(
                data=data_chunk,
                group_by=["hostId", "dist_bin", "owner_rsd_id"],
                cols=list(self.metric_map.keys()),
                index_names=["hostId", "dist_bin", "rsd", "metric"],
                chunk=chunk,
                stat_index=["rsd", "time_left", "dist_left", "hostId", "metric"],
            )

            logger.info(
                f"{chunk_num}/{chunk_len} creating stats for chunk  took {it.default_timer() - ts:2,.2f} seconds."
            )

            # save stats
            ts = it.default_timer()
            self._hdf.write_frame(group=self._hdf.group, frame=stat)
            self._hdf_rsd.write_frame(group=self._hdf_rsd.group, frame=stat_rsd)
            logger.info(
                f"{chunk_num}/{chunk_len} save stats for chunk  took {it.default_timer() - ts:2,.2f} seconds."
            )

        for g in self.groups:
            self._hdf.set_attribute(
                "processing_done", value=True, group=g
            )  # marker to indicate that hdf file is complete
            self._hdf.set_attribute("metric_map", value=self.metric_map, group=g)

    def stat_cleanup(self, data: pd.DataFrame) -> pd.DataFrame:
        _m = data["count"] != 0
        return data[_m].copy(deep=True)
