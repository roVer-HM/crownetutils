import json
import os
import timeit as it
from typing import List

import numpy as np
import pandas as pd

from crownetutils.analysis.dpmm.dpmm_cfg import DpmmCfgDb
from crownetutils.analysis.dpmm.dpmm_sql import DpmmSql, TimeChunks
from crownetutils.analysis.hdf.provider import BaseHdfProvider
from crownetutils.utils.logging import logger, set_level, timing
from crownetutils.utils.plot import calc_box_stats, percentile


class MapAgeOverDistance:
    g_age_over_dist = "age_over_dist"
    g_age_over_dist_rsd = "age_over_dist_rsd"

    def __init__(
        self,
        hdf_path,
    ) -> None:
        self.hdf_path = hdf_path
        self._hdf = None
        self._hdf_rsd = None
        self.box_path = os.path.join(
            os.path.dirname(self.hdf_path), "age_over_distance_box_data.txt"
        )
        self.box_path_rsd = os.path.join(
            os.path.dirname(self.hdf_path), "age_over_distance_box_data_rsd.txt"
        )

    @classmethod
    def get_or_create(
        cls,
        hdf_path,
        map_sql: DpmmSql,
        distance_interval: float = 50.0,
        start_time: float = 0.0,
        time_bin=10.0,
        end_time=1001.0,
    ):
        obj = cls(hdf_path)
        if os.path.exists(obj.hdf_path):
            return obj
        else:
            obj._build(map_sql, distance_interval, start_time, time_bin, end_time)
            return obj

    @property
    def hdf_age_over_dist(self) -> BaseHdfProvider:
        if self._hdf is None:
            self._hdf = BaseHdfProvider(self.hdf_path, group=self.g_age_over_dist)
        return self._hdf

    @property
    def hdf_age_over_dist_rsd(self) -> BaseHdfProvider:
        if self._hdf_rsd is None:
            self._hdf_rsd = BaseHdfProvider(
                self.hdf_path, group=self.g_age_over_dist_rsd
            )
        return self._hdf_rsd

    def _build(
        self,
        map_sql: DpmmSql,
        distance_interval: float = 50.0,
        start_time: float = 0.0,
        time_bin=10.0,
        end_time=1001.0,
    ):
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
            from dcd_map as d where d.simtime >= {} and d.simtime < {};
        """
        t_chunks = TimeChunks(
            name="time", start=start_time, end=end_time, size=time_bin
        )
        chunk_len = len(t_chunks)

        with map_sql.con() as _con:
            for chunk_num, chunk in enumerate(t_chunks):
                ts = it.default_timer()
                logger.debug(
                    f"{chunk_num}/{chunk_len} run query with chunk values: {chunk}"
                )
                glb_chunk = pd.read_sql_query(
                    sql_global_pos.format(*chunk.args), _con
                ).add_suffix("_g")
                data_chunk = pd.read_sql_query(sql_map_data.format(*chunk.args), _con)
                logger.debug(
                    f"{chunk_num}/{chunk_len} query chunk took {it.default_timer() - ts:2,.2f} seconds. shape {data_chunk.shape}"
                )
                ts = it.default_timer()
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
                data_chunk = _d

                # calc dist
                pos = data_chunk[["x", "y", "x_g", "y_g"]].values.reshape((-1, 2, 2))
                dist = 10 * pos[:, 0] - 10 * pos[:, 1]
                data_chunk["dist"] = np.linalg.norm(dist, axis=1)

                distance_bins = pd.interval_range(
                    start=0,
                    end=data_chunk["dist"].max() + distance_interval,
                    freq=distance_interval,
                    closed="left",
                )
                data_chunk["dist_bin"] = pd.cut(data_chunk["dist"], bins=distance_bins)
                logger.debug(
                    f"{chunk_num}/{chunk_len} query and merge chunk took {it.default_timer() - ts:2,.2f} seconds. shape {data_chunk.shape}"
                )
                # create stats
                ts = it.default_timer()
                stats_all, boxes_all = self.calc_stats(
                    data=data_chunk,
                    group_by=["dist_bin"],
                    cols=["age_of_information", "tx_delay", "age_of_access"],
                    index_names=["dist_bin", "metric"],
                    chunk=chunk,
                    new_index=["dist_left", "metric", "time_left"],
                )
                stats_rsd, boxes_rsd = self.calc_stats(
                    data=data_chunk,
                    group_by=["dist_bin", "owner_rsd_id"],
                    cols=["age_of_information", "tx_delay", "age_of_access"],
                    index_names=["dist_bin", "rsd", "metric"],
                    chunk=chunk,
                    new_index=["rsd", "dist_left", "metric", "time_left"],
                )
                logger.debug(
                    f"{chunk_num}/{chunk_len} creating stats for chunk  took {it.default_timer() - ts:2,.2f} seconds."
                )

                # create stats
                ts = it.default_timer()
                self.hdf_age_over_dist.write_frame(
                    frame=stats_all, group=self.hdf_age_over_dist.group
                )
                self.hdf_age_over_dist_rsd.write_frame(
                    frame=stats_rsd, group=self.hdf_age_over_dist_rsd.group
                )

                self.append_boxes(boxes_all, path=self.box_path)
                self.append_boxes(boxes_rsd, path=self.box_path_rsd)
                logger.debug(
                    f"{chunk_num}/{chunk_len} save stats for chunk  took {it.default_timer() - ts:2,.2f} seconds."
                )

    def calc_stats(
        self, data: pd.DataFrame, group_by, cols, index_names, chunk, new_index
    ):
        stat = (
            data.groupby(group_by)[cols]
            .agg(
                [
                    "count",
                    "mean",
                    "std",
                    "min",
                    "max",
                    "var",
                    percentile(1),
                    percentile(99),
                    *[percentile(x) for x in range(10, 101, 10)],
                    calc_box_stats(),
                ]
            )
            .dropna()
        )

        stat = stat.stack(0)
        stat.index.names = index_names

        boxes = []
        for _index, row in stat.iterrows():
            b = row["box_stats"]
            b["time"] = [chunk.args[0], chunk.args[1]]
            if len(_index) == 3:
                # with rsd
                b["bin"] = [_index[0].right, _index[0].left]
                b["metric"] = _index[2]
                b["rsd"] = _index[1]
            else:
                b["bin"] = [_index[0].right, _index[0].left]
                b["metric"] = _index[1]
                b["rsd"] = -1
            boxes.append(b)

        stat = stat.drop(columns=["box_stats"])
        stat = self.remove_interval(stat, col="dist_bin", base_name="dist")
        stat["time_left"] = chunk.args[0]
        stat["time_right"] = chunk.args[1]
        stat = stat.set_index(new_index).sort_index()
        return stat, boxes

    def _build_old(
        self,
        map_sql: DpmmSql,
        distance_interval: float = 50.0,
        start_time: float = 0.0,
        time_bin=10.0,
        end_time=1001.0,
    ):
        sql = """
            select 
                    m.simtime - m.measured_t as 'age_of_information', 
                    m.received_t - m.measured_t as 'tx_delay',
                    m.simtime - m.received_t as 'age_of_access',   
                    m.owner_rsd_id,
                    10*(m.x - glb.x)*10*(m.x - glb.x) + 10*(m.y - glb.y)*10*(m.y - glb.y) as sqr_dist
            from 
                dcd_map as m
            join 
                (
                select 
                    *
                from 
                    dcd_map_glb as g
                join 
                    glb_node_id_mapping  as n
                on
                    g.uid = n.glb_map_uid
                where 
                    g.simtime >= {} and g.simtime < {}
                ) as glb 
            on 
                m.hostId = glb.node_id and m.simtime = glb.simtime
        """
        t_chunks = TimeChunks(
            name="time", start=start_time, end=end_time, size=time_bin
        )

        for d, chunk in map_sql.chunk_query(sql_template=sql, chunk_provider=t_chunks):
            ts = it.default_timer()
            d["dist"] = np.sqrt(d["sqr_dist"])
            bins = pd.interval_range(
                start=0,
                end=d["dist"].max() + distance_interval,
                freq=distance_interval,
                closed="left",
            )
            d["dist_bin"] = pd.cut(d["dist"], bins=bins)
            d["bin_right"] = d["dist_bin"].apply(lambda x: x.right)
            d["bin_right"] = d["bin_right"].astype(int)
            print(f"got data {d.shape} for chunk {chunk}")

            stats_glb: pd.DataFrame = self.calc_stats(
                d,
                group_by="dist_bin",
                cols=["age_of_information", "tx_delay", "age_of_access"],
            ).dropna()
            stats_glb = stats_glb.stack(0)
            stats_glb.index.names = ["dist_bin", "metric"]

            boxes = []
            for (bin, metric), rows in stats_glb.iterrows():
                b = rows["box_stats"]
                b["metric"] = metric
                b["bin"] = [bin.right, bin.left]
                b["bin_closed"] = bin.closed
                b["time"] = [chunk.args[0], chunk.args[1]]
                b["rsd"] = -1
                boxes.append(b)
            stats_glb = stats_glb.drop(columns=["box_stats"])
            stats_glb = self.remove_interval(stats_glb)
            stats_glb["time_left"] = chunk.args[0]
            stats_glb["time_right"] = chunk.args[1]
            stats_glb = stats_glb.set_index(["metric", "bin_left"]).sort_index()

            boxes_rsd = []
            stats_rsd = self.calc_stats(
                d,
                group_by=["dist_bin", "owner_rsd_id"],
                cols=["age_of_information", "tx_delay", "age_of_access"],
            ).dropna()
            stats_rsd = stats_rsd.stack(0)
            stats_rsd.index.names = ["dist_bin", "rsd", "metric"]
            for (bin, rsd, metric), rows in stats_rsd.iterrows():
                b = rows["box_stats"]
                b["metric"] = metric
                b["bin"] = [bin.right, bin.left]
                b["bin_closed"] = bin.closed
                b["time"] = [chunk.args[0], chunk.args[1]]
                b["rsd"] = rsd
                boxes_rsd.append(b)
            stats_rsd = stats_rsd.drop(columns=["box_stats"])
            stats_rsd = self.remove_interval(stats_rsd)
            stats_rsd["time_left"] = chunk.args[0]
            stats_rsd["time_right"] = chunk.args[1]
            stats_rsd = stats_rsd.set_index(
                ["rsd", "metric", "bin_left", "time_left"]
            ).sort_index()

            self.hdf_age_over_dist.write_frame(frame=stats_glb)
            self.append_boxes(boxes, path=self.box_path)

            self.hdf_age_over_dist_rsd.write_frame(frame=stats_glb)
            self.append_boxes(boxes_rsd, path=self.box_path_rsd)

            logger.debug(f"stats took {it.default_timer() - ts:2,.2f} seconds to")

    def remove_interval(self, data: pd.DataFrame, col, base_name):
        data = data.reset_index()
        data[f"{base_name}_left"] = data[col].apply(lambda x: x.left).astype(float)
        data[f"{base_name}_right"] = data[col].apply(lambda x: x.right).astype(float)
        data = data.drop(columns=col)
        return data

    def append_boxes(self, boxes: List[dict], path):
        with open(path, "+a", encoding="utf-8") as fd:
            for b in boxes:
                j_str = json.dumps(b, ensure_ascii=True, indent=None, sort_keys=False)
                fd.write(j_str)
                fd.write("\n")
