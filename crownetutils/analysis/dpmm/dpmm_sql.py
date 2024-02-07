from __future__ import annotations

import contextlib
import os
import sqlite3 as sq
import sys
import timeit as it
from typing import List, Tuple

import pandas as pd

from crownetutils.analysis.dpmm.csv_loader import _apply_real_coords
from crownetutils.analysis.dpmm.dpmm_cfg import DpmmCfgDb
from crownetutils.analysis.dpmm.metadata import DpmmMetaData
from crownetutils.utils.logging import logger, timing


class TimeChunks:
    def __init__(self, name, start, end, size) -> None:
        self.start = start
        self.end = end
        self.size = size
        self.chunks = []
        _s = start
        while _s < end:
            self.chunks.append(Chunk(name, _s, _s + size))
            _s += size

    def __len__(self):
        return len(self.chunks)

    def __iter__(self):
        return iter(self.chunks)


class Chunk:
    def __init__(self, name, *args) -> None:
        self.name = name
        self.args = args

    def __str__(self) -> str:
        return f"{self.name} chunk with args: {self.args}"


class DpmmSql:
    def __init__(self, cfg: DpmmCfgDb) -> None:
        self.cfg: DpmmCfgDb = cfg
        self.path = os.path.abspath(os.path.join(cfg.base_dir, cfg.map_db_name))

    @property
    def tbl_metadata(self) -> str:
        return self.cfg.tbl_metadata

    @property
    def tbl_map(self) -> str:
        return self.cfg.tbl_map

    @property
    def tbl_map_glb(self) -> str:
        return self.cfg.tbl_map_glb

    @property
    def tbl_alg_mapping(self) -> str:
        return self.cfg.tbl_alg_mapping

    @property
    def tbl_glb_node_id_mapping(self) -> str:
        return self.cfg.tbl_glb_node_id_mapping

    def read_map(
        self,
        host_id,
        index_types: dict,
        col_types_dict: dict,
        real_coords: bool = True,
        df_filter=None,
    ) -> Tuple[pd.DataFrame, DpmmMetaData]:
        _cols = {**index_types, **col_types_dict}
        cols = [f"m.{c}" for c in _cols.keys()]
        cols = ", ".join(cols)
        if "selection" in cols:
            cols = cols.replace("m.selection,", "a.alg_name as 'selection',")
        order = [f"m.{c}" for c in index_types.keys()]
        order = ", ".join(order)
        _sql_str = f"select {cols} from {self.tbl_map} as m  join {self.tbl_alg_mapping} a on m.selection = a.uid where m.hostId = {host_id} order by {order} asc"
        df = self.query(_sql_str, type="df", dtype=_cols)
        df = df.set_index(list(index_types.keys()))
        df = df[list(col_types_dict.keys())].copy()
        # apply given filters first (if any)
        if df_filter is not None:
            if type(df_filter) == list:
                for _f in df_filter:
                    df = _f(df)
            else:
                # apply early filter to remove not needed data to increase performance
                df = df_filter(df)
        meta = self.metadata(host_id)
        if real_coords:
            df = _apply_real_coords(df, meta)
        return df, meta

    def get_host_ids(self) -> List[int]:
        _sql_str = f"select distinct m.hostId from {self.tbl_metadata} as m where m.hostId > 0 order by m.hostId asc"
        df = self.query(_sql_str, type="df")
        return df["hostId"].to_list()

    def metadata(self, hostId) -> DpmmMetaData:
        _sql_str = f"select m.key, m.value from {self.tbl_metadata} as m where m.hostId == {hostId}"
        df = self.query(_sql_str, type="df")
        m = {k: v for k, v in df.values}
        return DpmmMetaData.from_dict(m)

    def glb_metadata(self) -> DpmmMetaData:
        return self.metadata(-1)

    def dmap_row_count(self) -> int:
        q = self.query("SELECT count(*) from dcd_map;")
        return q.values[0][0]

    @timing
    def append_dmap_time_index_if_missing(self):
        index_sql = [
            "CREATE INDEX IF NOT EXISTS dcd_map_idx_simtime ON dcd_map (simtime);",
        ]
        with self.con() as c:
            for s in index_sql:
                logger.debug(f"execute: {s}")
                ret = c.execute(s).fetchall()

    def chunk_query(
        self,
        sql_template: str,
        chunk_provider,
        journal: str = "MEMORY",
        cache_size_kb: int = 2e6,
        mmap=0,
    ) -> pd.DataFrame:
        with self.con() as _con:
            r = _con.execute(f"PRAGMA cache_size;").fetchall()
            print(r)
            r = _con.execute(f"PRAGMA cache_size = -10000;").fetchall()
            print(r)
            r = _con.execute(f"PRAGMA cache_size;").fetchall()
            print(r)
            if mmap > 0:
                r = _con.execute(f"PRAGMA mmap_size;").fetchall()
                print(f"mmap old {r}")
                r = _con.execute(f"PRAGMA mmap_size = {mmap};").fetchall()
                print(f"mmap new {r}")
            # r = _con.execute(f"PRAGMA journal_mode = {journal};").fetchall()

            for chunk in chunk_provider:
                ts = it.default_timer()
                logger.debug(f"run query with chunk values: {chunk}")
                sql_str = sql_template.format(*chunk.args)
                # sql_str = sql_template.format(0, 5)
                # print(sql_str)
                df = pd.read_sql_query(sql_str, _con)
                s1 = sys.getsizeof(df) / 1e6
                logger.debug(
                    f"query took {it.default_timer() - ts:2,.2f} seconds to load {df.shape[0]} rows with {s1:,.2f}MB"
                )
                yield df, chunk

    def read_glb_map(
        self,
        index_types: dict,
        col_types_dict: dict,
        real_coords: bool = True,
        df_filter=None,
    ) -> Tuple[pd.DataFrame, DpmmMetaData]:
        _cols = {**index_types, **col_types_dict}
        cols = [f"g.{c}" for c in _cols.keys()]
        cols = ", ".join(cols)
        _sql_str = f"select {cols} from {self.tbl_map_glb} as g"

        df = self.query(_sql_str, type="df", dtype=_cols)
        df = df.set_index(list(index_types.keys()))
        # apply given filters first (if any)
        if df_filter is not None:
            if type(df_filter) == list:
                for _f in df_filter:
                    df = _f(df)
            else:
                # apply early filter to remove not needed data to increase performance
                df = df_filter(df)
        meta = self.glb_metadata()
        if real_coords:
            df = _apply_real_coords(df, meta)
        return df

    def read_global_position(
        self, index_types: dict, real_coords: bool = True, df_filter=None
    ) -> Tuple[pd.DataFrame, DpmmMetaData]:
        i_c = [f"g.{c}" for c in list(index_types.keys())]
        i_c = ", ".join(i_c)
        _sql_str = f"select {i_c}, n.node_id from {self.tbl_map_glb} as g join {self.tbl_glb_node_id_mapping} as n on g.uid == n.glb_map_uid order by g.simtime, g.x, g.y asc"
        df = self.query(_sql_str, type="df")
        df = df.set_index(list(index_types.keys()))
        # apply given filters first (if any)
        if df_filter is not None:
            if type(df_filter) == list:
                for _f in df_filter:
                    df = _f(df)
            else:
                # apply early filter to remove not needed data to increase performance
                df = df_filter(df)
        meta = self.glb_metadata()
        if real_coords:
            df = _apply_real_coords(df, meta)
        return df.reset_index()

    @contextlib.contextmanager
    def con(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        try:
            _con = sq.connect(self.path)
            yield _con
        finally:
            _con.close()

    def query(self, sql_str, type="df", **kwargs) -> pd.DataFrame | sq.Cursor:
        with self.con() as _con:
            if type == "df":
                return pd.read_sql_query(sql_str, _con, **kwargs)
            elif type == "cursor":
                return _con.execute(sql_str, **kwargs)
            else:
                raise RuntimeError("Expected df or cursor as type")

    def create_cell_count_by_host_id_over_time_if_missing(self):
        _sql = """CREATE TABLE IF NOT EXISTS
                        cell_count_by_host_id_over_time as
                    SELECT 
                        m.simtime, m.hostId, count(m.count) as 'numberOfCells'
                    FROM 
                        dcd_map as m 
                    GROUP BY 
                        m.simtime, m.hostId
                    ORDER BY 
                        m.simtime asc;
                """
        with self.con() as _con:
            _con.execute(_sql)
            _con.commit()

    def get_cell_count_by_host_id_over_time(self, sql_str=None) -> pd.DataFrame:
        """Return frame with [simtime, hostId, numberOfCells]"""

        self.create_cell_count_by_host_id_over_time_if_missing()
        if sql_str is None:
            _sql = """select * from cell_count_by_host_id_over_time as t """
        else:
            _sql = sql_str
        return self.query(_sql, type="df")
