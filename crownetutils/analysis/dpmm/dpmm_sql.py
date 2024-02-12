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

    _tbl_name_dpm_map_row_id_mapping_by_time = "dcd_map_row_id_mapping_by_time"
    _tbl_create_dpm_map_row_id_mapping_by_time = """CREATE TABLE IF NOT EXISTS '{tbl}'
            ( 
            uid             INTEGER PRIMARY KEY, 
            simtime         REAL NOT NULL, 
            row_id_start    INTEGER NOT NULL, 
            row_id_end      INTEGER NOT NULL
            );
    """.format(
        tbl=_tbl_name_dpm_map_row_id_mapping_by_time
    )

    _tbl_insert_dpm_map_row_id_mapping_by_time = """INSERT INTO 'dcd_map_row_id_mapping_by_time'
                          (simtime, row_id_start, row_id_end) 
                          VALUES (?, ?, ?);
    """.format(
        tbl=_tbl_name_dpm_map_row_id_mapping_by_time
    )

    _tbl_name_dpm_map_row_id_mapping_by_time_hostId = (
        "dcd_map_row_id_mapping_by_time_hostId"
    )
    _tbl_create_dpm_map_row_id_mapping_by_time_hostId = """CREATE TABLE IF NOT EXISTS '{tbl}'
            ( 
            uid             INTEGER PRIMARY KEY, 
            simtime         REAL NOT NULL, 
            host_id         INTEGER NOT NULL, 
            row_id_start    INTEGER NOT NULL, 
            row_id_end      INTEGER NOT NULL, 
            number_of_cells INTEGER NOT NULL 
            );
    """.format(
        tbl=_tbl_name_dpm_map_row_id_mapping_by_time_hostId
    )

    _tbl_insert_dpm_map_row_id_mapping_by_time_hostId = """INSERT INTO '{tbl}'
                          (simtime, host_id, row_id_start, row_id_end, number_of_cells) 
                          VALUES (?, ?, ?, ?, ?);
    """.format(
        tbl=_tbl_name_dpm_map_row_id_mapping_by_time_hostId
    )

    _tbl_create_idx_dpm_map_row_id_mapping_by_time_hostId = """
        CREATE UNIQUE INDEX IF NOT EXISTS "idx_simtime_host_id_on_dcd_map_row_id_mapping_by_time_hostId" ON "{tbl}" (
            "simtime"	ASC,
            "host_id"	ASC
        );
    """.format(
        tbl=_tbl_name_dpm_map_row_id_mapping_by_time_hostId
    )

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
                logger.info(f"execute: {s}")
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
                logger.info(f"run query with chunk values: {chunk}")
                sql_str = sql_template.format(*chunk.args)
                # sql_str = sql_template.format(0, 5)
                # print(sql_str)
                df = pd.read_sql_query(sql_str, _con)
                s1 = sys.getsizeof(df) / 1e6
                logger.info(
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

    def get_max_simtime(self) -> int:
        """Get maximum simtime based on dcd_map_glb table"""

        _sql = """select max(d.simtime) from dcd_map_glb as d;"""
        max_time = self.query(_sql, type="df").values[0][0]
        return max_time

    def has_complete_dcd_map_cache(self) -> bool:
        """Check if both tables 'dpm_map_row_id_mapping_by_time' and 'dpm_map_row_id_mapping_by_time_hostId' exist and have values until max_simtime. Method will raise error if tables are inconsistent."""
        _sql_has_tbl = """
            SELECT name 
            FROM sqlite_master 
            WHERE type='table' AND name='{tbl}';
        """

        cache_time_missing = self.query(
            _sql_has_tbl.format(tbl=self._tbl_name_dpm_map_row_id_mapping_by_time)
        ).empty
        cache_time_host_id_missing = self.query(
            _sql_has_tbl.format(
                tbl=self._tbl_name_dpm_map_row_id_mapping_by_time_hostId
            )
        ).empty

        if cache_time_missing or cache_time_host_id_missing:
            # cache not complete
            return False

        max_time_glb = self.get_max_simtime()
        _, max_time = self.get_last_processed_uid_of_row_mapping_cache()

        return max_time_glb == max_time

    def create_row_cache_tables_if_missing(self):
        """Create row id mapping cache tables with all needed indices. Is a NOOP if tables or index exists."""

        with self.con() as _con:
            # ensure tables exist.
            _con.execute(self._tbl_create_dpm_map_row_id_mapping_by_time_hostId)
            _con.execute(self._tbl_create_dpm_map_row_id_mapping_by_time)
            _con.execute(self._tbl_create_idx_dpm_map_row_id_mapping_by_time_hostId)
            _con.commit()

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

    def get_last_processed_uid_of_row_mapping_cache(self) -> Tuple[int, int]:
        """Returns largest row_id and the associated simtime found in both row_mapping_cache tables.

        The row_id and time are equal in both tables. If not the tables are in an inconsistent state!
        """

        _sql_max_time = """select max(t.simtime) from {tbl} as t"""
        _sql_max_row_id = """select max(t.row_id_end) from {tbl} as t"""

        try:
            max_row_s1 = self.query(
                sql_str=_sql_max_row_id.format(
                    tbl=self._tbl_name_dpm_map_row_id_mapping_by_time
                ),
                type="df",
            ).values[0][0]
            max_time_s1 = self.query(
                sql_str=_sql_max_time.format(
                    tbl=self._tbl_name_dpm_map_row_id_mapping_by_time
                ),
                type="df",
            ).values[0][0]

            if max_row_s1 is None:
                max_row_s1 = 0
        except Exception:
            return 0, 0.0

        try:
            max_row_s2 = self.query(
                sql_str=_sql_max_row_id.format(
                    tbl=self._tbl_name_dpm_map_row_id_mapping_by_time_hostId
                ),
                type="df",
            ).values[0][0]
            max_time_s2 = self.query(
                sql_str=_sql_max_time.format(
                    tbl=self._tbl_name_dpm_map_row_id_mapping_by_time_hostId
                ),
                type="df",
            ).values[0][0]
            if max_row_s2 is None:
                max_row_s2 = 0
        except Exception:
            return 0, 0.0

        if max_row_s1 != max_row_s2:
            raise ValueError(
                f"Inconsistency found in row_id_mapping tables. found max uid of {max_row_s1} in"
                f"dcd_map_row_id_mapping_by_time_hostId and {max_row_s2} dcd_map_row_id_mapping_by_time_hostId. Should be equal!"
            )
        if max_time_s1 != max_time_s2:
            raise ValueError(
                f"Inconsistency found in row_id_mapping tables. found max time of {max_time_s1} in"
                f"{self._tbl_name_dpm_map_row_id_mapping_by_time} and time {max_time_s2} {self._tbl_name_dpm_map_row_id_mapping_by_time_hostId}. Should be equal!"
            )

        return max_row_s1, max_time_s1

    @timing
    def create_dcd_map_row_mapping_cache(
        self, chunk_size: int = 1_000_000, initial_offset: int = 0
    ):
        """Generates two new tables, if missing, which contain a time and a time/host_id based row_id cache
        of the dcd_map table. The columns `row_id_start` and `row_id_end` are inclusive bounds and provide a
        window in the dcd_map table for that time or time/host_id key.

        This is possible because during the simulation the data is written time step by time step and as
        one block for each host_id. The used INTEGER primary key of the dcd_map is an alias to the row_id,
        which is stored as a B-Tree, allowing fast access to each entry and to interval based queries.
        See `https://www.sqlite.org/lang_createtable.html#rowid` for details.

        This method can continue after stopping mid processing by providing an initial offset. The offset must
        be the last row_id found in `dcd_map_row_id_mapping_by_time` or `dcd_map_row_id_mapping_by_time_hostId`,
        must be the same!. Use the method  `get_last_processed_uid_of_row_mapping_cache` to get the last processed
        row_id to continue processing. The method will return 0 if the tables do not exist.

        Args:
            chunk_size (int, optional): Number of rows to load with one query each select row contains (3 Integers, 2 Floats). Defaults to 1_000_000.
            offset (int, optional): Initial offset to use. Defaults to 0.

        Raises:
            ValueError: _description_
        """

        _sql = "select d.uid, d.simtime, d.hostId, d.x , d.y from dcd_map as d where d.uid > {lower_bound} and d.uid <= {upper_bound}"

        chunks = []
        time_first = -1
        time_last = -1
        load_next = True
        lower_bound = initial_offset
        upper_bound = lower_bound + chunk_size

        self.create_row_cache_tables_if_missing()
        with self.con() as _con:
            num_chunks = 0
            while load_next:
                # todo pragma
                ts = it.default_timer()
                num_chunks += 1
                sql = _sql.format(
                    _sql, lower_bound=lower_bound, upper_bound=upper_bound
                )
                data = pd.read_sql_query(sql, _con)
                logger.info(
                    f"{num_chunks}/? query uid(ROW_ID interval) {lower_bound:,d} < ROWID <= {upper_bound:,d}  chunk_size={upper_bound - lower_bound:,d} took {it.default_timer() - ts:2,.2f}s"
                )
                lower_bound = upper_bound
                upper_bound = lower_bound + chunk_size

                if data.empty:
                    # reached end of table stop processing
                    logger.info(f"reached end of table stop after {num_chunks}")
                    load_next = False
                    if len(chunks) == 0:
                        # in case  no chunks are left do not process further
                        break
                else:
                    if len(chunks) == 0:
                        # only set time_first if no older chunk is present.
                        time_first = data["simtime"].iloc[0]
                    time_last = data["simtime"].iloc[-1]
                    chunks.append(data)

                if time_first != time_last or load_next is False:
                    logger.info(f"{num_chunks}/? Process loaded chunk(s).")
                    # found more than one time process time
                    # keep last time out of processing as it might not be
                    if len(chunks) == 1:
                        data = chunks[0]
                    else:
                        data = pd.concat(chunks, axis=0)

                    # process chunk and return last time if present
                    rest_chunk = self.process_chunk(
                        data, time_first, time_last, connection=_con
                    )
                    chunks.clear()
                    time_first = -1
                    time_last = -1

                    if rest_chunk is not None:
                        time_first = rest_chunk["simtime"].iloc[0]
                        time_last = rest_chunk["simtime"].iloc[-1]
                        if time_first != time_last:
                            raise ValueError(
                                f"Rest chunk must be one time only!. Got {time_first} and {time_last}"
                            )
                        logger.info(
                            f"Add rest chunk back to chunk with {rest_chunk.shape[0]:,d} rows back to list of chunks. New start time time is {time_first}"
                        )
                        chunks.append(rest_chunk)
                else:
                    # Chunk does not contain all data for one time.
                    # Keep pulling data.
                    logger.info(
                        f"{num_chunks}/? loaded chunk(s) contains only one time value. Pull next chunk to check if there are more lines for that time value."
                    )
                    pass
        logger.info(
            f"finished processing {self.get_last_processed_uid_of_row_mapping_cache():,d} rows of dcd_map table."
        )

    @timing
    def process_chunk(
        self, data: pd.DataFrame, time_first, time_last, connection: sq.Connection
    ):
        all_rows = data.shape[0]
        rest_chunk_rows = 0
        processed_chunk_rows = all_rows
        if time_first != time_last:
            # more than one time!
            last_time_mask = data["simtime"].iloc[-1] == data["simtime"]
            rest_chunk = data[last_time_mask].copy()
            data = data[~last_time_mask].copy()
            rest_chunk_rows = rest_chunk.shape[0]
            processed_chunk_rows = data.shape[0]
        else:
            rest_chunk = None

        logger.info(
            f"process {processed_chunk_rows:,d}/{all_rows:,d} rows. Keep {rest_chunk_rows:,d} rows of time {time_last}"
        )
        mapping_by_time = (
            data.groupby("simtime")["uid"]
            .agg(["min", "max"])
            .reset_index()
            .set_axis(["simtime", "row_id_start", "row_id_end"], axis=1)
        )
        mapping_by_time_host_id = (
            data.groupby(["simtime", "hostId"])["uid"]
            .agg(["min", "max", "count"])
            .reset_index()
            .set_axis(
                ["simtime", "host_id", "row_id_start", "row_id_end", "number_of_cells"],
                axis=1,
            )
        )
        logger.info(f"save {mapping_by_time.shape[0]:,d} rows in mapping_by_time table")
        connection.executemany(
            self._tbl_insert_dpm_map_row_id_mapping_by_time, mapping_by_time.values
        )
        logger.info(
            f"save {mapping_by_time_host_id.shape[0]:,d} rows in mapping_by_time_host_id table"
        )
        connection.executemany(
            self._tbl_insert_dpm_map_row_id_mapping_by_time_hostId,
            mapping_by_time_host_id.values,
        )
        connection.commit()
        return rest_chunk

    def get_cell_count_by_host_id_over_time(self, sql_str=None) -> pd.DataFrame:
        """Return frame with [simtime, hostId, numberOfCells]

        numberOfCells is the total size of map each 'hostId' sees at each 'simtime'
        """

        _sql = "select t.simtime, t.host_id as 'hostId', t.number_of_cells as 'numberOfCells' from dcd_map_row_id_mapping_by_time_hostId as t"
        if sql_str is None:
            _sql = _sql
        else:
            _sql = sql_str
        return self.query(_sql, type="df")
