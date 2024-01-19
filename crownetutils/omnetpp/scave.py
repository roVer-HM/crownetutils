from __future__ import annotations

import contextlib
import glob
import inspect
import io
import os
import pprint as pp
import re
import signal
import sqlite3 as sq
import subprocess
import time
from typing import Any, Callable, List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from crownetutils.analysis.dpmm.dpmm_cfg import DpmmCfg, MapType
from crownetutils.omnetpp.scave_config import ScaveConfig
from crownetutils.omnetpp.sim_bound import SimBound
from crownetutils.omnetpp.sql import SqlOp
from crownetutils.utils.logging import TimeIt, logger, timing
from crownetutils.utils.parallel import run_kwargs_map


class SqlEmptyResult(Exception):
    pass


class ScaveFilter:
    """
    Build opp_scavetool filter using builder pattern.
    """

    @classmethod
    def create(cls):
        return cls()

    def __init__(self, config: ScaveConfig = None):
        if config is None:
            self._config = ScaveConfig()  # default
        else:
            self._config = config
        self._filter = []
        self._groups = 0

    def gOpen(self):
        self._filter.append("(")
        self._groups += 1
        return self

    def gClose(self):
        self._filter.append(")")
        self._groups -= 1
        if self._groups < 0:
            raise ValueError(
                f"Scave filter group mismatch. Closed one group that was not "
                f"opened: '{' '.join(self._filter)}'"
            )
        return self

    @staticmethod
    def _breaket_workaround(val):
        val_old = val
        val = val = val.replace(r"(", r"?")
        val = val = val.replace(r")", r"?")
        if val_old != val:
            print(
                "warning: using breaket workaround due to parse issue in omnetpp see #2"
            )
        return val

    def AND(self):
        self._filter.append("AND")
        return self

    def OR(self):
        self._filter.append("OR")
        return self

    def file(self, val):
        self._filter.extend(["file", "=~", val])
        return self

    def run(self, val):
        self._filter.extend(["run", "=~", val])
        return self

    def t_scalar(self):
        self._filter.extend(["type", "=~", "scalar"])
        return self

    def t_vector(self):
        self._filter.extend(["type", "=~", "vector"])
        return self

    def t_statistics(self):
        self._filter.extend(["type", "=~", "statistics"])
        return self

    def t_histogram(self):
        self._filter.extend(["type", "=~", "histogram"])
        return self

    def t_parameter(self):
        self._filter.extend(["type", "=~", "parameter"])
        return self

    def type(self, val):
        self._filter.extend(["type", "=~", val])
        return self

    def name(self, val):
        self._filter.extend(["name", "=~", self._breaket_workaround(val)])
        return self

    def module(self, val):
        self._filter.extend(["module", "=~", val])
        return self

    def runattr(self, name):
        self._filter.append(f"runattr:{name}")
        return self

    def itervar(self, name):
        self._filter.append(f"itervar:{name}")
        return self

    def config(self, name):
        self._filter.append(f"config:{name}")
        return self

    def attr(self, name):
        self._filter.append(f"attr:{name}")
        return self

    def build(self, escape=True):
        if self._groups != 0:
            raise ValueError(
                f"Scave filter group mismatch." f"opened: '{' '.join(self._filter)}'"
            )
        if escape:
            return self._config.escape(" ".join(self._filter))
        else:
            return " ".join(self._filter)

    def str(self):
        return " ".join(self._filter)


class OppSql:

    """
    Util class to access vec and sca database files.
    """

    OR = SqlOp.OR
    AND = SqlOp.AND

    def __init__(self, vec_path=None, sca_path=None):
        self._vec_path = vec_path
        self._sca_path = sca_path

    def _file(self, key):
        if key == "vec":
            return self.vec_con
        elif key == "sca":
            return self.sca_con
        else:
            raise ValueError(f"No file for key '{key}'")

    @property
    def vec_path(self):
        if self._vec_path is None:
            raise RuntimeError("vector path not set")
        return self._vec_path

    @property
    def sca_path(self):
        if self._sca_path is None:
            raise RuntimeError("scalar path not set")
        return self._sca_path

    @contextlib.contextmanager
    def vec_con(self):
        if not os.path.exists(self.vec_path):
            raise FileNotFoundError(self.vec_path)
        try:
            _vec_con = sq.connect(self.vec_path)
            yield _vec_con
        finally:
            _vec_con.close()

    @contextlib.contextmanager
    def sca_con(self):
        if not os.path.exists(self.sca_path):
            raise FileNotFoundError(self.sca_path)
        try:
            _sca_con = sq.connect(self.sca_path)
            yield _sca_con
        finally:
            _sca_con.close()

    def _query(
        self, sql_str, file="vec", type="df", **kwargs
    ) -> Union[pd.DataFrame, sq.Cursor]:
        sql_file = self._file(file)
        logger.debug(f"execute sql on db {file}: {sql_str}")
        with sql_file() as _con:
            if type == "df":
                return pd.read_sql_query(sql_str, _con, **kwargs)
            elif type == "cursor":
                return _con.execute(sql_str, **kwargs)
            else:
                raise RuntimeError("Expected df or cursor as type")

    def _write(self, sql_str, file="vec", **kwargs) -> None:
        sql_file = self._file(file)
        with sql_file() as _con:
            _con.execute(sql_str, **kwargs)
            _con.commit()

    def query_vec(self, sql_str, type="df", **kwargs):
        return self._query(sql_str, file="vec", type=type, **kwargs)

    def query_sca(self, sql_str, type="df", **kwargs):
        return self._query(sql_str, file="sca", type=type, **kwargs)

    def write_sca(self, sql_str, **kwargs):
        return self._write(sql_str, file="sca", **kwargs)

    @staticmethod
    def _to_sql(obj: Union[str, SqlOp], table, column, prefix="", suffix=""):
        """
        Transform string or sql operator into sql string
        """
        if type(obj) == SqlOp:
            return f"{prefix} {obj.apply(table, column)} {suffix}"
        elif type(obj) == str:
            if "%" in obj:
                return f"{prefix} {table}.{column} like '{obj}' {suffix}"
            else:
                return f"{prefix} {table}.{column} = '{obj}' {suffix}"
        else:
            raise ValueError("expected SqlOp or string")

    def check_if_index_exists(self, print_info: bool = False) -> bool:
        """Check for indexes on 'vectorId' and 'eventNumber' on vectorData table.

        CREATE INDEX IF NOT EXISTS vectorData_vectorId_index ON vectorData (vectorId);
        CREATE INDEX IF NOT EXISTS vectorData_eventNumber_index ON vectorData (eventNumber);

        CREATE INDEX IF NOT EXISTS vector_moduleName_index ON vector (moduleName);
        CREATE INDEX IF NOT EXISTS vector_vectorName_index ON vector (vectorName);
        """
        except_indices = [
            ("vector", "vector_moduleName_index", "moduleName"),
            ("vector", "vector_vectorName_index", "vectorName"),
            ("vectorData", "vectorData_vectorId_index", "vectorId"),
            ("vectorData", "vectorData_eventNumber_index", "eventNumber"),
        ]
        with self.vec_con() as c:
            idx_info = []
            found_at = []
            for table in ["vectorData", "vector"]:
                indexes = c.execute(f"PRAGMA index_list({table});").fetchall()
                if len(indexes) <= 0:
                    continue
                index_names = [r[1] for r in indexes]
                for idx, i_name in enumerate(index_names):
                    index_infos = c.execute(f"PRAGMA index_info({i_name});").fetchall()
                    if len(index_infos) == 0:
                        idx_info.append((table, index_names[idx], None))
                    else:
                        idx_info.append((table, index_names[idx], index_infos[0][2]))

            for expected in except_indices:
                found = False
                for idx, i in enumerate(idx_info):
                    found = False
                    if expected == i:
                        found_at.append((idx, True))
                        found = True
                        break
                    elif expected[0] == i[0] and expected[2] == i[2]:
                        found_at.append((idx, False))
                        found = True
                        break
                if not found:
                    found_at.append((-1, False))

            if print_info:
                print(
                    "expected indices:\nnum | table | index_name | column | match(full/partial)\n "
                )
                for idx, e in enumerate(except_indices):
                    m = found_at[idx]
                    if m[0] >= 0:
                        if m[1]:
                            m = f"match at {idx}"
                        else:
                            m = f"partial match at {idx}"
                    else:
                        m = "not found!"
                    print(f"{idx} | {e[0]} |  {e[1]} | {e[2]} |  {m}")
                print("-" * 80)
                print(
                    "found indices:\nnum | table | index_name | column | match(full/partial)\n "
                )
                for idx, e in enumerate(idx_info):
                    print(f"{idx} | {e[0]} | {e[1]} | {e[2]}")

            if all([i[0] >= 0 for i in found_at]):
                return True
            return False

    def append_index_if_missing(self):
        """Append missing indexes to vector database. Will increase memory footprint by about 1/3.

        CREATE INDEX IF NOT EXISTS vectorDataIndex ON vectorData (vectorId);
        CREATE INDEX IF NOT EXISTS vectorDataIndexEventNumber ON vectorData (eventNumber);

        CREATE INDEX IF NOT EXISTS moduleNameIndex ON vector (moduleName);
        CREATE INDEX IF NOT EXISTS vectorNameIndex ON vector (vectorName);
        """
        index_sql = [
            "CREATE INDEX IF NOT EXISTS vectorData_vectorId_index ON vectorData (vectorId);",
            "CREATE INDEX IF NOT EXISTS vectorData_eventNumber_index ON vectorData (eventNumber);",
            "CREATE INDEX IF NOT EXISTS vector_moduleName_index ON vector (moduleName);",
            "CREATE INDEX IF NOT EXISTS vector_vectorName_index ON vector (vectorName);",
        ]
        timer = TimeIt()
        with timer:
            with self.vec_con() as c:
                for s in index_sql:
                    logger.debug(f"execute: {s}")
                    ret = c.execute(s).fetchall()
                    logger.debug(
                        f"Done in {timer.round_str()} with return value: {ret}"
                    )
        logger.debug(f"indices create in {timer.str()}")

    def vector_exists(
        self,
        module_name: SqlOp | str,
        vector_name: SqlOp | str,
    ) -> bool:
        """Check if at least one vector exists in given module_name"""
        ret = self.find_vector_name(module_name, vector_name, raise_on_missing=False)
        return ret is not None

    def find_vector_name(
        self,
        module_name: SqlOp | str | None,
        vector_name: str | List[str],
        find_first: bool = True,
        raise_on_missing: bool = True,
    ):
        """Check if a vector with a given name exists. If a list of vector names is given
        they are checked in that order. If 'find_first' ist true the first vector name that
        returns a none-empty vec_info data frame is returned. If no vector_name exists None
        is returned or an ValueError is raised if 'raise_on_missing' is true."""

        if isinstance(vector_name, str):
            vector_name = [vector_name]

        for vec_n in vector_name:
            df = self.vec_info(module_name, vec_n)
            if not df.empty and find_first:
                return vec_n
            elif df.empty and not find_first:
                # all vectors must be present.
                break

        # no vectors found at this point
        if raise_on_missing:
            raise ValueError(
                f"No or not all vectors found with names: {', '.join(vector_name)} at "
                f"module: {module_name}"
            )

        return None

    def hist_info(
        self,
        module_name: SqlOp | str | None = None,
        stat_name: SqlOp | str | None = None,
        stat_ids: List[int] | None = None,
        run_id: int = 1,
        cols: List[str] | None = None,
        **kwargs,
    ):
        """Query statistic table for histogram information"""
        if cols is None:
            cols = "*"  # select all columns
        else:
            cols = ", ".join([f"v.{c}" for c in cols])

        if all(i is not None for i in [module_name, stat_name]):
            _sql = f"select {cols} from statistic v where v.runId = '{run_id}' and v.isHistogram == 1"
            _sql += self._to_sql(module_name, "v", "moduleName", "and")
            _sql += self._to_sql(stat_name, "v", "statName", "and")
        elif stat_ids is not None:
            _id_str = [str(i) for i in stat_ids]
            _sql = f"select {cols} from statistic v where v.runId = '{run_id}' and v.isHistogram == 1"
            _sql += f" and v.statId in ({', '.join(_id_str)})"
        else:
            raise ValueError(
                "expected either moduleName and vectorName or list of vector ids"
            )
        df = self.query_sca(_sql, type="df", **kwargs)
        return df

    def hist_data(
        self,
        module_name: SqlOp | str | None = None,
        stat_name: SqlOp | str | None = None,
        stat_ids: List[int] | None = None,
        run_id: int = 1,
        **kwargs,
    ):
        if all(i is not None for i in [module_name, stat_name]):
            _sql = f"select v.*, b.lowerEdge, b.binValue from statistic v  inner join histogramBin as b on  v.statId == b.statId where v.runId = '{run_id}'"
            _sql += self._to_sql(module_name, "v", "moduleName", "and")
            _sql += self._to_sql(stat_name, "v", "statName", "and")
        elif stat_ids is not None:
            _id_str = [str(i) for i in stat_ids]
            _sql = f"select v.*, b.lowerEdge, b.binValue from statistic v inner join histogramBin as b on  v.statId == b.statId where v.runId = '{run_id}'"
            _sql += f" and v.statId in ({', '.join(_id_str)})"
        else:
            raise ValueError(
                "expected either moduleName and vectorName or list of vector ids"
            )
        df = self.query_sca(_sql, type="df", **kwargs)
        if not df.empty:
            df = df.sort_values(["statId", "lowerEdge"])
            df["upperEdge"] = df["lowerEdge"].shift(-1)
            df["_statId"] = df["statId"].shift(-1)
            df[df["statId"] == df["_statId"]]
            df = df[~df["_statId"].isna()]
            df["bin_size"] = np.abs(df["upperEdge"] - df["lowerEdge"])
            df["inf_bin"] = df["bin_size"] == np.inf
        return df

    def vec_info(
        self,
        module_name: SqlOp | str | None = None,
        vector_name: SqlOp | str | None = None,
        vector_ids: List[int] | None = None,
        run_id: int = 1,
        cols: List[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Query vector base information contained in xxx.vec files. This will access the 'vector'
        table from the xxx.vec file provided in the constructor.

        Args:
            moduleName (SqlOp|str, optional): column selector for moduleName column. Defaults to None.
            vectorName (SqlOp|str, optional): column selector for vectorName. Defaults to None.
            ids (List[int], optional): vector id list. Defaults to None.
            runId (int, optional): Defaults to 1.
            cols (List[str], optional): List of columns to return.
                Possible columns: [vectorId, runId, moduleName, vectorName,
                vectorCount, vectorMin, vectorMax, vectorSum, vectorSumSqr,
                startEventNum, endEventNum, startSimtimeRaw, endSimtimeRaw].
                If None, return all columns. Defaults to None.
            human_names (List[str], optional):

        Raises:
            ValueError: must provide module/vector names or ids. If both are provided
                        the former will be used.

        Returns:
            pd.Dataframe:
        """
        if cols is None:
            cols = "*"  # select all columns
        else:
            cols = ", ".join([f"v.{c}" for c in cols])

        if all(i is not None for i in [module_name, vector_name]):
            _sql = f"select {cols} from vector v where v.runId = '{run_id}' "
            _sql += self._to_sql(module_name, "v", "moduleName", "and")
            _sql += self._to_sql(vector_name, "v", "vectorName", "and")
        elif vector_ids is not None:
            _id_str = [str(i) for i in vector_ids]
            _sql = f"select {cols} from vector v where v.runId = '{run_id}' "
            _sql += f" and v.vectorId in ({', '.join(_id_str)})"
        else:
            raise ValueError(
                "expected either moduleName and vectorName or list of vector ids"
            )

        # print(_sql)
        df = self.query_vec(_sql, type="df", **kwargs)
        return df

    def vec_ids(
        self,
        module_name: SqlOp | str,
        vector_name: SqlOp | str,
        run_id: int = 1,
        **kwargs,
    ) -> List[int]:
        """VectorId list for module/vector names.

        Args:
            moduleName (SqlOp|str, optional): column selector for moduleName column. Defaults to None.
            vectorName (SqlOp|str, optional): column selector for vectorName. Defaults to None.
            ids (List[int], optional): [description]. vector id list. Defaults to None.
            runId (int, optional): [description]. Defaults to 1.

        Returns:
            List[int]:
        """
        return self.vec_info(
            module_name, vector_name, run_id=run_id, cols=["vectorId"], **kwargs
        )["vectorId"].to_list()

    def vec_merge_on(
        self,
        module_name: Union[None, str, SqlOp] = None,
        vector_name: Union[None, str, SqlOp] = None,
        runId=1,
        time_resolution=1e12,
        **kwargs,
    ):
        """
        Merge
        """

        _info = self.vec_info(
            module_name, vector_name, runId, cols=("vectorId", "vectorName"), **kwargs
        )
        _ids = _info["vectorId"].to_list()
        _id_map = (
            _info.reset_index(drop=True)
            .set_index(keys=["vectorId"])
            .to_dict()["vectorName"]
        )

        v_str = ", ".join([f'v{id}.value as "{name}"' for id, name in _id_map.items()])
        sub_q = f"(select * from vectorData as v where v.vectorId = {_ids[0]}) as v{_ids[0]}"
        joins = "\n".join(
            f"    inner join (select * from vectorData as v where v.vectorId = {id}) as v{id} on v{_ids[0]}.simtimeRaw = v{id}.simtimeRaw"
            for id in _ids[1:]
        )
        _sql = f"select v{_ids[0]}.simtimeRaw, v{_ids[0]}.eventNumber, {v_str} \n  from {sub_q} \n{joins}"
        df = self.query_vec(_sql, type="df", **kwargs)
        if time_resolution is not None and "simtimeRaw" in df.columns:
            df["simtimeRaw"] = df["simtimeRaw"] / time_resolution
            df = df.rename(columns={"simtimeRaw": "time"})
        return df

    def parameter_data(
        self,
        module_name: Union[None, str, SqlOp] = None,
        scalar_name: Union[None, str, SqlOp] = None,
        ids: Union[None, List[int]] = None,
        cols: set = ("paramId", "paramName", "paramValue"),
        runId=1,
    ) -> pd.DataFrame:
        cols = ", ".join([f"s.{c}" for c in cols])

        if module_name is not None and scalar_name is not None:
            _sql = f"select {cols} from parameter as s where "
            _sql += self._to_sql(module_name, "s", "moduleName")
            _sql += self._to_sql(scalar_name, "s", "paramName", "and")
        elif ids is not None:
            _ids_str = ",".join(str(i) for i in ids)
            _sql = f"select {cols} from parameter as s where s.paramId in ({_ids_str})"
        else:
            raise ValueError(
                "provide either module and parameter name or list of parameter ids"
            )

        # print(_sql)
        df = self.query_sca(_sql)
        return df

    def sca_data(
        self,
        module_name: Union[None, str, SqlOp] = None,
        scalar_name: Union[None, str, SqlOp] = None,
        ids: Union[None, List[int]] = None,
        cols: set = ("scalarId", "scalarName", "scalarValue"),
        runId=1,
    ) -> pd.DataFrame:
        cols = ", ".join([f"s.{c}" for c in cols])

        if module_name is not None and scalar_name is not None:
            _sql = f"select {cols} from scalar as s where "
            _sql += self._to_sql(module_name, "s", "moduleName")
            _sql += self._to_sql(scalar_name, "s", "scalarName", "and")
        elif ids is not None:
            _ids_str = ",".join(str(i) for i in ids)
            _sql = f"select {cols} from scalar as s where s.scalarId in ({_ids_str})"
        else:
            raise ValueError(
                "provide either module and scalar name or list of scalar ids"
            )

        # print(_sql)
        df = self.query_sca(_sql)
        return df

    def get_run_attr(self, name, full_match: bool = True, run_id=1):
        if full_match:
            _sql = f'select * from runAttr as r where r.attrName == "{name}" and r.runId=={run_id}'
        else:
            _sql = f'select * from runAttr as r where r.attrName like "%{name}%" and r.runId=={run_id}'

        df = self.query_sca(_sql)
        if not df.empty:
            return df.iloc[0]["attrValue"]
        return None

    def get_run_parameter(
        self, module_name: SqlOp, name, full_match: bool = True, run_id=1
    ):
        m = self._to_sql(module_name, table="p", column="moduleName")
        if full_match:
            _sql = f'select * from parameter as p where {m} and p.paramName =="{name}" and p.runId=={run_id}'
        else:
            _sql = f'select * from parameter as p where {m} and p.paramName like "%{name}%" and p.runId=={run_id}'
        df = self.query_sca(_sql)
        return df

    def get_run_config(self, name: str, full_match: bool = True, run_id=1):
        if full_match:
            _sql = f'select *, Min(r.configOrder) from runConfig as r where r.configKey == "{name}" and r.runId=={run_id}'
        else:
            _sql = f'select *, Min(r.configOrder) from runConfig as r where r.configKey like "%{name}%" and r.runId=={run_id}'

        df = self.query_sca(_sql)
        if not df.empty:
            val = df.iloc[0]["configValue"]
            if isinstance(val, str):
                val = val.replace('"', "")
            return val
        else:
            return None

    def get_app_config(self):
        mm = ["misc", "pNode", "vNode"]
        ret = {}
        for m in mm:
            _r = {}
            num_apps = self.get_run_config(f"*.{m}[*].numApps", full_match=True)
            if num_apps is not None:
                num_apps = int(num_apps)
                for i in range(num_apps):
                    _r[i] = self.get_run_config(
                        f"*.{m}[*].app[{i}].typename", full_match=True
                    )
                ret[m] = _r
        return ret

    def get_all_run_config(self, run_id=1, order="ASC"):
        df = self.query_sca(
            f"select r.configOrder, r.configKey, r.configValue from runConfig as r where r.runId=={run_id} ORDER By r.configOrder {order}"
        )
        df["configValue"] = df["configValue"].apply(lambda x: x.strip())
        return df

    def extract_ini_file(self, path: str | None, run_id=1) -> List[str] | None:
        df = self.get_all_run_config(run_id=run_id, order="DESC")
        lines = []
        lines.extend(
            [
                "[General]\n",
                "\n",
                "\n",
                f"[Config {self.get_run_attr('configname')}]\n\n",
            ]
        )
        for i in range(df.shape[0]):
            if df.iloc[i]["configKey"] != "extends":
                lines.append(f"{df.iloc[i]['configKey']}={df.iloc[i]['configValue']}\n")

        if path is None:
            return lines
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            return None

    def vec_data_paginate(
        self,
        module_name: SqlOp | str | None = None,
        vector_name: SqlOp | str | None = None,
        ids: List[int] | pd.DataFrame | None = None,
        runId: int = 1,
        vec_ids_per_page: int = 5,
        columns: List[str] = ("vectorId", "simtimeRaw", "value"),
        order_by: List[str] = (),
        value_name: str = "value",
        time_slice: slice = slice(None),
        time_resolution=1e12,
        index: List[str] | None = None,
        index_sort: bool = True,
        drop: str | List[str] | None = None,
        **kwargs,
    ):
        _loc = locals()
        sig = inspect.signature(self.vec_data_paginate)
        sig = [
            i[0]
            for i in list(sig.parameters.items())
            if i[0]
            not in ["module_name", "vector_name", "ids", "vec_ids_per_page", "kwargs"]
        ]
        vec_ids_sig = {k: _loc[k] for k in sig}
        if module_name is not None and vector_name is not None:
            _ids = self.vec_ids(module_name, vector_name)
        elif type(ids) == pd.DataFrame:
            _ids = ids["vectorId"].unique()
            if "vectorId" not in columns:
                columns = [*columns, "vectorId"]
        else:
            _ids = ids

        data = []
        pages = int(np.ceil(len(_ids) / vec_ids_per_page))
        for i in range(0, len(_ids), vec_ids_per_page):
            print(f"query page: {int(i/vec_ids_per_page)}/{pages}")
            page_ids = _ids[i : min(i + 5, len(_ids) - 1)]
            o = self.vec_data(ids=page_ids, **vec_ids_sig, **kwargs)
            data.append(o)
        return pd.concat(data, axis=1)

    @timing
    def vec_data(
        self,
        module_name: SqlOp | str | None = None,
        vector_name: SqlOp | str | None = None,
        ids: List[int] | pd.DataFrame | None = None,
        runId: int = 1,
        columns: List[str] = ("vectorId", "simtimeRaw", "value"),
        order_by: List[str] = (),
        value_name: str = "value",
        time_slice: slice = slice(None),
        time_resolution=1e12,
        index: List[str] | None = None,
        index_sort: bool = True,
        drop: str | List[str] | None = None,
        **kwargs,
    ):
        if module_name is not None and vector_name is not None:
            _ids = self.vec_ids(module_name, vector_name)
        elif type(ids) == pd.DataFrame:
            _ids = ids["vectorId"].unique()
            if "vectorId" not in columns:
                columns = [*columns, "vectorId"]
        else:
            _ids = ids

        _ids = ", ".join([str(i) for i in _ids])
        columns = ", ".join([f"v_data.{c}" for c in columns])
        order_by = ", ".join([f"v_data.{c}" for c in order_by])
        if time_slice != slice(None):
            if time_slice.start is None:
                _time = f" and v_data.simTimeRaw == {time_slice.stop * time_resolution}"
            elif time_slice.start is not None and time_slice.stop is not None:
                _time = f" and v_data.simTimeRaw >= {time_slice.start * time_resolution} and v_data.simTimeRaw <= {time_slice.stop * time_resolution}"
        else:
            _time = ""
        _sql = f"select {columns} from vectorData v_data where v_data.vectorId in ({_ids}) {_time}"
        if len(order_by) > 0:
            _sql = f"{_sql}  ORDER BY {order_by}"

        # print(_sql)
        df = self.query_vec(_sql, type="df", **kwargs)
        if time_resolution is not None and "simtimeRaw" in columns:
            df["simtimeRaw"] = df["simtimeRaw"] / time_resolution
            df = df.rename(columns={"simtimeRaw": "time"})

        if type(ids) == pd.DataFrame:
            df = pd.merge(df, ids, how="left", on=["vectorId"])

        df = df.rename(columns={"value": value_name})

        if index is not None:
            df = df.set_index(index)
            if index_sort:
                df = df.sort_index()

        if df.shape[0] == 0:
            logger.info("Query returned empty DataFrame.")
            logger.debug(_sql)

        if drop is not None:
            drop = [drop] if isinstance(drop, str) else drop
            df = df.drop(columns=drop, errors="ignore")

        return df


class HostIdMap:
    """Provides access to OMNeT++ module name to module ids. The caller must check if the provided id exists.
    If not and this is an acceptable state for the application a dummy unique identifier is provided.
    Key: str of module name
    Value: module identifier (or dummy id)
    """

    def __init__(self, id_map) -> None:
        self.id_map: dict = id_map
        self.key_set = set(list(id_map.keys()))

    def _next_id_peek(self):
        if len(self.key_set) == 0:
            return 0
        else:
            return max(self.key_set) + 1

    def _next_id(self):
        i = self._next_id_peek()
        self.key_set.add(i)
        return i

    def __getitem__(self, key):
        return self.id_map[key]

    def get_dummy_id(self, key):
        if key not in self.id_map:
            i = self._next_id()
            logger.warn(f"module '{key}' not found in id_map. Provide dummy id {i}")
            self.id_map[key] = i
        return self.id_map[key]

    def __contains__(self, key):
        return key in self.id_map


class ModuleMatcher:
    """Extract module name, vector index and module id (OMNeT interval node identifier) from
    OMNeT++ module path. The module name ist the OMNeT++ module path to the module which
    contains the '@NetworkNode` marker in the OMNeT++ ned file definition. In combination with
    the `hostId` statistic a match between the module id and the network path is possible.

    In case of missing `hostId` statistics a dummy unique identifier is used. See `HostIdMap`
    for more information.
    """

    def __init__(self, sql) -> None:
        self.sql: CrownetSql = sql
        self._module_map = None

    @property
    def module_map(self):
        if self._module_map is None:
            self._module_map = self.sql.module_to_host_ids()
        return self._module_map

    def _match_host(self, x):
        if _m := self.sql._host_index_regex.match(x):
            return f'{_m.groupdict()["type"]}[{_m.groupdict()["hostIdx"]}]'
        raise ValueError(
            f"given moduelName '{x}' does not match vector index regex {self.sql._host_index_regex}"
        )

    def _match_host_id(self, x):
        if _m := self.sql._host_id_regex.match(x):
            if _m.groupdict()["host"] in self.module_map:
                return self.module_map[_m.groupdict()["host"]]
            elif _m.groupdict()["type"] in ["eNB", "gNB"]:
                return self.sql._host_index_regex.match(x).groupdict()["hostIdx"]
            else:
                return self.module_map.get_dummy_id(_m.groupdict()["host"])
        raise ValueError(
            f"given moduleName '{x}' does match module regex {self._host_id_regex}"
        )

    def _match_vector_idx(self, x):
        if _m := self.sql._host_index_regex.match(x):
            return int(_m.groupdict()["hostIdx"])
        raise ValueError(
            f"given moduelName '{x}' does not match vector index regex {self._host_index_regex}"
        )

    def get_host(self, s: pd.Series) -> pd.Series:
        return s.apply(lambda x: self._match_host(x))

    def get_host_id(self, s: pd.Series) -> pd.Series:
        return s.apply(lambda x: self._match_host_id(x))

    def get_vector_index(self, s: pd.Series) -> pd.Series:
        return s.apply(lambda x: self._match_vector_idx(x))


def append_host_id(
    df: pd.DataFrame, sql: CrownetSql, col_name: str = "moduleName"
) -> pd.DataFrame:
    if col_name not in df.columns:
        raise ValueError(f"expected {col_name} in frame got {df.columns}")
    df["host_id"] = sql.module_matcher.get_host_id(df[col_name])
    return df


def append_host_vector_index(
    df: pd.DataFrame, sql: CrownetSql, col_name: str = "moduleName"
) -> pd.DataFrame:
    if col_name not in df.columns:
        raise ValueError(f"expected {col_name} in frame got {df.columns}")
    df["vec_idx"] = sql.module_matcher.get_vector_index(df[col_name])
    return df


class CrownetSql(OppSql):
    v_app_receiving = OppSql.OR(
        [
            "rcvdPkLifetime:vector",
            "rcvdPkSeqNo:vector",
            "rcvdPkHostId:vector",
            "packetReceived:vector(packetBytes)",
        ]
    )
    v_app_sending = OppSql.OR(
        ["packetSent:vector(packetBytes)", "packetCreated:vector(packetBytes)"]
    )

    _dtypes = {"host": "str", "hostId": np.int32, "vecIdx": np.int32}

    _pos_x = "posX:vector"
    _pos_y = "posy:vector"

    _module_vectors = ["misc", "pNode", "vNode"]
    _all_modules = [*_module_vectors, "eNB", "gNB"]

    _vehicle = "%s.misc[%d]"
    _vehicle_app = "%s.misc[%d].app[%d]"

    @property
    def module_vectors(self) -> List[str]:
        if self.dpmm_cfg is None:
            return self._module_vectors
        else:
            return self.dpmm_cfg.module_vectors

    @property
    def network(self) -> str:
        if self.dpmm_cfg is None:
            return self._network
        else:
            return self.dpmm_cfg.network_name

    @classmethod
    def from_dpmm_cfg(cls, cfg: DpmmCfg):
        return cls(
            vec_path=cfg.vec_path(),
            sca_path=cfg.sca_path(),
            network=cfg.network_name,
            dpmm_cfg=cfg,
        )

    def __init__(
        self,
        vec_path=None,
        sca_path=None,
        network="World",
        dpmm_cfg: DpmmCfg | None = None,
    ):
        super().__init__(vec_path=vec_path, sca_path=sca_path)
        self.dpmm_cfg = dpmm_cfg
        self._network = network
        self.module_names = self.OR(
            [f"{self.network}.{i}[%]" for i in self.module_vectors]
        )
        self._host_id_regex = re.compile(
            f"(?P<host>^{self.network}\.(?P<type>{'|'.join(self._all_modules)})\[\d+\]).*"
        )
        self._host_index_regex = re.compile(
            f"^{self.network}\.(?P<type>{'|'.join(self._all_modules)})\[(?P<hostIdx>\d+)\].*"
        )
        self._module_id_map: HostIdMap = None
        self.module_matcher: ModuleMatcher = ModuleMatcher(self)

    def host_ids(self, module_name: Union[None, str, SqlOp] = None):
        """
        Return hostIds of all vector nodes present in simulation (misc, pNode, vNode)
        """
        module_name = self.module_names if module_name is None else module_name

        _sql: pd.DataFrame = f"select s.moduleName, s.scalarValue from scalar as s where \n  {self._to_sql(module_name, 's', 'moduleName')} \n  AND s.scalarName = 'hostId:last'"
        _df = self.query_sca(sql_str=_sql)
        _df["scalarValue"] = pd.to_numeric(_df["scalarValue"], downcast="integer")
        _df = _df.reset_index(drop=True).set_index(keys="scalarValue")
        _df = _df.to_dict()["moduleName"]
        return _df

    def sim_bound(self):
        # get simulation bound offset
        offset = self.sca_data(
            module_name=f"{self.network}.coordConverter",
            scalar_name=self.OR(["simOffsetX:last", "simOffsetY:last"]),
        )["scalarValue"].to_numpy()

        # get simulation bound (width, height). Lower left point [0, 0] + offset
        bound = self.sca_data(
            module_name=f"{self.network}.coordConverter",
            scalar_name=self.OR(["simBoundX:last", "simBoundY:last"]),
        )["scalarValue"].to_numpy()

        try:
            bbox = (
                self.vec_data(
                    module_name=f"{self.network}.coordConverter",
                    vector_name="simBBox:vector",
                )["value"]
                .to_numpy()
                .reshape((2, 2))
            )
        except Exception:
            logger.warn(
                f"cannot access simBBox:vector. Assume zero sim_offset [[0,0],[{bound[0]},{bound[1]}]"
            )
            bbox = np.array([[0.0, 0.0], [bound[0], bound[1]]])

        b = SimBound(offset=offset, sim_bbox=bbox)
        return b

    @property
    def sim_time_limit(self):
        return self.get_run_config("sim-time-limit")

    @property
    def vadere_scenario(self):
        return self.get_run_config("vadereScenarioPath", full_match=False)

    def module_to_host_ids(self):
        if self._module_id_map is None:
            m = self.OR([f"{self.network}.{i}[%]" for i in self.module_vectors])
            _host_ids = self.host_ids(module_name=m)
            self._module_id_map = HostIdMap({v: k for k, v in _host_ids.items()})
        return self._module_id_map

    def debug_load_host_id_map_from_data(self):
        if len(self.module_to_host_ids().id_map) == 0:
            logger.warning(
                "no hostId map found create one from zero send delay. And append it to sca DB!"
            )
            vec_names = {
                "rcvdPkHostId:vector": dict(name="srcHostId", dtype=np.int32),
                "rcvdPkLifetime:vector": dict(name="delay", dtype=np.float32),
            }
            vec_data = self.vec_data_pivot(
                module_name=self.m_app0(),
                vector_name_map=vec_names,
                append_index=["srcHostId"],
            ).sort_index()
            _d = vec_data["delay"] == 0.0
            _dummy_to_host = {v: k for k, v in self.module_to_host_ids().id_map.items()}
            vec_data: pd.DataFrame = (
                vec_data.reset_index()
                .drop(columns=["eventNumber", "time"])[_d.values]
                .drop_duplicates()
            )
            insert = []
            for _, row in vec_data.iterrows():
                insert.append(
                    f"(1, '{_dummy_to_host[row['hostId']]}', 'hostId:last', {row['srcHostId']})"
                )

            insert = ",\n".join(insert)
            sql_str = f"INSERT INTO scalar (runId, moduleName, scalarName, scalarValue) VALUES \n {insert};"

            self.write_sca(sql_str)

            # update id map
            self._module_id_map = None
            self.module_to_host_ids()
            self.module_matcher = ModuleMatcher(self)

    def create_bonnmotion_trace(
        self,
        module_name: Union[SqlOp, str, None] = None,
        ids: pd.DataFrame | None = None,
        time_slice: slice = slice(None),
        path: str | None = None,
    ) -> List[str] | None:
        def _line_to_str(line):
            return f"{line['time']} {line['x']} {line['y']}"

        lines = []
        pos = self.node_position(
            module_name=module_name,
            ids=ids,
            time_slice=time_slice,
            apply_offset=False,
            bottom_left_origin=False,
            cols=["time", "hostId", "x", "y"],
        )
        for host, _df in pos.groupby(["hostId"]):
            lines.append(" ".join(_df.apply(_line_to_str, axis=1)))

        if path is not None:
            with open(path, "w", encoding="utf-8") as fd:
                fd.writelines(lines)
        else:
            return lines

    def enb_position(
        self,
        module_name: Union[SqlOp, str, None] = None,
        ids: pd.DataFrame | None = None,
        time_slice: slice = slice(None),
        epsg_code_base: str | None = None,
        epsg_code_to: str | None = None,
        apply_offset: bool = True,
        bottom_left_origin: bool = True,
        cols: tuple = ("time", "hostId", "host", "x", "y"),
    ):
        if module_name is None:
            module_name = self.m_enb()
        return self.node_position(
            module_name=module_name,
            ids=ids,
            time_slice=time_slice,
            epsg_code_base=epsg_code_base,
            epsg_code_to=epsg_code_to,
            apply_offset=apply_offset,
            bottom_left_origin=bottom_left_origin,
            cols=cols,
        )

    def get_resource_sharing_domains(
        self,
        ids_only: bool = False,
        epsg_code_base: str | None = None,
        epsg_code_to: str | None = None,
        apply_offset: bool = True,
        bottom_left_origin: bool = True,
    ) -> pd.Series | pd.DataFrame:
        module_name = self.m_enb()
        rsd = self.sca_data(
            module_name=module_name.append_path(".cellularNic.phy"),
            scalar_name="macNodeId:last",
            cols=("modulename", "scalarValue"),
        )
        if ids_only:
            r = rsd["scalarValue"].astype(int)
            r.name = "rsd_id"
            return r

        rsd = rsd.set_axis(["hostId", "rsd_id"], axis=1)
        rsd["rsd_id"] = rsd["rsd_id"].astype(int)
        p = re.compile(r".*\[(\d+)\].*")
        rsd["hostId"] = rsd["hostId"].apply(lambda x: int(p.match(x).groups()[0]))
        enb = self.enb_position(
            module_name=self.m_enb(),
            epsg_code_base=epsg_code_base,
            epsg_code_to=epsg_code_to,
            apply_offset=apply_offset,
            bottom_left_origin=bottom_left_origin,
            cols=("hostId", "host", "x", "y"),
        )
        enb["hostId"] = enb["hostId"].astype(int)
        if rsd.empty:
            logger.warning(
                f"scalar name of rsd_id not found. use enb vector index +1 as rsd_id"
            )
            enb["rsd_id"] = enb["hostId"] + 1
        else:
            enb = enb.merge(rsd, how="inner", on="hostId")
        return enb

    def node_position(
        self,
        module_name: Union[SqlOp, str, None] = None,
        ids: pd.DataFrame | None = None,
        time_slice: slice = slice(None),
        epsg_code_base: str | None = None,
        epsg_code_to: str | None = None,
        apply_offset: bool = True,
        bottom_left_origin: bool = True,
        cols: tuple = ("time", "hostId", "host", "vecIdx", "x", "y"),
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        """
        Get node position data in cartesian coordinates.

        By default the columns vectorId1 and vectorId2 are dropped. They correspond
        to the x and y values respectively. If epsg_code is present the `geomentry`
        column is added.

        Use either the module_name selector (sql operator or simple string)
        or provide am `ids` data frame with columns ['x', 'y'] which corresponds
        to vectorIds for posX and posY respectively. Caller must ensure that each
        row in the ids frame corresponse to one host. Use `ids` to filter based on
        hosts.

        For time filter use time_slice to specify a closed interval. Both start and stop
        value will be matched. [a, b] --> a <= x <= b

        If `epsg_code` is given the method returns a GeoDataFrame with an additional `geometry`
        column and a coordinate reference system defined by the epsg_code.
        Frequent values:
          * UTM ZONE 32N (most of Germany): `EPSG:32632` (cartesian, in meter)
          * OpenStreetMap, GoogleMaps, WSG84 Pseudo Mercartor: `EPSG:3857` (cartesian, in meter)
          * WGS84 (World Geodetic System 1984 used in GPS): `EPSG:4326 (lat/lon, in degree)

        If apply_offset add the offset used in the simulation to the coordinates. This
        is needed if any transformation should be applied.

        Examples:
        1) all host from all vectors (misc[*], pNode[*], vNode[*]) between 12.5s and 90.0s

        df = node_pos(time_slice=slice(12.5, 90.0))      # defaults to all

        2) Only misc and pNodes with base UTM ZONE 32N (default for sumo based simulations),
           projected to the Pseydo mercartor projection used by Goolge and Open Streets Map.
           Use the 'or' operator to select both misc and pNode vectors. Ensure that the
           full path is correct.

        gdf = node_pos(
            module_name=SqlOp.OR(["World.misc[%]", World.pNode[%]])
            epsg_code_base=32632, epsg_code_to="EPSG:3857"
        )

        3) Only select positions for selected host (may come from different vectors)
           In this example 3 host are selected where the vectorIds=[5, 12, 14]  are
           x postion vectors and [9, 44, 18] are y position vectors for 3 host.

        ids = pd.DataFrame([[5,9],[12,44],[14, 18]], columns=["x", "y"])
        df = node_pos(ids=ids)

        """
        module_name = self.module_names if module_name is None else module_name
        if ids is None:
            _x = self.vector_ids_to_host(
                module_name=module_name, vector_name="posX:vector"
            )
            _y = self.vector_ids_to_host(
                module_name=module_name, vector_name="posY:vector"
            )
        else:
            _x = self.vector_ids_to_host(vector_ids=ids["x"].unique())
            _y = self.vector_ids_to_host(vector_ids=ids["y"].unique())
            raise ValueError("provide either module_name or ids data frame")

        _x = self.vec_data(ids=_x, value_name="x", time_slice=time_slice)

        _y = self.vec_data(ids=_y, value_name="y", time_slice=time_slice)
        df = pd.merge(
            _x,
            _y,
            how="outer",
            on=["time", "hostId", "host", "vecIdx"],
            suffixes=["1", "2"],
        )
        if df["x"].hasnans or df["y"].hasnans:
            print("warning: host positions are inconsistent")

        bound = self.sim_bound()

        # convert to bottom-left origin and remove offset used during the simulation
        if bottom_left_origin:
            df["y"] = (
                bound.sim_bbox[1][1] - df["y"]
            )  # move from top-left-orig to bottom-left-orig
            df["x"] = df["x"] + bound.sim_bbox[0][0]
        # nothing to do for x, only y-axis needs conversion
        if apply_offset:
            df["x"] = (df["x"] - bound.offset[0]) - bound.sim_offset[0]
            df["y"] = (df["y"] - bound.offset[1]) - bound.sim_offset[1]

        if epsg_code_base is not None:
            if apply_offset is False:
                raise ValueError(
                    "To apply any transformation the offset must be applied. "
                    "Change apply_offset or remove epsg codes from call"
                )
            df = self.apply_geo_position(df, epsg_code_base, epsg_code_to)
            cols = [*cols, "geometry"]

        return df.loc[:, cols]

    def get_sim_offset_and_bound(
        self,
    ):
        # get simulation bound offset
        offset = self.sca_data(
            module_name=f"{self.network}.coordConverter",
            scalar_name=self.OR(["simOffsetX:last", "simOffsetY:last"]),
        )["scalarValue"].to_numpy()

        # get simulation bound (width, height). Lower left point [0, 0] + offset
        bound = self.sca_data(
            module_name=f"{self.network}.coordConverter",
            scalar_name=self.OR(["simBoundX:last", "simBoundY:last"]),
        )["scalarValue"].to_numpy()
        return offset, bound

    def get_bound_polygon(self):
        bound = self.sca_data(
            module_name=f"{self.network}.coordConverter",
            scalar_name=self.OR(["simBoundX:last", "simBoundY:last"]),
        )["scalarValue"].to_numpy()
        return Polygon([(0, 0), (0, bound[1]), (bound[0], bound[1]), (bound[0], 0)])

    def apply_geo_position(
        self,
        df: pd.DataFrame,
        epsg_code_base: str | None = None,
        epsg_code_to: str | None = None,
    ) -> gpd.GeoDataFrame:
        geometry = [Point(x, y) for x, y in zip(df["x"], df["y"])]
        df = gpd.GeoDataFrame(df, crs=epsg_code_base, geometry=geometry)
        if epsg_code_to is not None:
            df = df.to_crs(epsg=epsg_code_to.replace("EPSG:", ""))
        return df

    def get_column_types(self, existing_columns, **kwargs):
        _cols = dict(self._dtypes)
        for k, v in kwargs.items():
            _cols[k] = v
        for k in set(_cols.keys()) - set(existing_columns):
            del _cols[k]
        return _cols

    @timing
    def vector_ids_to_host(
        self,
        module_name: SqlOp | str | None = None,
        vector_name: SqlOp | str | None = None,
        vector_ids: List[int] | None = None,
        run_id: int = 1,
        vec_info_columns: List[str] | None = ("vectorId",),
        name_columns: List[str] = ("host", "hostId", "vecIdx"),
        pull_data: bool = False,
        drop: List[str] | None = None,
        **pull_data_kw,
    ) -> pd.DataFrame:
        """Add human readable labels to vector data. The vector data can be provided either by
        module and vector name selectors or by a list of vector ids. If both are provided the
        vector_ids are ignored.

        Args:
            module_name (SqlOp, optional): Module selector. If None vector_ids must be set. Defaults to None.
            vector_name (SqlOp, optional): Vector selector. If None vector_ids must be set Defaults to None.
            vector_ids (List[int], optional): List of vectors to select. If set module_name and vector_name must be None. Defaults to None.
            run_id (int, optional): Defaults to 1.
            vec_info_columns (List[str], optional): List of columns to retrun. Defaults to ("vectorId").
                                                    Possible columns: [vectorId, runId, moduleName, vectorName,
                                                    vectorCount, vectorMin, vectorMax, vectorSum, vectorSumSqr,
                                                    startEventNum, endEventNum, startSimtimeRaw, endSimtimeRaw].
                                                    If None, return all columns. Defaults to None.
            name_columns (List[str], optional): Name columns for human readbale names. Defaults to ("host", "hostId", "vecIdx").
                                                Possible columns: [host, hostId, vecIdx]
            pull_data (bool, optional): In addition to any information columns query the data for the selected vectors. Defaults to False.

        Raises:
            ValueError: Either provoide module_name and vector_name or vector_ids

        Returns:
            pd.DataFrame: Structure depends on args
        """
        _cols = list(vec_info_columns)  # copy
        if "moduleName" not in vec_info_columns:
            # moduleName must be returned to create needed names. Will be
            # removed later if column is not selected by user.
            if any(isinstance(vec_info_columns, t) for t in [tuple, list]):
                vec_info_columns = ["moduleName", *vec_info_columns]
            elif isinstance(vec_info_columns, str):
                vec_info_columns = ["moduleName", vec_info_columns]
            else:
                raise ValueError(
                    f"Expected tuple, list or string but got {type(vec_info_columns)}"
                )

        _df = self.vec_info(
            module_name=module_name,
            vector_name=vector_name,
            vector_ids=vector_ids,
            run_id=run_id,
            cols=vec_info_columns,
        )

        if "host" in name_columns:
            _df["host"] = self.module_matcher.get_host(_df["moduleName"])
            _cols.append("host")
        if "hostId" in name_columns:
            _df["hostId"] = self.module_matcher.get_host_id(_df["moduleName"])
            _cols.append("hostId")
        if "vecIdx" in name_columns:
            _df["vecIdx"] = self.module_matcher.get_vector_index(_df["moduleName"])
            _cols.append("vecIdx")

        if pull_data:
            df = self.vec_data(ids=_df[_cols], **pull_data_kw)
        else:
            df = _df[_cols]
        if drop is not None:
            df = df.drop(columns=drop, errors="ignore")
        return df

    def create_event_number_join(self, records: dict) -> Tuple[str, str]:
        """Create join based on vectorData table and same eventNumber

        Performance: ensure that db has an index on vectorData.vectorId and
        vectorData.eventNumber. See self.check_if_index_exists() and
        self.append_index_if_missing() for checking and creating db index.

        Args:
            records (dict): record containing the hostId key with column name / vectorId mapping
                            as key value pairs.
        Raises:
            ValueError: If maximum number of joins is reached (20 for now)

        Returns:
            Tuple[str, str]: (hostId, sql statement)
        """
        statements = []
        for record in records:
            host_id = record["hostId"]
            joins = [(k, v) for k, v in record.items() if k != "hostId"]
            if len(joins) > 20:
                raise ValueError(f"To many joins max. 20 got {len(joins)}")

            selects = [
                f"v{idx}.value as '{col_name}'"
                for idx, (col_name, _) in enumerate(joins)
            ]
            selects_str = ",\n    ".join(selects)
            sql = f"select v0.simtimeRaw/1e12 as 'time', v0.eventNumber,\n    {selects_str}\n"
            sql = f"{sql}from vectorData as v0\n"

            def inner_join(first_id, second_id, join_idx):
                ret = f"    INNER JOIN vectorData as v{join_idx}\n"
                ret = f"{ret}        on v0.eventNumber == v{join_idx}.eventNumber\n"
                if join_idx == 1:
                    ret = f"{ret}        and v0.vectorID == {first_id}\n"
                ret = f"{ret}        and v{join_idx}.vectorID == {second_id}\n"
                return ret

            first_id = joins[0][1]  # first vectorId
            for idx, (_, second_id) in enumerate(joins[1:]):
                sql = f"{sql}{inner_join(first_id, second_id, idx+1)}"
            statements.append((host_id, sql))
        return statements

    @timing
    def vec_data_pivot(
        self,
        module_name: SqlOp | str,
        vector_name_map: dict,
        append_index: List[str] = (),
        index: List[str] | None = None,
    ) -> pd.DataFrame:
        """Query multiple vector values and transfrom data based on vector names and hostId/time key.

        e.g.
        Transfrom df
        hostId time value vector_name
        1      0.1  1       vec_1
        1      0.1  2       vec_2
        2      0.2  3       vec_1
        2      0.2  4       vec_2
        1      0.2  5       vec_1
        1      0.2  6       vec_2

        ... to
        hostId  time    vec_1   vec_2
        1       0.1     1       2
        1       0.2     5       6
        2       0.2     3       4


        Args:
            module_name (SqlOp): Module selector
            vector_name_map (dict): Dict of the form {<VectorName1>: {"name: _, "dtype": _ }, <VectorName1>: {...}, ...}
                                    <VectorNameX> must be the full vector name as listed in the *.vec database file. The
                                    vector will be renamed based on the "name" value. Use "dtype" to set the correct data type
                                    for each vector.
            append_index (List[str], optional): Append vector provided by the vector_name_map  to the default index ([hostId, time]) of the dataframe. Defaults to ().
            index (List[str], optional): Override default and 'append_index' by providing the complete index. User must ensure the columns exist. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe of the from [index](column):
                Variant1: [hostId, *append_index, time](<*vector_name>)   columns will be all vectors which are NOT added as additional indices.
                Variant2: [<index>](*)  based on index argument and vector_name_map
        """
        df = self.vector_ids_to_host(
            module_name,
            self.OR(list(vector_name_map.keys())),
            vec_info_columns=["vectorId", "vectorName"],
            name_columns=["hostId"],
        )
        if df.empty:
            raise SqlEmptyResult(
                f"No data for vector names: {list(vector_name_map.keys())} found."
            )
        timer = TimeIt()

        # get rows of vectorIds where columns are the vector names that should be joined.
        # performance: ensure that db has an index on vectorData.vectorId and vectorData.eventNumber.
        #              See self.check_if_index_exists() and self.append_index_if_missing()
        join_by_host_id = df.pivot(
            index="hostId", columns="vectorName", values="vectorId"
        )
        num_vecs = join_by_host_id.shape[1]
        with timer:
            join_it = self.create_event_number_join(
                join_by_host_id.reset_index().to_dict("records")
            )
            data = []
            for host_id, _sql in join_it:
                df = self.query_vec(sql_str=_sql)
                if df.shape[1] - 2 != num_vecs:
                    logger.warn(
                        f"Did not found all vectors for host {host_id} expected {num_vecs} vectors found {df.shape}: {df.columns}"
                    )
                df["hostId"] = host_id
                data.append(df)
        logger.info(
            f"Db inner join {join_by_host_id.shape[1]} vectors for {join_by_host_id.shape[0]} hostIds. Took: {timer.str()}"
        )

        vec_data = pd.concat(data, axis=0)
        vec_data = vec_data.rename(
            columns={k: v["name"] for k, v in vector_name_map.items()}
        )

        _dtypes = {v["name"]: v["dtype"] for _, v in vector_name_map.items()}
        col_dtypes = self.get_column_types(
            vec_data.columns.to_list(), time=float, **_dtypes
        )
        vec_data = vec_data.astype(col_dtypes)
        if index is None:
            _idx = ["hostId", *append_index, "eventNumber", "time"]
            # ensure no duplicates added and keep order. https://stackoverflow.com/a/480227
            seen = set()
            _idx = [x for x in _idx if not (x in seen or seen.add(x))]
        else:
            _idx = index

        if len(_idx) > 0:
            vec_data = vec_data.set_index(keys=_idx, verify_integrity=True)
            vec_data = vec_data.sort_index()
        return vec_data

    def get_app_selector(self):
        ret = {}
        cfg = self.get_app_config()
        for key, value in cfg.items():
            for app_num, app_type in value.items():
                if "map" in app_type.lower():
                    if app_num == 1:
                        ret["map"] = self.m_app1()
                    else:
                        ret["map"] = self.m_app0()
                elif "beacon" in app_type.lower():
                    if app_num == 1:
                        ret["beacon"] = self.m_app1()
                    else:
                        ret["beacon"] = self.m_app0()
        return ret

    def is_entropy_map(self):
        if self.dpmm_cfg is not None:
            return self.dpmm_cfg.is_entropy_map()
        else:
            map_type, _ = self.guess_map_type()
            return map_type == MapType.ENTROPY

    def is_count_map(self):
        if self.dpmm_cfg is not None:
            return self.dpmm_cfg.is_count_map()
        else:
            map_type, _ = self.guess_map_type()
            return map_type == MapType.DENSITY

    def guess_map_type(self) -> Tuple[MapType, str]:
        _paths = ["globalDensityMap", "gloablDensityMap", "globalMeasurementMap"]
        logger.warning(
            "Guessing map type is departed provide DpmmCfg object with concrete map type and where to find it."
        )
        out = []
        for _path in _paths:
            cfg = self.get_run_config(f"*.{_path}.typename", full_match=True)
            if "entropy" in cfg.lower():
                out.append(MapType.ENTROPY, _path)
            else:
                out.append(MapType.DENSITY, _path)

        if len(out) == 0:
            raise ValueError(
                f"Cannot guess global map type based on he following paths {_paths} (No key found).  Provide a concrete MapType via a DpmmCfg object. Guessing map type is deprecated."
            )
        elif len(out) > 1:
            raise ValueError(
                f"Cannot guess global map type because multiple values found {out}.  Provide a concrete MapType via a DpmmCfg object. Guessing map type is deprecated."
            )

        return out[0]

    # some default module selectors based on the vector database

    def m_channel(self, modules: List[str] | None = None) -> SqlOp:
        _m = modules if modules is not None else self.module_vectors
        return self.OR(
            [f"{self.network}.{i}[%].cellularNic.channelModel[0]" for i in _m]
        )

    def m_phy(self, modules: List[str] | None = None, idx: int | str = "%") -> SqlOp:
        _m = modules if modules is not None else self.module_vectors
        return self.OR([f"{self.network}.{i}[{idx}].cellularNic.phy" for i in _m])

    def m_app0(
        self, modules: List[str] | None = None, path="app", idx: int | str = "%"
    ) -> SqlOp:
        _m = modules if modules is not None else self.module_vectors
        path = path if path.startswith(".") else f".{path}"
        return self.OR([f"{self.network}.{i}[{idx}].app[0]{path}" for i in _m])

    def m_app1(
        self, modules: List[str] | None = None, path="app", idx: int | str = "%"
    ) -> SqlOp:
        _m = modules if modules is not None else self.module_vectors
        path = path if path.startswith(".") else f".{path}"
        return self.OR([f"{self.network}.{i}[{idx}].app[1]{path}" for i in _m])

    def m_beacon(
        self, modules: List[str] | None = None, path="app", idx: int | str = "%"
    ) -> SqlOp:
        """SqlOperation to select beacon application"""
        _m = modules if modules is not None else self.module_vectors
        if self.dpmm_cfg is None:
            _typename_0 = self.OR([f"{self.network}.{i}[{idx}].app[0].app" for i in _m])
            _t0 = self.parameter_data(_typename_0, "typename")
            if all(["Beacon" in i for i in _t0["paramValue"].to_list()]):
                return self.m_app0(_m, path=path, idx=idx)

            _typename_1 = self.OR([f"{self.network}.{i}[{idx}].app[1].app" for i in _m])
            _t1 = self.parameter_data(_typename_1, "typename")
            if all(["Beacon" in i for i in _t1["paramValue"].to_list()]):
                return self.m_app1(_m, path=path, idx=idx)

            raise ValueError(
                "Did not find beacon application at index app[0] or app[1]"
            )
        else:
            return self.dpmm_cfg.m_beacon(_m, node_index=idx, path=path)

    def m_map(
        self, modules: List[str] | None = None, path="app", idx: int | str = "%"
    ) -> SqlOp:
        _m = modules if modules is not None else self.module_vectors
        if self.dpmm_cfg is None:
            _typename_0 = self.OR([f"{self.network}.{i}[{idx}].app[0].app" for i in _m])
            _t0 = self.parameter_data(_typename_0, "typename")
            if all(["DensityMap" in i for i in _t0["paramValue"].to_list()]):
                return self.m_app0(_m, path=path, idx=idx)

            _typename_1 = self.OR(
                [f"{self.network}.{i}[{idx}].app[1].app.typename" for i in _m]
            )
            _t1 = self.parameter_data(_typename_1, "typename")
            if all(["DensityMap" in i for i in _t1["paramValue"].to_list()]):
                return self.m_app1(_m, path=path, idx=idx)

            raise ValueError("Did not find map application at index app[0] or app[1]")
        else:
            return self.dpmm_cfg.m_map(_m, node_index=idx, path=path)

    def m_enb(self, index: int = -1, module: str = "") -> SqlOp:
        if index < 0:
            return SqlOp(operator="", group=f"{self.network}.eNB[%]{module}")
        else:
            return SqlOp(operator="", group=f"{self.network}.eNB[{index}]{module}")

    def m_append_suffix(
        self, suffix: str, modules: str | List[str] | SqlOp | None = None
    ) -> SqlOp:
        if isinstance(modules, str):
            return f"{modules}{suffix}"
        elif isinstance(modules, SqlOp):
            return modules.append_suffix(suffix)
        else:
            _m = modules if modules is not None else self.module_vectors
            return self.OR([f"{self.network}.{i}[%]{suffix}" for i in _m])

    def m_table(self, modules: List[str] | None = None) -> SqlOp:
        _m = modules if modules is not None else self.module_vectors
        return self.OR([f"{self.network}.{i}[%].nTable" for i in _m])


class ScaveTool:
    """
    Python wrapper for OMNeT++ scavetool.

    Allows simple access to query and export functions defined in the scavetool.
    See #print_help, #print_export_help and #print_filter_help for scavetool usage.

    Use #create_or_get_csv_file to create (or use existing) csv files from one or
    many OMNeT++ result files. The method  accepts multiple glob patters which are
    search recursive (default) for files ending in *.vec and *.sca.
    If given a scave_filter is applied to reduce the amount of imported data. See #print_print_filter_help
    on usage.

    Use #load_csv to load an existing OMNeT++ csv file. The following columns are expected to exist.
      'run', 'type', 'module', 'name', 'attrname', 'attrvalue', 'value', 'count', 'sumweights',
      'mean', 'stddev', 'min', 'max', 'binedges', 'binvalues', 'vectime', 'vecvalue'.
    """

    _EXPORT = "x"
    _QUERY = "q"
    _INDEX = "i"
    _OUTPUT = "-o"
    _FILTER = "--filter"

    def __init__(self, config: ScaveConfig = None, timeout=360):
        if config is None:
            self._config = ScaveConfig()  # default
        else:
            self._config = config
        self._SCAVE_TOOL = self._config.scave_cmd(silent=True)
        self.timeout = timeout

    @classmethod
    def _is_valid(cls, file: str):
        if file.endswith(".sca") or file.endswith(".vec"):
            if os.path.exists(file):
                return True
        return False

    def filter_builder(self) -> ScaveFilter:
        return ScaveFilter(self._config)

    def load_csv(self, csv_file, converters=None) -> pd.DataFrame:
        """
        #load_csv to load an existing OMNeT++ csv file. The following columns are expected to exist.
          'run', 'type', 'module', 'name', 'attrname', 'attrvalue', 'value', 'count', 'sumweights',
          'mean', 'stddev', 'min', 'max', 'binedges', 'binvalues', 'vectime', 'vecvalue'.
        :param csv_file:    Path to csv file
        :return:            pd.DataFrame
        """
        if converters is None:
            converters = ScaveConverter()
        df = pd.read_csv(csv_file, converters=converters.get())
        return df

    def create_or_get_csv_file(
        self,
        csv_path,
        input_paths: List[str],
        override=False,
        scave_filter: Union[str, ScaveFilter] = None,
        recursive=True,
        print_selected_files=True,
    ):
        """
        #create_or_get_csv_file to create (or use existing) csv files from one or
        many OMNeT++ result files. The method  accepts multiple glob patters which are
         search recursive (default) for files ending in *.vec and *.sca.
        If given a scave_filter is applied to reduce the amount of imported data. See #print_print_filter_help
        on usage.
        :param csv_path:             path to existing csv file or path to new csv file (see :param override)
        :param input_paths:          List of glob patters search for *.vec and *.sca files
        :param override:             (default: False) override existing csv_path
        :param scave_filter:         (default: None) string based filter for scavetool see #print_filter_help for syntax
        :param recursive:            (default: True) use recursive glob patterns
        :param print_selected_files: print list of files selected by the given input_paths.
        :return:
        """
        if os.path.isfile(csv_path) and not override:
            return os.path.abspath(csv_path)

        cmd = self.export_cmd(
            input_paths=input_paths,
            output=os.path.abspath(csv_path),
            scave_filter=scave_filter,
            recursive=recursive,
            print_selected_files=print_selected_files,
        )
        self.exec(cmd)

        return os.path.abspath(csv_path)

    def load_df_from_scave(
        self,
        input_paths: Union[str, List[str]],
        scave_filter: Union[str, ScaveFilter] = None,
        recursive=True,
        converters=None,
    ) -> pd.DataFrame:
        """
         Directly load data into Dataframe from *.vec and *.sca files without creating a
         csv file first. Use stdout of scavetool to create Dataframe.

         Helpful variant for automated scripts to reduce memory footprint.

        :param input_paths:     List of glob patters search for *.vec and *.sca files
        :param scave_filter:    (default: None) string based filter for scavetool see #print_filter_help for syntax
        :param recursive:       (default: True) use recursive glob patterns
        :return:
        """
        if type(input_paths) == str:
            input_paths = [input_paths]

        cmd = self.export_cmd(
            input_paths=input_paths,
            output="-",  # read from stdout of scavetool
            scave_filter=scave_filter,
            recursive=recursive,
            options=["-F", "CSV-R"],
        )
        print(" ".join(cmd))
        stdout, stderr = self.read_stdout(cmd, encoding="")
        if stdout == b"":
            logger.error("error executing scavetool")
            print(str(stderr, encoding="utf8"))
            return pd.DataFrame()

        if converters is None:
            converters = ScaveConverter()
        # skip first row (container output)
        df = pd.read_csv(
            io.BytesIO(stdout),
            encoding="utf-8",
            converters=converters.get(),
        )
        return df

    def export_cmd(
        self,
        input_paths,
        output,
        scave_filter: Union[str, ScaveFilter] = None,
        recursive=True,
        options=None,
        print_selected_files=False,
    ):
        cmd = self._SCAVE_TOOL[:]
        cmd.append(self._EXPORT)
        cmd.append(self._OUTPUT)
        cmd.append(output)
        if scave_filter is not None:
            cmd.append(self._FILTER)
            if type(scave_filter) == str:
                cmd.append(self._config.escape(scave_filter))
            else:
                cmd.append(scave_filter.build(escape=True))

        if options is not None:
            cmd.extend(options)

        if len(input_paths) == 0:
            raise ValueError("no *.vec or *.sca files given.")

        # todo check if glob pattern exists first only then do this and the check
        opp_result_files = list()
        if any([_f for _f in input_paths if "*" in _f]):
            for file in input_paths:
                opp_result_files.extend(glob.glob(file, recursive=recursive))
        else:
            opp_result_files.extend(input_paths)

        opp_result_files = [
            f for f in opp_result_files if f.endswith(".vec") or f.endswith(".sca")
        ]
        if len(opp_result_files) == 0:
            raise ValueError("no opp input files selected.")

        log = "\n".join(opp_result_files)
        logger.info(f"found *.vec and *.sca:\n {log}")
        if print_selected_files:
            print("selected files:")
            for f in opp_result_files:
                print(f"\t{f}")

        cmd.extend(opp_result_files)
        return cmd

    def print_help(self):
        cmd = self._SCAVE_TOOL
        cmd.append("--help")
        self.exec(cmd)

    def print_export_help(self):
        cmd = self._SCAVE_TOOL
        cmd.append(self._EXPORT)
        cmd.append("--help")
        self.exec(cmd)

    def print_filter_help(self):
        cmd = self._SCAVE_TOOL
        cmd.append("help")
        cmd.append("filter")
        self.exec(cmd)

    def read_parameters(self, result_file, scave_filter=None):
        if scave_filter is None:
            scave_filter = self.filter_builder().t_parameter().build()
        cmd = self._config.scave_cmd(silent=True)
        cmd.extend(
            [
                "query",
                "--list-results",
                "--bare",
                "--grep-friendly",
                "--tabs",
                "--filter",
                scave_filter,
                result_file,
            ]
        )
        print(" ".join(cmd))
        out, err = self.read_stdout(cmd)
        if err != "":
            raise RuntimeError(f"container return error: \n{err}")

        out = [line.split("\t") for line in out.split("\n") if line != ""]
        return pd.DataFrame(out, columns=["run", "type", "module", "name", "value"])

    def read_stdout(self, cmd, encoding="utf-8"):
        scave_cmd = subprocess.Popen(
            cmd,
            cwd=os.path.curdir,
            stdin=None,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            out, err = scave_cmd.communicate(timeout=self.timeout)
            if encoding != "":
                return out.decode(encoding), err.decode(encoding)
            else:
                return out, err
        except subprocess.TimeoutExpired:
            logger.error("Timout reached")
            scave_cmd.kill()
            return b"", io.StringIO("timeout reached")

    def exec(self, cmd):
        scave_cmd = subprocess.Popen(
            cmd,
            cwd=os.path.curdir,
            shell=False,
            stdin=None,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            scave_cmd.wait()
            if scave_cmd.returncode != 0:
                logger.error(f"return code was {scave_cmd.returncode}")
                logger.error("command:")
                logger.error(f"{pp.pprint(cmd)}")
                print(scave_cmd.stdout.read().decode("utf-8"))
                print(scave_cmd.stderr.read().decode("utf-8"))

            else:
                logger.info(f"return code was {scave_cmd.returncode}")
                print(scave_cmd.stdout.read().decode("utf-8"))

        except subprocess.TimeoutExpired:
            logger.info(f"scavetool timeout reached. Kill it")
            os.kill(scave_cmd.pid, signal.SIGKILL)
            time.sleep(0.5)
            if scave_cmd.returncode is None:
                logger.error("scavetool still not dead after SIGKILL")
                raise

        logger.info(f"return code: {scave_cmd.returncode}")


class ScaveConverter:
    """
    pandas csv to DataFrame converter. Provides a dict of functions to use while
    reading csv file. The keys in the dict must match the column names.
    """

    def __init__(self):
        pass

    def parse_if_number(self, s):
        try:
            return float(s)
        except:
            return True if s == "true" else False if s == "false" else s if s else None

    def parse_ndarray(self, s):
        return np.fromstring(s, sep=" ", dtype=float) if s else None

    def parse_series(self, s):
        return pd.Series(np.fromstring(s, sep=" ", dtype=float)) if s else None

    def get_series_parser(self):
        return {
            "attrvalue": self.parse_if_number,
            "binedges": self.parse_series,  # histogram data
            "binvalues": self.parse_series,  # histogram data
            "vectime": self.parse_series,  # vector data
            "vecvalue": self.parse_series,
        }

    def get_array_parser(self):
        return {
            "attrvalue": self.parse_if_number,
            "binedges": self.parse_ndarray,  # histogram data
            "binvalues": self.parse_ndarray,  # histogram data
            "vectime": self.parse_ndarray,  # vector data
            "vecvalue": self.parse_ndarray,
        }

    def get(self):
        return self.get_array_parser()


class ScaveRunConverter(ScaveConverter):
    """
    pandas csv to DataFrame converter. Provides a dict of functions to use while
    reading csv file. The keys in the dict must match the column names.

    Simplify run name by providing a shorter name
    """

    def __init__(self, run_short_hand="r"):
        super().__init__()
        self._short_hand = run_short_hand
        self.run_map = {}
        self.network_map = {}

    def parse_run(self, s):
        if s in self.run_map:
            return self.run_map[s]
        else:
            ret = f"{self._short_hand}_{len(self.run_map)}"
            self.run_map.setdefault(s, ret)
            return ret

    def mapping_data_frame(self):
        d_a = [["run", k, v] for k, v in self.run_map.items()]
        return pd.DataFrame(d_a, columns=["level", "id", "mapping"])

    def get(self):
        return self.get_array_parser()

    def get_array_parser(self):
        return {
            "run": self.parse_run,
            "attrvalue": self.parse_if_number,
            "binedges": self.parse_ndarray,  # histogram data
            "binvalues": self.parse_ndarray,  # histogram data
            "vectime": self.parse_ndarray,  # vector data
            "vecvalue": self.parse_ndarray,
        }
