import glob
import io
import os
import pprint as pp
import re
import signal
import sqlite3 as sq
import subprocess
import time
from typing import List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from roveranalyzer.simulators.opp.accessor import Opp
from roveranalyzer.simulators.opp.configuration import Config
from roveranalyzer.utils import Timer, logger


class ScaveData:
    @classmethod
    def build_time_series(
        cls,
        opp_df,
        opp_vector_names,
        hdf_store=None,
        hdf_key=None,
        time_bin_size=0.0,
    ):
        """
        Build normalized data frames for OMNeT++ vectors. See Opp.normalize_vectors
        opp_df:         data frame from OMNeT++
        opp_vector_names:      same length as opp_vectors containing the column names for used for vecvalues
        hdf_store:      HDF store used to save generated data frame. Default=None (do not save to disk)
        hdf_key:        key to use for HDF storage.
        time_bin_size:       size of time bins used for time_step index. Default=0.0 (do not create a time_step index)
        """
        # "dcf.channelAccess.inProgressFrames.["queueingTime:vector", "queueingLength:vector"]"
        # "dcf.channelAccess.pendingQueue.["queueingTime:vector", "queueingLength:vector"]"

        # check input
        timer = Timer.create_and_start(
            "check input", label=cls.build_time_series.__name__
        )

        _df_ret = (
            opp_df.opp.filter()
            .vector()
            .name_in(opp_vector_names)
            .normalize_vectors(axis=0)
        )

        if time_bin_size > 0.0:
            timer.stop_start(f"add time_step index with bin size {time_bin_size}")
            time_idx = _df_ret["time"]
            # create bins based on given time_bin_size
            bins = np.arange(
                np.floor(time_idx.min()), np.ceil(time_idx.max()), step=time_bin_size
            )
            time_bin = pd.Series(
                np.digitize(time_idx, bins) * time_bin_size + time_idx.min(),
                name="time_step",
            )
            _df_ret["time_step"] = time_bin
            _df_ret = _df_ret.drop(columns=["time"])
            _df_ret = _df_ret.set_index("time_step", drop=True)

        timer.stop()
        return _df_ret

    @classmethod
    def stack_vectors(
        cls,
        df,
        index,
        columns=("vectime", "vecvalue"),
        col_data_name="data",
        drop=None,
        time_as_index=False,
    ):
        """
        data frame which only contains opp vector rows.
        """
        timer = Timer.create_and_start("set index", label=cls.stack_vectors.__name__)
        df = df.set_index(index)

        timer.stop_start("stack data")
        stacked = list()
        for col in columns:
            stacked.append(df[col].apply(pd.Series).stack())

        timer.stop_start("concatenate data")
        df = pd.concat(stacked, axis=1, keys=columns)

        timer.stop_start("cleanup index")  #
        if None in df.index.names:
            df.index = df.index.droplevel(level=None)

        timer.stop_start("drop columns or index level")
        for c in drop:
            if c in df.index.names:
                df.index = df.index.droplevel(level=c)
            elif c in df.columns:
                df = df.drop(c, axis=1)
            # else:
            #     print(f"waring: given name {c} cannot be droped. Does not exist.")

        timer.stop_start("rename vecvalue")
        df = df.rename({"vectime": "time", "vecvalue": col_data_name}, axis=1)

        if time_as_index:
            df = df.set_index("time", append=True)

        df = df.sort_index()
        timer.stop()
        return df


class ScaveFilter:
    """
    Build opp_scavetool filter using builder pattern.
    """

    @classmethod
    def create(cls):
        return cls()

    def __init__(self, config: Config = None):
        if config is None:
            self._config = Config()  # default
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


class SqlOp:
    """
    Helper class to build `WHERE` clause for matching a
    column against a single or multiple values which
    may contain placeholder strings `%`.

    Use classmethods `OR` or `AND` for the respective boolean operator
    needed.
    """

    def __init__(self, operator, group):
        self._operator = operator
        self._group = group if type(group) == list else [group]

    @classmethod
    def OR(cls, items):
        return cls("or", items)

    @classmethod
    def AND(cls, items):
        return cls("and", items)

    def get_names(self):
        return self._group

    def apply(self, table, column):
        ret = []
        for i in self._group:
            if "%" in i:
                ret.append(f"{table}.{column} like '{i}'")
            else:
                ret.append(f"{table}.{column} = '{i}'")
        if len(ret) == 1:
            return ret[0]
        else:
            ret = f" {self._operator} ".join(ret)
            return f"({ret})"


class OppSql:

    """
    Util class to access vec and sca database files.
    """

    OR = SqlOp.OR
    AND = SqlOp.AND

    def __init__(self, vec_path=None, sca_path=None):
        self._vec_path = vec_path
        self._vec_con = None
        self._sca_path = sca_path
        self._sca_con = None
        self._file = {
            "vec": lambda: self.vec_con,
            "sca": lambda: self.sca_con,
        }

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

    @property
    def vec_con(self):
        if self._vec_con is None:
            if not os.path.exists(self.vec_path):
                raise FileNotFoundError(self.vec_path)
            self._vec_con = sq.connect(self.vec_path)
        return self._vec_con

    @property
    def sca_con(self):
        if self._sca_con is None:
            if not os.path.exists(self.sca_path):
                raise FileNotFoundError(self.sca_path)
            self._sca_con = sq.connect(self.sca_path)
        return self._sca_con

    def _query(
        self, sql_str, file="vec", type="df", **kwargs
    ) -> Union[pd.DataFrame, sq.Cursor]:
        _con = self._file[file]()
        if type == "df":
            return pd.read_sql_query(sql_str, _con, **kwargs)
        elif type == "cursor":
            return _con.execute(sql_str, **kwargs)
        else:
            raise RuntimeError("Expected df or cursor as type")

    def query_vec(self, sql_str, type="df", **kwargs):
        return self._query(sql_str, file="vec", type=type, **kwargs)

    def query_sca(self, sql_str, type="df", **kwargs):
        return self._query(sql_str, file="sca", type=type, **kwargs)

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

    def vec_info(
        self, moduleName=None, vectorName=None, runId=1, cols=("vectorId",), **kwargs
    ):
        cols = ", ".join([f"v.{c}" for c in cols])
        _sql = f"select {cols} from vector v where v.runId = '{runId}' "
        _sql += self._to_sql(moduleName, "v", "moduleName", "and")
        _sql += self._to_sql(vectorName, "v", "vectorName", "and")

        # print(_sql)
        df = self.query_vec(_sql, type="df", **kwargs)
        return df

    def vec_ids(self, moduleName=None, vectorName=None, runId=1, **kwargs) -> List[int]:
        return self.vec_info(moduleName, vectorName, runId, **kwargs)[
            "vectorId"
        ].to_list()

    def vec_merge_on(
        self,
        moduleName: Union[None, str, SqlOp] = None,
        vectorName: Union[None, str, SqlOp] = None,
        runId=1,
        time_resolution=1e12,
        **kwargs,
    ):
        """
        Merge
        """

        _info = self.vec_info(
            moduleName, vectorName, runId, cols=("vectorId", "vectorName"), **kwargs
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

    def vec_data(
        self,
        module_name: Union[None, str, SqlOp] = None,
        vector_name: Union[None, str, SqlOp] = None,
        ids: Union[None, List[int], pd.DataFrame] = None,
        runId=1,
        cols: set = ("vectorId", "simtimeRaw", "value"),
        value_name: str = "value",
        time_slice: slice = slice(None),
        time_resolution=1e12,
        **kwargs,
    ):

        if module_name is not None and vector_name is not None:
            _ids = self.vec_ids(module_name, vector_name)
        elif type(ids) == pd.DataFrame:
            _ids = ids["vectorId"].unique()
            if "vectorId" not in cols:
                cols = [*cols, "vectorId"]
        else:
            _ids = ids

        _ids = ", ".join([str(i) for i in _ids])
        cols = ", ".join([f"v_data.{c}" for c in cols])
        if time_slice != slice(None):
            _time = f" and v_data.simTimeRaw >= {time_slice.start * time_resolution} and v_data.simTimeRaw <= {time_slice.stop * time_resolution}"
        else:
            _time = ""
        _sql = f"select {cols} from vectorData v_data where v_data.vectorId in ({_ids}) {_time}"
        df = self.query_vec(_sql, type="df", **kwargs)
        if time_resolution is not None and "simtimeRaw" in cols:
            df["simtimeRaw"] = df["simtimeRaw"] / time_resolution
            df = df.rename(columns={"simtimeRaw": "time"})

        if type(ids) == pd.DataFrame:
            df = pd.merge(df, ids, how="left", on=["vectorId"])

        df = df.rename(columns={"value": value_name})

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

    _pos_x = "posX:vector"
    _pos_y = "posy:vector"

    _module_vectors = ["misc", "pNode", "vNode"]

    _vehicle = "%s.misc[%d]"
    _vehicle_app = "%s.misc[%d].app[%d]"

    module_vectors = ["misc", "pNode", "vNode"]

    def __init__(self, vec_path=None, sca_path=None, network="World"):
        super().__init__(vec_path=vec_path, sca_path=sca_path)
        self.network = network
        self.module_names = self.OR(
            [f"{self.network}.{i}[%]" for i in self._module_vectors]
        )
        self._host_id_regex = re.compile(
            f"(?P<host>^{self.network}\.(?P<type>{'|'.join(self._module_vectors)})\[\d+\]).*"
        )
        self._host_index_regex = re.compile(
            f"^{self.network}\.(?P<type>{'|'.join(self._module_vectors)})\[(?P<hostIdx>\d+)\].*"
        )

    def host_ids(self, module_name: Union[None, str, SqlOp] = None):
        """
        Return hostIds of all vector nodes present in simulation (misc, pNode, vNode)
        """
        module_name = self.module_names if module_name is None else module_name

        _sql: pd.DataFrame = f"select s.moduleName, s.scalarValue from scalar as s where \n  {self._to_sql(module_name, 's', 'moduleName')} \n  AND s.scalarName = 'hostId:last'"
        _df = self.query_sca(sql_str=_sql)
        _df["scalarValue"] = pd.to_numeric(_df["scalarValue"], downcast="integer")
        _df = _df.reset_index(drop=True).set_index(keys="scalarValue")
        return _df.to_dict()["moduleName"]

    def module_to_host_ids(self):
        return {v: k for k, v in self.host_ids().items()}

    def host_position(
        self,
        module_name: Union[SqlOp, str, None] = None,
        ids: Union[pd.DataFrame, None] = None,
        time_slice: slice = slice(None),
        epsg_code_base: Union[str, None] = None,
        epsg_code_to: Union[str, None] = None,
        cols: tuple = ("time", "hostId", "host", "vecIdx", "x", "y"),
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Get host position data in cartesian coordinates.

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

        Examples:
        1) all host from all vectors (misc[*], pNode[*], vNode[*]) between 12.5s and 90.0s

        df = host_pos(time_slice=slice(12.5, 90.0))      # defaults to all

        2) Only misc and pNodes with UTM ZONE 32N (default for sumo based simulations)
           Use the 'or' operator to select both misc and pNode vectors. Ensure that the
           full path is correct.

        gdf = host_pos(
            module_name=SqlOp.OR(["World.misc[%]", World.pNode[%]])
            epsg_code=32632
        )

        3) Only select positions for selected host (may come from different vectors)
           In this example 3 host are selected where the vectorIds=[5, 12, 14]  are
           x postion vectors and [9, 44, 18] are y position vectors for 3 host.

        ids = pd.DataFrame([[5,9],[12,44],[14, 18]], columns=["x", "y"])
        df = host_pos(ids=ids)

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
            _x = self.vector_ids_to_host(vec_ids=ids["x"].unique())
            _y = self.vector_ids_to_host(vec_ids=ids["y"].unique())
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

        # convert to bottom-left origin and remove offset used during the simulation
        df["y"] = bound[1] - df["y"]  # move from top-left-orig to bottom-left-orig
        # nothing to do for x, only y-axis needs conversion
        df["x"] = df["x"] - offset[0]
        df["y"] = df["y"] - offset[1]

        if epsg_code_base is not None:
            geometry = [Point(x, y) for x, y in zip(df["x"], df["y"])]
            df = gpd.GeoDataFrame(df, crs=epsg_code_base, geometry=geometry)
            cols = [*cols, "geometry"]
            if epsg_code_to is not None:
                df = df.to_crs(epsg=epsg_code_to.replace("EPSG:", ""))

        return df.loc[:, cols]

    def vector_ids_to_host(
        self,
        module_name: Union[None, str, SqlOp] = None,
        vector_name: Union[None, str, SqlOp] = None,
        vec_ids: Union[None, List[int]] = None,
        columns=("vectorId", "host", "hostId", "vecIdx"),
    ) -> pd.DataFrame:
        module_map = self.module_to_host_ids()

        if module_name is not None and vector_name is not None:
            vec_ids = self.vec_ids(module_name, vector_name)

        if vec_ids is None:
            raise ValueError("set either module and vector name or provide vector ids")

        _vec_id_str = ",".join([str(i) for i in vec_ids])

        def _match_host(x):
            if _m := self._host_index_regex.match(x):
                return f'{_m.groupdict()["type"]}[{_m.groupdict()["hostIdx"]}]'
            raise ValueError(
                f"given moduelName '{x}' does not match vector index regex {self._host_index_regex}"
            )

        def _match_host_id(x):
            if _m := self._host_id_regex.match(x):
                if _m.groupdict()["host"] in module_map:
                    return module_map[_m.groupdict()["host"]]
                else:
                    ValueError(f"Module {_m} not found in module map")
            raise ValueError(
                f"given moduleName '{x}' does match module regex {self._host_id_regex}"
            )

        def _match_vector_idx(x):
            if _m := self._host_index_regex.match(x):
                return int(_m.groupdict()["hostIdx"])
            raise ValueError(
                f"given moduelName '{x}' does not match vector index regex {self._host_index_regex}"
            )

        _sql = f"select v.vectorId, v.moduleName from vector as v where v.vectorId in ({_vec_id_str})"
        _df = self.query_vec(_sql)
        if "host" in columns:
            _df["host"] = _df["moduleName"].apply(lambda x: _match_host(x))
        if "hostId" in columns:
            _df["hostId"] = _df["moduleName"].apply(lambda x: _match_host_id(x))
        if "vecIdx" in columns:
            _df["vecIdx"] = _df["moduleName"].apply(lambda x: _match_vector_idx(x))

        return _df[list(columns)]


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

    def __init__(self, config: Config = None, timeout=360):
        if config is None:
            self._config = Config()  # default
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
        :return:            pd.DataFrame with extra namespace 'opp' (an OppAccessor object with helpers)
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
