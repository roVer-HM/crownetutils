from __future__ import annotations

import abc
import contextlib
import os
import subprocess
import threading
import warnings
from enum import Enum
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import pandas as pd
import tables
from geopandas.geodataframe import GeoDataFrame

from roveranalyzer.simulators.opp.provider.hdf.IHdfGeoProvider import GeoProvider
from roveranalyzer.simulators.opp.provider.hdf.Operation import Operation
from roveranalyzer.utils import logger


class UnsupportedOperation(RuntimeError):
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)


class BaseHdfProvider:
    def __init__(self, hdf_path: str, group: str = "root"):
        self._lock = threading.Lock()
        self.group: str = group
        self._hdf_path: str = hdf_path
        self._hdf_args: Dict[str, Any] = {"complevel": 9, "complib": "zlib"}

    # allow pickling of hdf providers
    def __getstate__(self):
        _state = self.__dict__.copy()
        del _state["_lock"]  # remove unpicklable entry
        return _state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def get_dataframe(self, group=None) -> pd.DataFrame:
        """
        Select the entire dataframe behind given key or the default location.
        Warning: this may take some time and may cause memory problems
        """
        _key = self.group if group is None else group
        with self.ctx(mode="r") as store:
            df = store.get(key=_key)
        return pd.DataFrame(df)

    def write_frame(self, group, frame, index=True, index_data_columns=True):
        with self.ctx() as store:
            store.append(
                key=group,
                value=frame,
                index=index,
                format="table",
                data_columns=index_data_columns,
            )

    def put_frame_fixed(self, group, frame, index_data_columns=True):
        with self.ctx() as store:
            store.put(
                key=group,
                value=frame,
                index=True,
                format="fixed",
                data_columns=index_data_columns,
            )

    def override_frame(
        self, group: str, frame: pd.DataFrame, index=True, index_data_columns=True
    ):
        if self.contains_group(group):
            with self.ctx() as store:
                store.remove(group)
        self.write_frame(group, frame, index, index_data_columns)

    def repack_hdf(self, keep_old_file: bool = True):
        new_path = f"{self._hdf_path[0:-3]}_new.h5"
        old_path = f"{self._hdf_path[0:-3]}_old.h5"
        if os.path.exists(new_path) or os.path.exists(old_path):
            raise ValueError(
                "Cannot repack file. Temp files *_old.h5 or *_new.h5 already exist. Delete/Move files and retry"
            )
        args = [
            "ptrepack",
            "--chunkshape=auto",
            "--propindexes",
            "--complib",
            self._hdf_args.get("complib", "zlib"),
            "--complevel",
            str(self._hdf_args.get("complevel", 9)),
            self._hdf_path,
            new_path,
        ]

        try:
            fd = NamedTemporaryFile()
            print(f"repack {self._hdf_path}. This might take some time...")
            ret = subprocess.check_call(
                args,
                stdout=fd,
                stderr=fd,
                env=os.environ,
                cwd=os.path.dirname(self._hdf_path),
            )
        except subprocess.CalledProcessError:
            print(f"Error while repacking {self._hdf_path}\nargs:{args}")
            with open(fd.name, "r") as f:
                print("\n".join(f.readlines()))
            return

        if ret != 0:
            print(f"non zero return code from ptrepack: {ret}\nargs:{args}")
            with open(fd.name, "r") as f:
                print("\n".join(f.readlines()))
        else:
            os.rename(self._hdf_path, old_path)
            os.rename(new_path, self._hdf_path)
            if not keep_old_file:
                os.remove(old_path)

    def set_attribute(self, attr_key: str, value: Any, group=None):

        _key = self.group if group is None else group
        with self.tables_file(self._hdf_path, "a") as hdf_file:
            # with tables.open_file(self._hdf_path, "a") as hdf_file:
            if _key not in hdf_file.root:
                hdf_file.create_group("/", _key, "")
            hdf_file.root[_key].table.attrs[attr_key] = value

    def get_attribute(self, attr_key: str, group=None, default: Any = None):

        if not self.hdf_file_exists:
            return default

        _key = self.group if group is None else group
        with self.tables_file(self._hdf_path, "r") as hdf_file:
            # with tables.open_file(self._hdf_path, "r") as hdf_file:
            if attr_key in hdf_file.root[_key].table.attrs:
                return hdf_file.root[_key].table.attrs[attr_key]
            else:
                return default

    @property
    def hdf_file_exists(self):
        return os.path.exists(self._hdf_path)

    def get_time_interval(self):
        return self.get_attribute("time_interval")

    def contains_group(self, group):
        with self.ctx() as ctx:
            return group in [g._v_name for g in ctx.groups()]

    @contextlib.contextmanager  # to ensure store closes after access
    def ctx(self, mode="a", **kwargs) -> Iterator[pd.HDFStore]:
        try:
            self._lock.acquire()
            _args = dict(self._hdf_args)
            _args.update(kwargs)
            store: pd.HDFStore = pd.HDFStore(self._hdf_path, mode=mode, **_args)
            try:
                yield store
            finally:
                store.close()
        finally:
            self._lock.release()

    @contextlib.contextmanager
    def tables_file(self, path, mode="r", **kwargs) -> Iterator[tables.File]:
        try:
            self._lock.acquire()
            file = tables.open_file(path, mode)
            try:
                yield file
            finally:
                file.close()
        finally:
            self._lock.release()

    @property
    def query(self) -> Iterator[pd.HDFStore]:
        return self.ctx(mode="r")


class ProviderVersion(Enum):

    V0_1 = "0.1"
    V0_2 = "0.2"
    V0_3 = "0.3"

    @classmethod
    def current(cls):
        return list(cls.__members__.values())[-1]

    @classmethod
    def to_list(cls, ascending: bool = True):
        if ascending:
            return list(cls.__members__.values())
        else:
            return list(cls.__members__.values())[::-1]


class VersionDict:
    """Provide a simple versioned dictionary with a fallback to 'highest'
    previous version when the currently select version is not part of the
    dictionary. Reasoning. If other parts of the HDF module changes which
    needs a new version (i.e. new columns) other parts do not need to be
    updated.
    """

    def __init__(self, data, **kwds) -> None:
        self.data = data
        for k, v in kwds.items():
            self.data[self.parse_key(k)] = v

    def current(self):
        return self.data[ProviderVersion.current]

    def parse_key(self, key):
        key = ProviderVersion.current if key is None else key
        if isinstance(key, ProviderVersion):
            if key not in self.data:
                for _biggest_key in ProviderVersion.to_list(ascending=False):
                    if _biggest_key in self.data:
                        return _biggest_key
                raise KeyError(f"did not found matching key data for key {key}")
            else:
                return key
        print("foo")

    def __getitem__(self, key):
        return self.data[self.parse_key(key)]

    def __setitem__(self, key, value):
        raise RuntimeError("ProviderBaseDict is read only")


class IHdfProvider(BaseHdfProvider, metaclass=abc.ABCMeta):
    """
    Wrap access to a given HDF store (hdf_path) in a context manager. Wrapper is lazy and checks if store exist
    are *Not* done. Caller must ensure file exists
    """

    def __init__(self, hdf_path: str, version: str | None = None):
        super().__init__(hdf_path, group=self.group_key())
        # self._hdf_path: str = hdf_path
        # self._hdf_args: Dict[str, Any] = {"complevel": 9, "complib": "zlib"}
        # self.group: str = self.group_key()
        if version is None:
            # use version of provided hdf-file or default to current  version.
            self._version = self.get_attribute(
                "version", default=ProviderVersion.current()
            )
            if not isinstance(self._version, ProviderVersion):
                self._version = ProviderVersion(self._version)
        else:
            if self.hdf_file_exists and version != self.get_attribute("version"):
                raise ValueError(
                    f"Version missmatch. hdf file reports version {self.get_attribute('version')} but object expected {version}"
                )
            self._version = version
        logger.debug(f"HDF version: {self.version}")
        self.idx_order: Dict = self.index_order()
        self._dispatcher = {
            int: self._handle_primitive,
            float: self._handle_primitive,
            str: self._handle_primitive,
            list: self._handle_list,
            slice: self._handle_slice,
            tuple: self._handle_tuple,
            Operation: self._handle_operation,
        }
        self._filters = set()
        self.operators = Operation

    @property
    def version(self):
        return self._version

    @abc.abstractmethod
    def group_key(self) -> str:
        return "None"

    @abc.abstractmethod
    def index_order(self) -> Dict:
        return {}

    @abc.abstractmethod
    def columns(self) -> List[str]:
        return []

    @abc.abstractmethod
    def default_index_key(self) -> str:
        return "None"

    def _to_geo(
        self, df: pd.DataFrame, to_crs: Union[str, None] = None
    ) -> GeoDataFrame:
        raise NotImplementedError("not supported operation")

    @property
    def hdf_path(self):
        return self._hdf_path

    @property
    def dispatcher(self):
        return self._dispatcher

    def geo(self, to_crs=None) -> GeoProvider:
        return GeoProvider(self, to_crs)

    def add_filter(self, **kwargs):
        keys = [*self.columns(), *(self.index_order().values())]
        for key, value in kwargs.items():
            if key not in keys:
                raise ValueError("Filter key not in index or columns")
            con, _ = self.dispatcher[type(value)](key, value)
            for c in con:
                self._filters.add(c)
        return self

    def clear_filter(self):
        self._filters.clear()

    @staticmethod
    def cast_to_set(value: Any):
        t = type(value)
        if t in [str, int, float]:
            return {value}
        elif t in [list, tuple]:
            return set(value)
        else:
            return value

    def _handle_operation(
        self, key: str, value: Operation
    ) -> Tuple[List[str], Optional[List[str]]]:
        condition = self._build_exact_condition(
            key=key, value=value.value, operation=value.operation
        )
        return condition, None

    def _handle_primitive(
        self, key: str, value: any
    ) -> Tuple[List[str], Optional[List[str]]]:
        condition = self._build_exact_condition(key=key, value=value)
        return condition, None

    def _handle_list(
        self, key: str, values: List
    ) -> Tuple[List[str], Optional[List[str]]]:
        list_without_none = [v for v in values if v is not None]
        condition = self._build_exact_condition(key, list_without_none)
        return condition, None

    def _handle_slice(
        self, key: str, value: slice
    ) -> Tuple[List[str], Optional[List[str]]]:
        if value.step and value.step > 1:
            warnings.warn(
                message=f"Step size of '{value.step}' is not supported. Step size must be '1'.",
                category=UserWarning,
            )
        condition = self._build_range_condition(
            key=key, _min=value.start, _max=value.stop
        )
        return condition, None

    def _handle_index_tuple(self, value: tuple) -> List[str]:
        if len(value) > len(self.idx_order):
            raise ValueError(
                f"To many values in tuple. Got: {len(value)} expected: <={len(self.idx_order)}"
            )

        condition: List[str] = []
        for idx in range(len(self.idx_order)):
            tuple_item = value[idx] if len(value) > idx else None
            if tuple_item is not None:
                condition += self.dispatch(self.idx_order[idx], tuple_item)[
                    0
                ]  # ignore columns

        return condition

    # key is required because the dispatcher handles every type the same way
    def _handle_tuple(
        self, key: str, value: tuple
    ) -> Tuple[List[str], Optional[List[str]]]:
        column_check_set: Set = self.cast_to_set(value[1])
        if type(column_check_set) == set:
            if column_check_set.issubset(self.columns()):
                # tuple of tuple with columns
                condition = self.dispatch(key, value[0])[0]
                columns = list(column_check_set)
                return condition, columns
            elif all(isinstance(x, str) for x in column_check_set):
                raise ValueError(f"Unknown column index in {column_check_set}")
        condition = self._handle_index_tuple(value)
        return condition, None

    def dispatch(self, key, item) -> Tuple[List[str], Optional[List[str]]]:
        item_type = type(item)
        f = self.dispatcher[item_type]
        condition, columns = f(key, item)
        return condition, columns

    def __getitem__(self, item: any) -> pd.DataFrame:
        condition, columns = self.dispatch(self.default_index_key(), item)
        condition.extend(list(self._filters))
        # remove conditions containing 'None' values
        condition = [i for i in condition if not "None" in i]
        if len(condition) == 0 and columns is None:
            # empty condition -> return full frame
            dataframe = self.get_dataframe()
        else:
            dataframe = self._select_where(condition, columns)
        # if (
        #     dataframe.empty and columns is None
        # ):  # if len(column) == 0 user only wants index!
        #     raise ValueError(
        #         f"Returned dataframe was empty. Please check your index names.{condition=}"
        #     )
        # allow empty frames
        return dataframe

    def __setitem__(self, key, value):
        raise UnsupportedOperation("Not supported!")

    def write_dataframe(self, data: pd.DataFrame) -> None:
        with self.ctx(mode="a") as store:
            store.put(key=self.group, value=data, format="table", data_columns=True)

    def exists(self) -> bool:
        """check for HDF store"""
        return os.path.exists(self._hdf_path)

    def _select_where(
        self, condition: List[str], columns: List[str] = None
    ) -> pd.DataFrame:
        with self.ctx(mode="r") as store:
            con = None if len(condition) == 0 else condition
            df = store.select(key=self.group, where=con, columns=columns)
        return pd.DataFrame(df)

    @staticmethod
    def _build_range_condition(key: str, _min: float, _max: float) -> List[str]:
        return [f"{key}<={str(_max)}", f"{key}>={str(_min)}"]

    @staticmethod
    def _build_exact_condition(
        key: str, value: any, operation: str = Operation.EQ
    ) -> List[str]:
        if isinstance(value, List):
            return [f"{key} in {value}"]
        else:
            return [f"{key}{operation}{str(value)}"]
