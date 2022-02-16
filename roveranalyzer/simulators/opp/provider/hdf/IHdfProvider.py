import abc
import contextlib
import os
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from geopandas.geodataframe import GeoDataFrame

from roveranalyzer.simulators.opp.provider.hdf.IHdfGeoProvider import GeoProvider
from roveranalyzer.simulators.opp.provider.hdf.Operation import Operation


class UnsupportedOperation(RuntimeError):
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)


class FrameConsumer(metaclass=abc.ABCMeta):
    """
    Consume dataframe and use it for some actions
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def consume(self, df: pd.DataFrame):
        pass


class BaseHdfProvider:
    def __init__(self, hdf_path: str, group: str = "root"):
        self.group: str = group
        self._hdf_path: str = hdf_path
        self._hdf_args: Dict[str, Any] = {"complevel": 9, "complib": "zlib"}

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
                data_columns=index_data_columns
            )

    def set_attribute(self, attr_key: str, value: Any, group=None):
        import tables

        _key = self.group if group is None else group
        with tables.open_file(self._hdf_path, "a") as hdf_file:
            if _key not in hdf_file.root:
                hdf_file.create_group("/", _key, "")
            hdf_file.root[_key].table.attrs[attr_key] = value

    def get_attribute(self, attr_key: str, group=None):
        import tables

        _key = self.group if group is None else group
        with tables.open_file(self._hdf_path, "r") as hdf_file:
            return hdf_file.root[_key].table.attrs[attr_key]

    def get_time_interval(self):
        return self.get_attribute("time_interval")

    def contains_group(self, group):
        with self.ctx() as ctx:
            return group in [g._v_name for g in ctx.groups()]

    @contextlib.contextmanager  # to ensure store closes after access
    def ctx(self, mode="a", **kwargs) -> pd.HDFStore:
        _args = dict(self._hdf_args)
        _args.update(kwargs)
        store: pd.HDFStore = pd.HDFStore(self._hdf_path, mode=mode, **_args)
        try:
            yield store
        finally:
            store.close()
    
    @property
    def read(self) -> pd.HDFStore:
        return self.ctx(mode="r")


class IHdfProvider(BaseHdfProvider, metaclass=abc.ABCMeta):
    """
    Wrap access to a given HDF store (hdf_path) in a context manager. Wrapper is lazy and checks if store exist
    are *Not* done. Caller must ensure file exists
    """

    def __init__(self, hdf_path: str):
        super().__init__(hdf_path, group=self.group_key())
        # self._hdf_path: str = hdf_path
        # self._hdf_args: Dict[str, Any] = {"complevel": 9, "complib": "zlib"}
        # self.group: str = self.group_key()
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
            if tuple_item:
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
        if dataframe.empty:
            raise ValueError(
                f"Returned dataframe was empty. Please check your index names.{condition=}"
            )
        return dataframe

    def __setitem__(self, key, value):
        raise UnsupportedOperation("Not supported!")

    def write_dataframe(self, data: pd.DataFrame) -> None:
        with self.ctx(mode="a") as store:
            store.put(key=self.group, value=data, format="table", data_columns=True)

    def exists(self) -> bool:
        """ check for HDF store """
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
