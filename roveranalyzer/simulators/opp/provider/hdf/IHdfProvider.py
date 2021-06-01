import abc
import contextlib
import os
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from roveranalyzer.simulators.opp.provider.hdf.Operation import Operation


class IHdfProvider(metaclass=abc.ABCMeta):
    """
    Wrap access to a given HDF store (hdf_path) in a context manager. Wrapper is lazy and checks if store exist
    are *Not* done. Caller must ensure file exists
    """

    def __init__(self, hdf_path: str):
        self._hdf_path: str = hdf_path
        self._hdf_args: Dict[str, Any] = {"complevel": 9, "complib": "zlib"}
        self.group: str = self.group_key()
        self.idx_order: {} = self.index_order()
        self._dispatcher = {
            int: self._handle_primitive,
            float: self._handle_primitive,
            str: self._handle_primitive,
            list: self._handle_list,
            slice: self._handle_slice,
            tuple: self._handle_tuple,
        }

    @abc.abstractmethod
    def group_key(self) -> str:
        return "None"

    @abc.abstractmethod
    def index_order(self) -> {}:
        return {}

    @abc.abstractmethod
    def columns(self) -> List[str]:
        return []

    @abc.abstractmethod
    def default_index_key(self) -> str:
        return "None"

    @property
    def hdf_path(self):
        return self._hdf_path

    @property
    def dispatcher(self):
        return self._dispatcher

    @contextlib.contextmanager  # to ensure store closes after access
    def ctx(self, mode="a", **kwargs) -> pd.HDFStore:
        _args = dict(self._hdf_args)
        _args.update(kwargs)
        store: pd.HDFStore = pd.HDFStore(self._hdf_path, mode=mode, **_args)
        try:
            yield store
        finally:
            store.close()

    @staticmethod
    def cast_to_set(value: Any):
        t = type(value)
        if t in [str, int, float]:
            return {value}
        elif t in [list, tuple]:
            return set(value)
        else:
            return value

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
        condition = self._handle_index_tuple(value)
        return condition, None

    def dispatch(self, key, item) -> Tuple[List[str], Optional[List[str]]]:
        item_type = type(item)
        f = self.dispatcher[item_type]
        condition, columns = f(key, item)
        return condition, columns

    def __getitem__(self, item: any):
        condition, columns = self.dispatch(self.default_index_key(), item)
        print(
            f"hdf select condition: {condition}, "
            f"columns: {'[]' if columns is None else columns}"
        )
        dataframe = self._select_where(condition, columns)
        if dataframe.empty:
            raise ValueError(
                "Returned dataframe is empty. Please check your index names."
            )
        return dataframe

    def __setitem__(self, key, value):
        raise NotImplementedError("Not supported!")

    def get_dataframe(self) -> pd.DataFrame:
        with self.ctx(mode="r") as store:
            df = store.get(key=self.group)
        return pd.DataFrame(df)

    def write_dataframe(self, data: pd.DataFrame) -> None:
        with self.ctx(mode="w") as store:
            store.put(key=self.group, value=data, format="table", data_columns=True)

    def exists(self) -> bool:
        """ check for HDF store """
        return os.path.exists(self._hdf_path)

    def _select_where(
        self, condition: List[str], columns: List[str] = None
    ) -> pd.DataFrame:
        with self.ctx(mode="r") as store:
            df = store.select(key=self.group, where=condition, columns=columns)
        return pd.DataFrame(df)

    def _build_range_condition(self, key: str, _min: float, _max: float) -> List[str]:
        return [f"{key}<={str(_max)}", f"{key}>={str(_min)}"]

    def _build_exact_condition(
        self, key: str, value: any, operation: str = Operation.EQ
    ) -> List[str]:
        if isinstance(value, List):
            return [f"{key} in {value}"]
        else:
            return [f"{key}{operation}{str(value)}"]
