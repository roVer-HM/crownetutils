import abc
import contextlib
import os
from typing import List

import pandas as pd

from roveranalyzer.simulators.opp.provider.hdf.Operation import Operation


class IHdfProvider(metaclass=abc.ABCMeta):
    """
    Wrap access to a given HDF store (hdf_path) in a context manager. Wrapper is lazy and checks if store exist
    are *Not* done. Caller must ensure file exists
    """

    def __init__(self, hdf_path: str):
        self._hdf_path: str = hdf_path
        self._hdf_args: object = {"complevel": 9, "complib": "zlib"}
        self.group: str = self.group_key()
        self.idx_order: {} = self.index_order()

    # todo maybe getKeyList die von Kidnern implementiert wird damit man weiß welche reihenfolge die keys haben (für tuple wichtig)

    @contextlib.contextmanager  # to ensure store closes after access
    def ctx(self, mode="a", **kwargs) -> pd.HDFStore:
        _args = dict(self._hdf_args)
        _args.update(kwargs)
        store: pd.HDFStore = pd.HDFStore(self._hdf_path, mode=mode, **_args)
        try:
            yield store
        finally:
            store.close()

    def __set_args(self, append=False, **kwargs) -> None:
        if append:
            self._hdf_args.update(kwargs)
        else:
            self._hdf_args = kwargs

    @abc.abstractmethod
    def group_key(self) -> str:
        return "None"

    @abc.abstractmethod
    def index_order(self) -> {}:
        return {}

    @property
    def hdf_path(self):
        return self._hdf_path

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

    # todo Useless ?!?
    #   def check_index_order(self, item: tuple) -> bool:
    #       """
    #       Checks if the given tuple is in the correct order
    #       """
    #       if len(item) == 0:
    #           return False
    #       for i in range(len(item)):
    #           if item[i] != self.idx_order[i]:
    #               return False
    #       return True

    def _select_where(self, condition: List[str]) -> pd.DataFrame:
        with self.ctx(mode="r") as store:
            df = store.select(key=self.group, where=condition)
        return pd.DataFrame(df)

    def _build_range_condition(self, key: str, _min: str, _max: str) -> List[str]:
        return [f"{key}<={_max}", f"{key}>={_min}"]

    def _build_exact_condition(
        self, key: str, value: any, operation: str = Operation.EQ
    ) -> List[str]:
        if isinstance(value, List):
            return [f"ID in {value}"]
        else:
            return [f"{key}{operation}{str(value)}"]
