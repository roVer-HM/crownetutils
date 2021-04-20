from typing import List

from roveranalyzer.simulators.opp.provider.hdf.IHdfProvider import IHdfProvider
from roveranalyzer.simulators.opp.provider.hdf.HdfGroups import HdfGroups
from roveranalyzer.simulators.opp.provider.hdf.Operation import Operation

import pandas as pd


class CountMapKey:
    ID = 'ID'
    SIMTIME = 'simtime'
    X = 'x'
    Y = 'y'
    COUNT = 'count'
    ERR = 'err'
    OWNER_DIST = 'owner_dist'
    SQERR = 'sqerr'


class CountMapHdfProvider(IHdfProvider):

    def __init__(self, hdf_path):
        super().__init__(hdf_path)

    def group_key(self) -> str:
        return HdfGroups.COUNT_MAP

    ########################
    # Exact Functions      #
    # single value or list #
    ########################
    def select_id_exact(self, value: any, operation: str = Operation.EQ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(key=CountMapKey.ID, value=value, operation=operation)
        return self._select_where(condition=condition)

    def select_simtime_exact(self, value: any, operation: str = Operation.EQ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(key=CountMapKey.SIMTIME, value=value, operation=operation)
        return self._select_where(condition=condition)

    def select_x_exact(self, value: any, operation: str = Operation.EQ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(key=CountMapKey.X, value=value, operation=operation)
        return self._select_where(condition=condition)

    def select_y_exact(self, value: any, operation: str = Operation.EQ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(key=CountMapKey.Y, value=value, operation=operation)
        return self._select_where(condition=condition)

    def select_count_exact(self, value: any, operation: str = Operation.EQ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(key=CountMapKey.COUNT, value=value, operation=operation)
        return self._select_where(condition=condition)

    def select_err_exact(self, value: any, operation: str = Operation.EQ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(key=CountMapKey.ERR, value=value, operation=operation)
        return self._select_where(condition=condition)

    def select_owner_dist_exact(self, value: any, operation: str = Operation.EQ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(key=CountMapKey.OWNER_DIST, value=value, operation=operation)
        return self._select_where(condition=condition)

    def select_sqerr_exact(self, value: any, operation: str = Operation.EQ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(key=CountMapKey.SQERR, value=value, operation=operation)
        return self._select_where(condition=condition)

    ###################
    # Range Functions #
    ###################
    def select_id_range(self, _min: int, _max: int) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(key=CountMapKey.ID, _min=str(_min), _max=str(_max))
        return self._select_where(condition=condition)

    def select_simtime_range(self, _min: int, _max: int) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(key=CountMapKey.SIMTIME, _min=str(_min), _max=str(_max))
        return self._select_where(condition=condition)

    def select_x_range(self, _min: int, _max: int) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(key=CountMapKey.X, _min=str(_min), _max=str(_max))
        return self._select_where(condition=condition)

    def select_y_range(self, _min: int, _max: int) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(key=CountMapKey.Y, _min=str(_min), _max=str(_max))
        return self._select_where(condition=condition)

    def select_count_range(self, _min: float, _max: float) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(key=CountMapKey.COUNT, _min=str(_min), _max=str(_max))
        return self._select_where(condition=condition)

    def select_err_range(self, _min: float, _max: float) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(key=CountMapKey.ERR, _min=str(_min), _max=str(_max))
        return self._select_where(condition=condition)

    def select_owner_dist_range(self, _min: float, _max: float) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(key=CountMapKey.OWNER_DIST, _min=str(_min), _max=str(_max))
        return self._select_where(condition=condition)

    def select_sqerr_range(self, _min: float, _max: float) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(key=CountMapKey.SQERR, _min=str(_min), _max=str(_max))
        return self._select_where(condition=condition)

    def addColumn(self):
        # TRY 1 (not working)
        indx = [CountMapKey.ID,
                CountMapKey.SIMTIME,
                CountMapKey.X,
                CountMapKey.Y]
        cols = [CountMapKey.COUNT,
                CountMapKey.ERR,
                CountMapKey.OWNER_DIST,
                CountMapKey.SQERR]
        # chunk_size = 20
        # with self.ctx(mode='r') as store:
        #     for chunk in pd.read_table(self._hdf_path, names=cols,
        #                                chunksize=chunk_size):
        #         store.append(self.group, chunk, data_columns=cols + ["new_column"], index=False)
        #     store.create_table_index(self.group, columns=cols + ["new_column"], optlevel=9, kind='full')

        # TRY 2
        # cols = cols + ["ANY"]
        # df = pd.DataFrame()
        # for chunk in pd.read_hdf(self._hdf_path, self.group, chunksize=10):
        #     df = pd.concat([df, chunk])
        # print("")

        # with self.ctx(mode='r') as store:
        #     store.create_table_index(self.group, columns=cols)

        print("")

        # with self.ctx(mode="r") as store:
        #     df = store.get(key=self.group)
        # return pd.DataFrame(df)
