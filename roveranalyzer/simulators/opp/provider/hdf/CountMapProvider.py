import platform
import warnings
from typing import List

import pandas
import pandas as pd
from pandas import IndexSlice as _I

from roveranalyzer.simulators.opp.provider.hdf.HdfGroups import HdfGroups
from roveranalyzer.simulators.opp.provider.hdf.IHdfProvider import IHdfProvider
from roveranalyzer.simulators.opp.provider.hdf.Operation import Operation


class CountMapKey:
    ID = "ID"
    SIMTIME = "simtime"
    X = "x"
    Y = "y"
    COUNT = "count"
    ERR = "err"
    OWNER_DIST = "owner_dist"
    SQERR = "sqerr"


class CountMapHdfProvider(IHdfProvider):
    def __init__(self, hdf_path):
        super().__init__(hdf_path)
        self.dispatcher = {
            int: self.handle_primitive,
            float: self.handle_primitive,
            str: self.handle_primitive,
            list: self.handle_list,
            slice: self.handle_slice,
            tuple: self.handle_tuple,
        }

    def handle_primitive(self, key: str, value: any) -> List[str]:
        print("handle_primitive")
        condition = self._build_exact_condition(key=key, value=value)
        print(f"Condition: {condition}")
        return condition

    def handle_list(self, key: str, values: List) -> List[str]:
        print("handle_list")
        condition = []
        for element in values:
            condition += self.dispatch(key=key, item=element)
        print(f"Condition: {condition}")
        return condition

    def handle_slice(self, key: str, value: slice) -> List[str]:
        print("handle_slice")
        if value.step and value.step > 1:
            warnings.warn(
                message=f"Step size of '{value.step}' is not supported. Step size must be '1'."
            )
        condition = self._build_range_condition(
            key=key, _min=value.start, _max=value.stop
        )
        print(f"Condition: {condition}")
        return condition

    def handle_tuple(self, key: str, value: tuple) -> List[str]:
        # because the pandas index slicer returns a tuple
        # it has to be handled manually to pass the correct keys
        raise NotImplementedError("Tuples can't be handled.")

    def dispatch(self, key, item) -> List[str]:
        item_type = type(item)
        f = self.dispatcher[item_type]
        ret = f(key, item)
        return ret

    def __getitem__(self, item: any):
        # TODO: x conditions
        #       [✓] 1. p[2] -> ID (single) (✓)
        #       [✓] 2. p[0:5] -> ID (range 0-5)
        #       [✓] 3. p[slice(0,5,4)] -> ID (range 0-5)  + warning for step_size != 0
        #       [✓] 4. p[I[1,2,3,4]] -> simtime (single), x (single), y (single), ID (single)
        #       [✓] 5. p[I[1,None,None,4]] -> simtime (single), x (ignore), y (ignore), ID (single) + handle None
        #       [✓] 6. p[I[1,2]] -> simtime (single), x(single), y (ignore), ID(ignore) + fill
        #       [✓] 7. p[I[1,2,3,4,5,6,7,8]] -> to many values error
        if type(item) == tuple:
            print("We have a tuple over here")
            if len(item) > len(self.idx_order):
                raise ValueError(
                    f"To many values in tuple. Got: {len(item)} expected: <={len(self.idx_order)} "
                )

            condition: List[str] = []
            for idx in range(len(self.idx_order)):
                tuple_item = item[idx] if len(item) > idx else None
                print(f"idx[{idx}] = {tuple_item}")
                if tuple_item:
                    condition = condition + self.dispatch(
                        self.idx_order[idx], tuple_item
                    )
            print(f"condition: {condition}")
            return self._select_where(condition)
        else:
            condition = self.dispatch(CountMapKey.ID, item)
            return self._select_where(condition)

    def __setitem__(self, key, value):
        raise NotImplementedError("Not supported!")

    def group_key(self) -> str:
        return HdfGroups.COUNT_MAP

    def index_order(self) -> {}:
        return {
            0: CountMapKey.SIMTIME,
            1: CountMapKey.X,
            2: CountMapKey.Y,
            3: CountMapKey.ID,
        }

    #########################
    # Exact value functions #
    #########################
    def select_id_exact(
        self, value: any, operation: str = Operation.EQ
    ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(
            key=CountMapKey.ID, value=value, operation=operation
        )
        return self._select_where(condition=condition)

    def select_simtime_exact(
        self, value: any, operation: str = Operation.EQ
    ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(
            key=CountMapKey.SIMTIME, value=value, operation=operation
        )
        return self._select_where(condition=condition)

    def select_x_exact(self, value: any, operation: str = Operation.EQ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(
            key=CountMapKey.X, value=value, operation=operation
        )
        return self._select_where(condition=condition)

    def select_y_exact(self, value: any, operation: str = Operation.EQ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(
            key=CountMapKey.Y, value=value, operation=operation
        )
        return self._select_where(condition=condition)  # p[I[None,None,5,None]]

    def select_count_exact(
        self, value: any, operation: str = Operation.EQ
    ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(
            key=CountMapKey.COUNT, value=value, operation=operation
        )
        return self._select_where(condition=condition)

    def select_err_exact(
        self, value: any, operation: str = Operation.EQ
    ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(
            key=CountMapKey.ERR, value=value, operation=operation
        )
        return self._select_where(condition=condition)

    def select_owner_dist_exact(
        self, value: any, operation: str = Operation.EQ
    ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(
            key=CountMapKey.OWNER_DIST, value=value, operation=operation
        )
        return self._select_where(condition=condition)

    def select_sqerr_exact(
        self, value: any, operation: str = Operation.EQ
    ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(
            key=CountMapKey.SQERR, value=value, operation=operation
        )
        return self._select_where(condition=condition)

    def select_simtime_and_node_id_exact(
        self, simtime: int, node_id: int, operation: str = Operation.EQ
    ) -> pd.DataFrame:
        condition: List[str] = self._build_exact_condition(
            key=CountMapKey.SIMTIME, value=simtime, operation=operation
        ) + self._build_exact_condition(
            key=CountMapKey.ID, value=node_id, operation=operation
        )
        return self._select_where(condition=condition)

    #########################
    # Range value functions #
    #########################
    def select_id_range(self, _min: int, _max: int) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(
            key=CountMapKey.ID, _min=str(_min), _max=str(_max)
        )
        return self._select_where(condition=condition)

    def select_simtime_range(self, _min: int, _max: int) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(
            key=CountMapKey.SIMTIME, _min=str(_min), _max=str(_max)
        )
        return self._select_where(condition=condition)

    def select_x_range(self, _min: int, _max: int) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(
            key=CountMapKey.X, _min=str(_min), _max=str(_max)
        )
        return self._select_where(condition=condition)

    def select_y_range(self, _min: int, _max: int) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(
            key=CountMapKey.Y, _min=str(_min), _max=str(_max)
        )
        return self._select_where(condition=condition)

    def select_count_range(self, _min: float, _max: float) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(
            key=CountMapKey.COUNT, _min=str(_min), _max=str(_max)
        )
        return self._select_where(condition=condition)

    def select_err_range(self, _min: float, _max: float) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(
            key=CountMapKey.ERR, _min=str(_min), _max=str(_max)
        )
        return self._select_where(condition=condition)

    def select_owner_dist_range(self, _min: float, _max: float) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(
            key=CountMapKey.OWNER_DIST, _min=str(_min), _max=str(_max)
        )
        return self._select_where(condition=condition)

    def select_sqerr_range(self, _min: float, _max: float) -> pd.DataFrame:
        condition: List[str] = self._build_range_condition(
            key=CountMapKey.SQERR, _min=str(_min), _max=str(_max)
        )
        return self._select_where(condition=condition)

    def add_data(self, data: any, name: str, dtype: str = "uint8"):
        import platform

        if platform.system() != "windows":
            # at any point if it's required to safe more than dataframes into hdf files
            # check how to install h5py for windows
            import h5py

            file = h5py.File(self._hdf_path, "a")
            file.create_dataset(f"data/{name}", data=data, dtype=dtype)
            file.close()

    def get_data(self, name: str):
        import platform

        if platform.system() != "windows":
            # at any point if it's required to read other data than dataframes from hdf files
            # check how to install h5py for windows
            import h5py

            file = h5py.File(self._hdf_path, "r")
            ret = file["data"][name][:]
            file.close()
            return ret
