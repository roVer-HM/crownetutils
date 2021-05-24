import os
import shutil
import unittest
from typing import Any, List, TextIO, Tuple

import numpy as np
import pandas as pd

from roveranalyzer.simulators.opp.provider.hdf.CountMapProvider import (
    CountMapKey,
    CountMapProvider,
)
from roveranalyzer.simulators.opp.provider.hdf.HdfGroups import HdfGroups


def make_dirs(path: Any) -> None:
    if isinstance(path, str):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    elif isinstance(path, list):
        for p in path:
            make_dirs(p)
    else:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)


def create_sample_dataframe(number_entries: int = 50) -> pd.DataFrame:
    idxs = [i for i in range(number_entries)]
    xs = [float(i) for i in range(number_entries)]
    ys = [float(i) for i in range(number_entries)]
    ids = [i for i in range(number_entries)]
    entries = np.array([xs, xs, xs, xs]).transpose()
    mult_idx = pd.MultiIndex.from_arrays(
        [idxs, xs, ys, ids], names=["simtime", "x", "y", "ID"]
    )
    df = pd.DataFrame(
        entries, index=mult_idx, columns=["count", "err", "owner_dist", "sqerr"]
    )
    df.loc[(42, 42.0, 42.0, 43)] = [42.0, 42.0, 42.0, 42.0]
    return df


def safe_dataframe_to_hdf(dataframe: pd.DataFrame, path: str) -> None:
    dataframe.to_hdf(
        path_or_buf=path, key=HdfGroups.COUNT_MAP, format="table", data_columns=True
    )


class CountMapProviderTest(unittest.TestCase):
    test_out_dir: str = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "unittest"
    )
    sample_file_dir: str = os.path.join(test_out_dir, "sample.hdf5")
    test_file_dir: str = os.path.join(test_out_dir, "test.hdf5")
    provider: CountMapProvider = CountMapProvider(sample_file_dir)
    sample_dataframe: pd.DataFrame = create_sample_dataframe()

    @classmethod
    def setUpClass(cls):
        make_dirs(cls.test_out_dir)
        safe_dataframe_to_hdf(cls.sample_dataframe, cls.sample_file_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_out_dir)

    def test_CountMapProperties(self):
        sample_grp_key = HdfGroups.COUNT_MAP
        sample_default_index = CountMapKey.SIMTIME
        sample_index_order = {
            0: CountMapKey.SIMTIME,
            1: CountMapKey.X,
            2: CountMapKey.Y,
            3: CountMapKey.ID,
        }
        result_grp_key = self.provider.group_key()
        result_index_order = self.provider.index_order()
        result_default_index = self.provider.default_index_key()

        self.assertEqual(result_grp_key, sample_grp_key)
        self.assertEqual(result_index_order, sample_index_order)
        self.assertEqual(result_default_index, sample_default_index)

    def test_exact_methods(self):
        simtime: int = 1
        x: float = 2.0
        y: float = 3.0
        id: int = 4
        count: float = 5.0
        err: float = 6.0
        owner: float = 7.0
        sqerr: float = 8.0

        test_simtime_dataframe = self.provider.select_simtime_exact(simtime)
        test_x_dataframe = self.provider.select_x_exact(x)
        test_y_dataframe = self.provider.select_y_exact(y)
        test_id_dataframe = self.provider.select_id_exact(id)
        test_count_dataframe = self.provider.select_count_exact(count)
        test_err_dataframe = self.provider.select_err_exact(err)
        test_owner_dataframe = self.provider.select_owner_dist_exact(owner)
        test_sqerr_dataframe = self.provider.select_sqerr_exact(sqerr)

        self.assertTrue(
            self.sample_dataframe[simtime : simtime + 1].equals(test_simtime_dataframe)
        )
        self.assertTrue(
            self.sample_dataframe[int(x) : int(x) + 1].equals(test_x_dataframe)
        )
        self.assertTrue(
            self.sample_dataframe[int(y) : int(y) + 1].equals(test_y_dataframe)
        )
        self.assertTrue(self.sample_dataframe[id : id + 1].equals(test_id_dataframe))
        self.assertTrue(
            self.sample_dataframe[int(count) : int(count) + 1].equals(
                test_count_dataframe
            )
        )
        self.assertTrue(
            self.sample_dataframe[int(err) : int(err) + 1].equals(test_err_dataframe)
        )
        self.assertTrue(
            self.sample_dataframe[int(owner) : int(owner) + 1].equals(
                test_owner_dataframe
            )
        )
        self.assertTrue(
            self.sample_dataframe[int(sqerr) : int(sqerr) + 1].equals(
                test_sqerr_dataframe
            )
        )
        self.assertEquals(len(self.provider.select_simtime_exact(42)), 2)

    def test_range_methods(self):
        _range: int = 5
        simtime: int = 1
        x: float = 2.0
        y: float = 3.0
        id: int = 4
        count: int = 5
        err: int = 6
        owner = 7
        sqerr = 8

        test_simtime_dataframe = self.provider.select_simtime_range(
            simtime, simtime + _range
        )
        test_x_dataframe = self.provider.select_x_range(x, x + _range)
        test_y_dataframe = self.provider.select_y_range(y, y + _range)
        test_id_dataframe = self.provider.select_id_range(id, id + _range)
        test_count_dataframe = self.provider.select_count_range(count, count + _range)
        test_err_dataframe = self.provider.select_err_range(err, err + _range)
        test_owner_dataframe = self.provider.select_owner_dist_range(
            owner, owner + _range
        )
        test_sqerr_dataframe = self.provider.select_sqerr_range(sqerr, sqerr + _range)

        self.assertTrue(
            self.sample_dataframe[simtime : simtime + 1 + _range].equals(
                test_simtime_dataframe
            )
        )
        self.assertTrue(
            self.sample_dataframe[int(x) : int(x) + 1 + _range].equals(test_x_dataframe)
        )
        self.assertTrue(
            self.sample_dataframe[int(y) : int(y) + 1 + _range].equals(test_y_dataframe)
        )
        self.assertTrue(
            self.sample_dataframe[id : id + 1 + _range].equals(test_id_dataframe)
        )
        self.assertTrue(
            self.sample_dataframe[count : count + 1 + _range].equals(
                test_count_dataframe
            )
        )
        self.assertTrue(
            self.sample_dataframe[err : err + 1 + _range].equals(test_err_dataframe)
        )
        self.assertTrue(
            self.sample_dataframe[owner : owner + 1 + _range].equals(
                test_owner_dataframe
            )
        )
        self.assertTrue(
            self.sample_dataframe[sqerr : sqerr + 1 + _range].equals(
                test_sqerr_dataframe
            )
        )
        self.assertEqual(len(self.provider.select_simtime_range(42, 43)), 3)

    def test_select_simtime_and_node_id_exact(self):
        test_dataframe = self.provider.select_simtime_and_node_id_exact(42, 43)
        sample_dataframe = self.sample_dataframe[50:51]
        self.assertTrue(test_dataframe.equals(sample_dataframe))


if __name__ == "__main__":
    unittest.main()
