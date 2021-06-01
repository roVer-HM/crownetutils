import os
import shutil
import unittest

import pandas as pd
from pandas import IndexSlice as _I
from utils import create_count_map_dataframe, make_dirs, safe_dataframe_to_hdf

from roveranalyzer.simulators.opp.provider.hdf.CountMapProvider import CountMapProvider
from roveranalyzer.simulators.opp.provider.hdf.HdfGroups import HdfGroups


class IHDFProviderGoldenSampleTest(unittest.TestCase):
    test_out_dir: str = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "unittest"
    )
    sample_file_dir: str = os.path.join(test_out_dir, "sample.hdf5")
    provider: CountMapProvider = CountMapProvider(sample_file_dir)
    sample_dataframe: pd.DataFrame = create_count_map_dataframe()

    @classmethod
    def setUpClass(cls):
        make_dirs(cls.test_out_dir)
        safe_dataframe_to_hdf(
            cls.sample_dataframe, HdfGroups.COUNT_MAP, cls.sample_file_dir
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_out_dir)

    @unittest.skip("TODO: FIX ME to match the new sample dataframe")
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
        test__node_and_id_dataframe = self.provider.select_simtime_and_node_id_exact(
            42, 43
        )

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
        self.assertTrue(
            self.sample_dataframe[50:51].equals(test__node_and_id_dataframe)
        )
        self.assertEquals(len(self.provider.select_simtime_exact(42)), 2)

    @unittest.skip("TODO: FIX ME to match the new sample dataframe")
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

    def test_index_slicer(self):
        provider = self.provider
        case_1 = provider[42]  # ['simtime=42']
        case_1_pd = provider[_I[42]]  # ['simtime=42']
        sample_1 = self.sample_dataframe.iloc[[42, 43, 44, 45, 46, 47, 48, 49]]
        self.assertTrue(case_1.equals(case_1_pd))
        self.assertTrue(case_1.equals(sample_1))
        self.assertTrue(case_1_pd.equals(sample_1))

        case_2 = provider[0:10]  # ['simtime<=10', 'simtime>=0']
        case_2_pd = provider[_I[0:10]]  # ['simtime<=10', 'simtime>=0']
        sample_2 = self.sample_dataframe.iloc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        self.assertTrue(case_2.equals(case_2_pd))
        self.assertTrue(case_2.equals(sample_2))
        self.assertTrue(case_2_pd.equals(sample_2))

        case_3 = provider[42, 42.0]  # ['simtime=42', 'x=42']
        case_3_pd = provider[_I[42, 42.0]]  # ['simtime=42', 'x=42']
        sample_3 = self.sample_dataframe.iloc[[42, 43, 44, 45]]
        self.assertTrue(case_3.equals(case_3_pd))
        self.assertTrue(case_3.equals(sample_3))
        self.assertTrue(case_3_pd.equals(sample_3))

        case_4 = provider[42, None, 42.0]  # ['simtime=42', 'y=42']
        case_4_pd = provider[_I[42, None, 42.0]]  # ['simtime=42', 'y=42']
        sample_4 = self.sample_dataframe.iloc[[42, 43, 46, 47]]
        self.assertTrue(case_4.equals(case_4_pd))
        self.assertTrue(case_4.equals(sample_4))
        self.assertTrue(case_4_pd.equals(sample_4))

        case_5 = provider[42, 43.0, 43.0]  # ['simtime=42', x='43', y=43']
        case_5_pd = provider[_I[42, 43.0, 43.0]]  # ['simtime=42', x='43', y=43']
        sample_5 = self.sample_dataframe.iloc[[48, 49]]
        self.assertTrue(case_5.equals(case_5_pd))
        self.assertTrue(case_5.equals(sample_5))
        self.assertTrue(case_5_pd.equals(sample_5))

        case_6 = provider[42, 43.0, 43.0, 43]  # ['simtime=42', x='43', y=43', ID='43']
        case_6_pd = provider[
            _I[42, 43.0, 43.0, 43]
        ]  # ['simtime=42', x='43', y=43', ID='43']
        sample_6 = self.sample_dataframe.iloc[[49]]
        self.assertTrue(case_6.equals(case_6_pd))
        self.assertTrue(case_6.equals(sample_6))
        self.assertTrue(case_6_pd.equals(sample_6))

        # case 7 - to many index entries
        with self.assertRaises(ValueError) as context:
            provider[42, 42.0, 42.0, 42.0, 42.0]
            self.assertTrue("To many values in tuple." in str(context.exception))
        with self.assertRaises(ValueError) as context:
            provider[_I[42, 42.0, 42.0, 42.0, 42.0]]
            self.assertTrue("To many values in tuple." in str(context.exception))

        case_8 = provider[
            42, 42.0:43.0, 42.0:43.0, 43
        ]  # ['simtime=42', x='43', y=43', ID='43']
        case_8_pd = provider[
            _I[42, 42.0:43.0, 42.0:43.0, 43]
        ]  # ['simtime=42', x='43', y=43', ID='43']
        sample_8 = self.sample_dataframe.iloc[[43, 45, 47, 49]]
        self.assertTrue(case_8.equals(case_8_pd))
        self.assertTrue(case_8.equals(sample_8))
        self.assertTrue(case_8_pd.equals(sample_8))

        case_9 = provider[[0, 1, 5]]  # ['simtime in [0, 1, 5]']
        case_9_pd = provider[_I[[0, 1, 5]]]  # ['simtime in [0, 1, 5]']
        sample_9 = self.sample_dataframe.iloc[[0, 1, 5]]  # ['simtime in [0, 1, 5]']
        self.assertTrue(case_9.equals(case_9_pd))
        self.assertTrue(case_9.equals(sample_9))
        self.assertTrue(case_9_pd.equals(sample_9))

        # ['simtime=42', 'x in [42.0, 43.0]', 'y in [42.0, 43.0]', 'ID=42.0']
        case_10 = provider[42, [42.0, 43.0], [42.0, 43.0], 42.0]
        case_10_pd = provider[_I[42, [42.0, 43.0], [42.0, 43.0], 42.0]]
        sample_10 = self.sample_dataframe.iloc[[42, 44, 46, 48]]
        self.assertTrue(case_10.equals(case_10_pd))
        self.assertTrue(case_10.equals(sample_10))
        self.assertTrue(case_10_pd.equals(sample_10))

    def test_index_slicer_with_columns(self):
        # sample dataframe
        # 42 (42,42,42,42)
        # 43 (42,42,42,43)
        # 44 (42,42,43,42)
        # 45 (42,42,43,43)
        # 46 (42,43,42,42)
        # 47 (42,43,42,43)
        # 48 (42,43,43,42)
        # 49 (42,43,43,43)
        # 50 (43,43,43,43)

        # case15 = provider[_I[[1, 5, 10], None, [1, 4]], _I["err"]]  # condition: ['simtime=2', 'x=6.0', 'y=12.0', 'ID<=30', 'ID>=10'], columns: ['err']
        # case16 = provider[_I[[1, 5, 10], None, [1, 4]], _I["not_existing"]]  # condition: ['simtime=2', 'x=6.0', 'y=12.0', 'ID<=30', 'ID>=10'], columns: ['err']
        # case17 = provider[_I[0:10, 6.0], ["err", "sqerr"]]  # condition: ['simtime<=10', 'simtime>=0', 'x=6.0'], columns: ['err', 'sqerr']
        # case18 = provider[_I[0:10], ["err", "sqerr"]]  # throwing error

        # column cases
        # row_slice = _I[2, 6.0, 12.0, 10:30]
        # col_slice = ["err"]
        # dataframe = provider[row_slice, col_slice]
        # dataframe = provider[_I[0:10], ["err"]]
        # dataframe = provider[_I[0:10, 6.0, 12.0, 4]]
        # dataframe = provider[_I[0:10, 6.0, 12.0, 4], _I["sqerr"]]
        # dataframe = provider[_I[0:10, 6.0, 12.0, 4], _I["sqerr"]]
        # dataframe = provider[_I[0:10, 6.0, 12.0, 4], "sqerr"]
        # dataframe = provider[_I[0:10, 6.0, 12.0, 4], _I["err", "sqerr", "owner_dist", "count"]]
        # dataframe = provider[_I[0:10, 6.0, 12.0, 4], _I["err", "sqerr"]]
        # dataframe = provider[_I[0:10, 6.0], _I["err", "sqerr"]]
        # dataframe = provider[_I[4], _I["err", "sqerr"]]
        pass


if __name__ == "__main__":
    unittest.main()
