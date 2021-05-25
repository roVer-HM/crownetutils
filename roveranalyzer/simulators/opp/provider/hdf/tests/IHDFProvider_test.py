import os
import shutil
import unittest
import warnings
from unittest.mock import MagicMock, call, patch

import pandas as pd
from utils import create_count_map_dataframe, make_dirs, safe_dataframe_to_hdf

from roveranalyzer.simulators.opp.provider.hdf.CountMapProvider import CountMapProvider
from roveranalyzer.simulators.opp.provider.hdf.HdfGroups import HdfGroups


class IHDFProviderTest(unittest.TestCase):
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

    @patch("pandas.HDFStore")
    def test_ctx_2(self, mock_store: MagicMock):
        mock_store.return_value.close.return_value = None
        with self.provider.ctx() as hdf_store:
            hdf_store.info()
        mock_store.return_value.close.assert_called()
        mock_store.assert_called_once_with(
            self.sample_file_dir,
            mode="a",
            complevel=self.provider._hdf_args["complevel"],
            complib=self.provider._hdf_args["complib"],
        )

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._build_exact_condition"
    )
    def test_handle_primitive(self, mock_exact_condition: MagicMock):
        key = "any_key"
        value = 1
        condition = f"{key}={value}"
        mock_exact_condition.return_value = condition
        result_condition, result_columns = self.provider._handle_primitive(
            key=key, value=value
        )
        mock_exact_condition.assert_called_once_with(key=key, value=value)
        self.assertEquals(result_condition, condition)
        self.assertEquals(result_columns, None)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider.dispatch"
    )
    def test_handle_list(self, mock_dispatch: MagicMock):
        key = "any_key"
        values = [1, "2", 3.0, None]
        conditions = [f"{key}={values[0]}", f"{key}={values[1]}", f"{key}={values[2]}"]
        mock_dispatch.side_effect = [
            [conditions[0]],
            [conditions[1]],
            [conditions[2]],
            [],
        ]
        result_condition, result_columns = self.provider._handle_list(
            key=key, values=values
        )
        calls = [
            call(key="any_key", item=values[0]),
            call(key="any_key", item=values[1]),
            call(key="any_key", item=values[2]),
            call(key="any_key", item=values[3]),
        ]
        mock_dispatch.assert_has_calls(calls)
        self.assertEquals(result_condition, conditions)
        self.assertEquals(result_columns, None)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._build_range_condition"
    )
    def test_handle_slice(self, mock_range_condition: MagicMock):
        key = "any_key"
        _min = 0
        _max = 4
        # valid slice
        value_valid = slice(_min, _max)
        condition = [f"{key}>={_min}", f"{key}<={_max}"]
        mock_range_condition.return_value = condition
        result_condition, result_columns = self.provider._handle_slice(
            key=key, value=value_valid
        )
        mock_range_condition.assert_called_once_with(
            key=key, _min=value_valid.start, _max=value_valid.stop
        )
        self.assertEquals(result_condition, condition)
        self.assertEquals(result_columns, None)

        # invalid slice
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value_warning = slice(_min, _max, 2)
            self.provider._handle_slice(key=key, value=value_warning)
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "Step size" in str(w[-1].message)


if __name__ == "__main__":
    unittest.main()
