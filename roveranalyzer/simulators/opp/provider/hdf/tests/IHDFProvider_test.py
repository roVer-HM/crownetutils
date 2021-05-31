import os
import shutil
import unittest
import warnings
from unittest.mock import MagicMock, call, patch

import pandas as pd
from utils import create_count_map_dataframe, make_dirs, safe_dataframe_to_hdf

from roveranalyzer.simulators.opp.provider.hdf.CountMapProvider import (
    CountMapKey,
    CountMapProvider,
)
from roveranalyzer.simulators.opp.provider.hdf.HdfGroups import HdfGroups
from roveranalyzer.simulators.opp.provider.hdf.Operation import Operation


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

    def test_cast_to_set(self):
        result_1 = self.provider.cast_to_set(1)
        result_2 = self.provider.cast_to_set(1.0)
        result_3 = self.provider.cast_to_set("1")
        result_4 = self.provider.cast_to_set({1})
        result_5 = self.provider.cast_to_set((1, 2))

        self.assertEqual(result_1, {1})
        self.assertEqual(result_2, {1.0})
        self.assertEqual(result_3, {"1"})
        self.assertEqual(result_4, {1})
        self.assertEqual(result_5, {1, 2})

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
        self.assertEqual(result_condition, condition)
        self.assertEqual(result_columns, None)

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
        self.assertEqual(result_condition, conditions)
        self.assertEqual(result_columns, None)

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
        self.assertEqual(result_condition, condition)
        self.assertEqual(result_columns, None)

        # invalid slice
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value_warning = slice(_min, _max, 2)
            self.provider._handle_slice(key=key, value=value_warning)
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "Step size" in str(w[-1].message)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider.dispatch"
    )
    def test_handle_index_tuple(self, mock_dispatch: MagicMock):
        # tuple to long
        with self.assertRaises(ValueError) as context:
            invalid_len_tuple = (1, 2, 3, 4, 5)
            self.provider._handle_index_tuple(invalid_len_tuple)
            self.assertTrue("To many values in tuple." in str(context.exception))
            mock_dispatch.assert_not_called()

        # mock return values
        valid_tuple_complete = (1, 2, 3, 4)
        valid_tuple_missing = (1, 2)
        valid_tuple_none_values = (1, None, None, 4)
        cond_1 = f"{CountMapKey.SIMTIME}={1}"
        cond_2 = f"{CountMapKey.X}={2}"
        cond_3 = f"{CountMapKey.Y}={3}"
        cond_4 = f"{CountMapKey.ID}={4}"
        mock_dispatch.side_effect = [
            [[cond_1]],
            [[cond_2]],
            [[cond_3]],
            [[cond_4]],
            [[cond_1]],
            [[cond_2]],
            [[cond_1]],
            [[cond_4]],
        ]

        calls = [
            call(self.provider.idx_order[0], valid_tuple_complete[0]),
            call(self.provider.idx_order[1], valid_tuple_complete[1]),
            call(self.provider.idx_order[2], valid_tuple_complete[2]),
            call(self.provider.idx_order[3], valid_tuple_complete[3]),
            call(self.provider.idx_order[0], valid_tuple_missing[0]),
            call(self.provider.idx_order[1], valid_tuple_missing[1]),
            call(self.provider.idx_order[0], valid_tuple_none_values[0]),
            call(self.provider.idx_order[3], valid_tuple_none_values[3]),
        ]

        result_1 = self.provider._handle_index_tuple(valid_tuple_complete)
        result_2 = self.provider._handle_index_tuple(valid_tuple_missing)
        result_3 = self.provider._handle_index_tuple(valid_tuple_none_values)

        mock_dispatch.assert_has_calls(calls, any_order=True)
        self.assertEqual(mock_dispatch.mock_calls[0 : len(calls)], calls)
        self.assertEqual(result_1, [cond_1, cond_2, cond_3, cond_4])
        self.assertEqual(result_2, [cond_1, cond_2])
        self.assertEqual(result_3, [cond_1, cond_4])

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider.dispatch"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._handle_index_tuple"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider.cast_to_set"
    )
    def test_handle_tuple(
        self,
        mock_cast_set: MagicMock,
        mock_handle_index_tuple: MagicMock,
        mock_dispatch: MagicMock,
    ):
        tuple_with_columns = (1, CountMapKey.COUNT)
        expected_condition = [f"{CountMapKey.SIMTIME}={1}"]
        mock_cast_set.side_effect = [{self.provider.columns()[0]}, {2}]
        mock_dispatch.return_value = [expected_condition]
        result_condition, result_columns = self.provider._handle_tuple(
            self.provider.default_index_key(), tuple_with_columns
        )
        mock_dispatch.assert_called_once_with(
            self.provider.default_index_key(), tuple_with_columns[0]
        )
        self.assertEqual(result_condition, expected_condition)
        self.assertEqual(result_columns, [CountMapKey.COUNT])

        tuple_only_index = (1, 2, 3, 4)
        expected_condition = [
            f"{CountMapKey.SIMTIME}={1}",
            f"{CountMapKey.X}={2}",
            f"{CountMapKey.Y}={3}",
            f"{CountMapKey.ID}={4}",
        ]
        mock_handle_index_tuple.return_value = expected_condition
        result_condition, result_columns = self.provider._handle_tuple(
            self.provider.default_index_key(), tuple_only_index
        )
        self.assertEqual(result_condition, expected_condition)
        self.assertEqual(result_columns, None)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider.dispatcher"
    )
    def test_dispatch(self, mock_dispatcher: MagicMock):
        expected_condition = ["condition"]
        expected_columns = ["columns"]
        mock_dispatcher[int].return_value = (expected_condition, expected_columns)
        result_condition, result_columns = self.provider.dispatch("Any", "Any")
        self.assertEqual(expected_condition, result_condition)
        self.assertEqual(expected_columns, result_columns)

    def test_dispatch_types(self):
        value_int = 1
        value_float = 1.0
        value_string = "1"
        value_list = [1, 2]
        value_slice = slice(1, 4, 1)
        value_tuple = (1, 2, 3, 4)
        value_tuple_in_tuple = ((1, 2), CountMapKey.COUNT)

        handle_primitive_name = self.provider._handle_primitive.__name__
        handle_slice_name = self.provider._handle_slice.__name__
        handle_list_name = self.provider._handle_list.__name__
        handle_tuple_name = self.provider._handle_tuple.__name__

        self.assertEqual(
            self.provider.dispatcher[type(value_int)].__name__, handle_primitive_name
        )
        self.assertEqual(
            self.provider.dispatcher[type(value_float)].__name__, handle_primitive_name
        )
        self.assertEqual(
            self.provider.dispatcher[type(value_string)].__name__, handle_primitive_name
        )
        self.assertEqual(
            self.provider.dispatcher[type(value_list)].__name__, handle_list_name
        )
        self.assertEqual(
            self.provider.dispatcher[type(value_slice)].__name__, handle_slice_name
        )
        self.assertEqual(
            self.provider.dispatcher[type(value_tuple)].__name__, handle_tuple_name
        )
        self.assertEqual(
            self.provider.dispatcher[type(value_tuple_in_tuple)].__name__,
            handle_tuple_name,
        )

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider.dispatch"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._select_where"
    )
    @patch("builtins.print")
    def test_get_item(
        self,
        print_mock: MagicMock,
        mock_select_where: MagicMock,
        mock_dispatch: MagicMock,
    ):
        expected_condition = ["condition"]
        expected_columns = ["columns"]
        call_value = 1
        mock_dispatch.return_value = (expected_condition, expected_columns)
        mock_select_where.return_value = self.sample_dataframe
        result_dataframe = self.provider[call_value]
        mock_dispatch.assert_called_once_with(
            self.provider.default_index_key(), call_value
        )
        mock_select_where.assert_called_once_with(expected_condition, expected_columns)
        print_mock.assert_called_once_with(
            f"hdf select condition: {expected_condition}, "
            f"columns: {'[]' if expected_columns is None else expected_columns}"
        )
        self.assertTrue(result_dataframe.equals(self.sample_dataframe))

        # test empty dataframe return
        with self.assertRaises(ValueError) as context:
            mock_select_where.reset_mock()
            mock_select_where.return_value = pd.DataFrame()
            result_dataframe = self.provider[call_value]
            mock_select_where.assert_not_called()
            self.assertTrue(result_dataframe.empty)
            self.assertTrue("Returned dataframe is empty." in str(context.exception))

    def test_set_item(self):
        with self.assertRaises(NotImplementedError) as context:
            self.provider[1] = "Any"
            self.assertTrue("Not supported!" in str(context.exception))

    def test_get_dataframe(self):
        # Note: pandas isn't mocked because it is important to check the read functionality at least once
        self.assertTrue(self.provider.get_dataframe().equals(self.sample_dataframe))

    def test_write_dataframe(self):
        write_path = os.path.join(self.test_out_dir, "write_test.hdf5")
        write_provider = CountMapProvider(write_path)
        write_provider.write_dataframe(self.sample_dataframe)
        self.assertTrue(os.path.isfile(write_path))

    def test_exists(self):
        non_existing_path_provider = CountMapProvider(
            os.path.join(self.test_out_dir, "non_existing_path.hdf5")
        )
        self.assertTrue(self.provider.exists())
        self.assertFalse(non_existing_path_provider.exists())

    @patch("pandas.HDFStore.select")
    def test_select_where(self, mock_select: MagicMock):
        condition = ["condition"]
        columns = ["columns"]
        mock_select.return_value = self.sample_dataframe
        result_dataframe = self.provider._select_where(condition, columns)
        mock_select.assert_called_once_with(
            key=self.provider.group_key(), where=condition, columns=columns
        )
        self.assertTrue(result_dataframe.equals(self.sample_dataframe))

    def test_build_range_condition(self):
        key = CountMapKey.SIMTIME
        _min = 0
        _max = 5
        expected_condition = [f"{key}<={str(_max)}", f"{key}>={str(_min)}"]
        result_condition = self.provider._build_range_condition(key, _min, _max)
        self.assertEqual(expected_condition, result_condition)

    def test_build_exact_condition(self):
        key = CountMapKey.SIMTIME
        value = 42
        list_value = [42, 43]
        expected_gr = [f"{key}{Operation.GREATER}{value}"]
        expected_sm = [f"{key}{Operation.SMALLER}{value}"]
        expected_greq = [f"{key}{Operation.GREATER_EQ}{value}"]
        expected_smeq = [f"{key}{Operation.SMALLER_EQ}{value}"]
        expected_eq = [f"{key}{Operation.EQ}{value}"]
        expected_list = [f"{key} in {list_value}"]

        result_gr = self.provider._build_exact_condition(key, value, Operation.GREATER)
        result_sm = self.provider._build_exact_condition(key, value, Operation.SMALLER)
        result_greq = self.provider._build_exact_condition(
            key, value, Operation.GREATER_EQ
        )
        result_smeq = self.provider._build_exact_condition(
            key, value, Operation.SMALLER_EQ
        )
        result_eq = self.provider._build_exact_condition(key, value, Operation.EQ)
        result_default = self.provider._build_exact_condition(key, value)
        result_list = self.provider._build_exact_condition(key, list_value)

        self.assertEqual(result_gr, expected_gr)
        self.assertEqual(result_sm, expected_sm)
        self.assertEqual(result_greq, expected_greq)
        self.assertEqual(result_smeq, expected_smeq)
        self.assertEqual(result_eq, expected_eq)
        self.assertEqual(result_default, expected_eq)
        self.assertEqual(result_list, expected_list)


if __name__ == "__main__":
    unittest.main()
