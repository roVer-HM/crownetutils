import os
import unittest
from unittest.mock import MagicMock, call, patch

import pandas as pd
from fs.tempfs import TempFS

from crownetutils.analysis.dpmm.hdf.dpmm_count_provider import DpmmCount, DpmmCountKey
from crownetutils.analysis.dpmm.tests.utils import (
    create_count_map_dataframe,
    create_tmp_fs,
    make_dirs,
    safe_dataframe_to_hdf,
)
from crownetutils.analysis.hdf.groups import HdfGroups
from crownetutils.analysis.hdf.operator import Operation


class DpmmCountProviderTest(unittest.TestCase):
    # create tmp fs. (use fs.root_path to access as normal path)
    fs: TempFS = create_tmp_fs("DcdMapCountProviderTest")
    test_out_dir: str = os.path.join(fs.root_path, "unittest")
    sample_file_dir: str = os.path.join(test_out_dir, "sample.hdf5")
    provider: DpmmCount = DpmmCount(sample_file_dir)
    sample_dataframe: pd.DataFrame = create_count_map_dataframe()
    index_range = 5
    simtime: int = 1
    x: float = 2.0
    y: float = 3.0
    _id: int = 4
    count: float = 5.0
    err: float = 6.0
    owner: float = 7.0
    sqerr: float = 8.0

    @classmethod
    def setUpClass(cls):
        make_dirs(cls.test_out_dir)
        safe_dataframe_to_hdf(
            cls.sample_dataframe, HdfGroups.COUNT_MAP, cls.sample_file_dir
        )

    @classmethod
    def tearDownClass(cls):
        # close temporary filesystem
        cls.fs.close()

    def test_CountMapProperties(self):
        sample_grp_key = HdfGroups.COUNT_MAP
        sample_default_index = DpmmCountKey.SIMTIME
        sample_index_order = {
            0: DpmmCountKey.SIMTIME,
            1: DpmmCountKey.X,
            2: DpmmCountKey.Y,
            3: DpmmCountKey.ID,
        }
        sample_columns = [
            DpmmCountKey.COUNT,
            DpmmCountKey.ERR,
            DpmmCountKey.OWNER_DIST,
            DpmmCountKey.SQERR,
            DpmmCountKey.MISSING_VAL,
        ]
        result_grp_key = self.provider.group_key()
        result_index_order = self.provider.index_order()
        result_default_index = self.provider.default_index_key()
        result_columns = self.provider.columns()

        self.assertEqual(result_grp_key, sample_grp_key)
        self.assertEqual(result_index_order, sample_index_order)
        self.assertEqual(result_default_index, sample_default_index)
        self.assertEqual(result_columns, sample_columns)

    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._build_exact_condition")
    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._select_where")
    def test_select_id_exact(
        self, mock_select_where: MagicMock, mock_build_exact_condition: MagicMock
    ):
        condition = [f"{DpmmCountKey.ID}{Operation.EQ}{self._id}"]
        mock_build_exact_condition.return_value = condition
        self.provider.select_id_exact(self._id)
        mock_build_exact_condition.assert_called_once_with(
            key=DpmmCountKey.ID, value=self._id, operation=Operation.EQ
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._build_exact_condition")
    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._select_where")
    def test_select_simtime_exact(
        self, mock_select_where: MagicMock, mock_build_exact_condition: MagicMock
    ):
        condition = [f"{DpmmCountKey.SIMTIME}{Operation.EQ}{self.simtime}"]
        mock_build_exact_condition.return_value = condition
        self.provider.select_simtime_exact(self.simtime)
        mock_build_exact_condition.assert_called_once_with(
            key=DpmmCountKey.SIMTIME, value=self.simtime, operation=Operation.EQ
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._build_exact_condition")
    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._select_where")
    def test_select_x_exact(
        self, mock_select_where: MagicMock, mock_build_exact_condition: MagicMock
    ):
        condition = [f"{DpmmCountKey.X}{Operation.EQ}{self.x}"]
        mock_build_exact_condition.return_value = condition
        self.provider.select_x_exact(self.x)
        mock_build_exact_condition.assert_called_once_with(
            key=DpmmCountKey.X, value=self.x, operation=Operation.EQ
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._build_exact_condition")
    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._select_where")
    def test_select_y_exact(
        self, mock_select_where: MagicMock, mock_build_exact_condition: MagicMock
    ):
        condition = [f"{DpmmCountKey.Y}{Operation.EQ}{self.y}"]
        mock_build_exact_condition.return_value = condition
        self.provider.select_y_exact(self.y)
        mock_build_exact_condition.assert_called_once_with(
            key=DpmmCountKey.Y, value=self.y, operation=Operation.EQ
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._build_exact_condition")
    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._select_where")
    def test_select_count_exact(
        self, mock_select_where: MagicMock, mock_build_exact_condition: MagicMock
    ):
        condition = [f"{DpmmCountKey.COUNT}{Operation.EQ}{self.count}"]
        mock_build_exact_condition.return_value = condition
        self.provider.select_count_exact(self.count)
        mock_build_exact_condition.assert_called_once_with(
            key=DpmmCountKey.COUNT, value=self.count, operation=Operation.EQ
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._build_exact_condition")
    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._select_where")
    def test_select_err_exact(
        self, mock_select_where: MagicMock, mock_build_exact_condition: MagicMock
    ):
        condition = [f"{DpmmCountKey.ERR}{Operation.EQ}{self.err}"]
        mock_build_exact_condition.return_value = condition
        self.provider.select_err_exact(self.err)
        mock_build_exact_condition.assert_called_once_with(
            key=DpmmCountKey.ERR, value=self.err, operation=Operation.EQ
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._build_exact_condition")
    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._select_where")
    def test_owner_dist_exact(
        self, mock_select_where: MagicMock, mock_build_exact_condition: MagicMock
    ):
        condition = [f"{DpmmCountKey.OWNER_DIST}{Operation.EQ}{self.owner}"]
        mock_build_exact_condition.return_value = condition
        self.provider.select_owner_dist_exact(self.owner)
        mock_build_exact_condition.assert_called_once_with(
            key=DpmmCountKey.OWNER_DIST, value=self.owner, operation=Operation.EQ
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._build_exact_condition")
    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._select_where")
    def test_select_sqerr_exact(
        self, mock_select_where: MagicMock, mock_build_exact_condition: MagicMock
    ):
        condition = [f"{DpmmCountKey.SQERR}{Operation.EQ}{self.sqerr}"]
        mock_build_exact_condition.return_value = condition
        self.provider.select_sqerr_exact(self.sqerr)
        mock_build_exact_condition.assert_called_once_with(
            key=DpmmCountKey.SQERR, value=self.sqerr, operation=Operation.EQ
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._build_exact_condition")
    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._select_where")
    def test_select_simtime_and_node_id_exact(
        self, mock_select_where: MagicMock, mock_build_exact_condition: MagicMock
    ):
        simtime = 42
        ID = 43
        mock_build_exact_condition.side_effect = [
            [f"{DpmmCountKey.SIMTIME}={simtime}"],
            [f"{DpmmCountKey.ID}={ID}"],
        ]
        self.provider.select_simtime_and_node_id_exact(simtime, ID)
        calls = [
            call(key=f"{DpmmCountKey.SIMTIME}", value=simtime, operation=Operation.EQ),
            call(key=f"{DpmmCountKey.ID}", value=ID, operation=Operation.EQ),
        ]
        mock_build_exact_condition.mock_has_calls(calls)
        condition = [f"{DpmmCountKey.SIMTIME}={simtime}", f"{DpmmCountKey.ID}={ID}"]
        mock_select_where.assert_called_once_with(condition=condition)

    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._build_range_condition")
    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._select_where")
    def test_select_id_range(
        self, mock_select_where: MagicMock, mock_build_range_condition: MagicMock
    ):
        condition = [
            f"{DpmmCountKey.ID}{Operation.GREATER_EQ}{self._id}",
            f"{DpmmCountKey.ID}{Operation.SMALLER_EQ}{self._id + self.index_range}",
        ]
        mock_build_range_condition.return_value = condition
        self.provider.select_id_range(self._id, self._id + self.index_range)
        mock_build_range_condition.assert_called_once_with(
            key=DpmmCountKey.ID, _min=self._id, _max=self._id + self.index_range
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._build_range_condition")
    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._select_where")
    def test_select_simtime_range(
        self, mock_select_where: MagicMock, mock_build_range_condition: MagicMock
    ):
        condition = [
            f"{DpmmCountKey.SIMTIME}{Operation.GREATER_EQ}{self.simtime}",
            f"{DpmmCountKey.SIMTIME}{Operation.SMALLER_EQ}{self.simtime + self.index_range}",
        ]
        mock_build_range_condition.return_value = condition
        self.provider.select_simtime_range(
            self.simtime, self.simtime + self.index_range
        )
        mock_build_range_condition.assert_called_once_with(
            key=DpmmCountKey.SIMTIME,
            _min=self.simtime,
            _max=self.simtime + self.index_range,
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._build_range_condition")
    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._select_where")
    def test_select_x_range(
        self, mock_select_where: MagicMock, mock_build_range_condition: MagicMock
    ):
        condition = [
            f"{DpmmCountKey.X}{Operation.GREATER_EQ}{self.x}",
            f"{DpmmCountKey.X}{Operation.SMALLER_EQ}{self.x + self.index_range}",
        ]
        mock_build_range_condition.return_value = condition
        self.provider.select_x_range(self.x, self.x + self.index_range)
        mock_build_range_condition.assert_called_once_with(
            key=DpmmCountKey.X, _min=self.x, _max=self.x + self.index_range
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._build_range_condition")
    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._select_where")
    def test_select_y_range(
        self, mock_select_where: MagicMock, mock_build_range_condition: MagicMock
    ):
        condition = [
            f"{DpmmCountKey.Y}{Operation.GREATER_EQ}{self.y}",
            f"{DpmmCountKey.Y}{Operation.SMALLER_EQ}{self.y + self.index_range}",
        ]
        mock_build_range_condition.return_value = condition
        self.provider.select_y_range(self.y, self.y + self.index_range)
        mock_build_range_condition.assert_called_once_with(
            key=DpmmCountKey.Y, _min=self.y, _max=self.y + self.index_range
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._build_range_condition")
    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._select_where")
    def test_select_count_range(
        self, mock_select_where: MagicMock, mock_build_range_condition: MagicMock
    ):
        condition = [
            f"{DpmmCountKey.COUNT}{Operation.GREATER_EQ}{self.count}",
            f"{DpmmCountKey.COUNT}{Operation.SMALLER_EQ}{self.count + self.index_range}",
        ]
        mock_build_range_condition.return_value = condition
        self.provider.select_count_range(self.count, self.count + self.index_range)
        mock_build_range_condition.assert_called_once_with(
            key=DpmmCountKey.COUNT, _min=self.count, _max=self.count + self.index_range
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._build_range_condition")
    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._select_where")
    def test_select_err_range(
        self, mock_select_where: MagicMock, mock_build_range_condition: MagicMock
    ):
        condition = [
            f"{DpmmCountKey.ERR}{Operation.GREATER_EQ}{self.err}",
            f"{DpmmCountKey.ERR}{Operation.SMALLER_EQ}{self.err + self.index_range}",
        ]
        mock_build_range_condition.return_value = condition
        self.provider.select_err_range(self.err, self.err + self.index_range)
        mock_build_range_condition.assert_called_once_with(
            key=DpmmCountKey.ERR, _min=self.err, _max=self.err + self.index_range
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._build_range_condition")
    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._select_where")
    def test_select_owner_dist_range(
        self, mock_select_where: MagicMock, mock_build_range_condition: MagicMock
    ):
        condition = [
            f"{DpmmCountKey.OWNER_DIST}{Operation.GREATER_EQ}{self.owner}",
            f"{DpmmCountKey.OWNER_DIST}{Operation.SMALLER_EQ}{self.owner + self.index_range}",
        ]
        mock_build_range_condition.return_value = condition
        self.provider.select_owner_dist_range(self.owner, self.owner + self.index_range)
        mock_build_range_condition.assert_called_once_with(
            key=DpmmCountKey.OWNER_DIST,
            _min=self.owner,
            _max=self.owner + self.index_range,
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._build_range_condition")
    @patch("crownetutils.analysis.hdf.provider.IHdfProvider._select_where")
    def test_select_sqerr_range(
        self, mock_select_where: MagicMock, mock_build_range_condition: MagicMock
    ):
        condition = [
            f"{DpmmCountKey.SQERR}{Operation.GREATER_EQ}{self.sqerr}",
            f"{DpmmCountKey.SQERR}{Operation.SMALLER_EQ}{self.sqerr + self.index_range}",
        ]
        mock_build_range_condition.return_value = condition
        self.provider.select_sqerr_range(self.sqerr, self.sqerr + self.index_range)
        mock_build_range_condition.assert_called_once_with(
            key=DpmmCountKey.SQERR, _min=self.sqerr, _max=self.sqerr + self.index_range
        )
        mock_select_where.assert_called_once_with(condition=condition)


if __name__ == "__main__":
    unittest.main()
