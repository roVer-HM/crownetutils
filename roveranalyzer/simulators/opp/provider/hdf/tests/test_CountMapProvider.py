import os
import unittest
from unittest.mock import MagicMock, call, patch

import pandas as pd
from fs.tempfs import TempFS

from roveranalyzer.simulators.opp.provider.hdf.DcdMapCountProvider import (
    CountMapKey,
    DcdMapCount,
)
from roveranalyzer.simulators.opp.provider.hdf.HdfGroups import HdfGroups
from roveranalyzer.simulators.opp.provider.hdf.Operation import Operation
from roveranalyzer.simulators.opp.provider.hdf.tests.utils import (
    create_count_map_dataframe,
    create_tmp_fs,
    make_dirs,
    safe_dataframe_to_hdf,
)


class DcdMapCountProviderTest(unittest.TestCase):
    # create tmp fs. (use fs.root_path to access as normal path)
    fs: TempFS = create_tmp_fs("DcdMapCountProviderTest")
    test_out_dir: str = os.path.join(fs.root_path, "unittest")
    sample_file_dir: str = os.path.join(test_out_dir, "sample.hdf5")
    provider: DcdMapCount = DcdMapCount(sample_file_dir)
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
        sample_default_index = CountMapKey.SIMTIME
        sample_index_order = {
            0: CountMapKey.SIMTIME,
            1: CountMapKey.X,
            2: CountMapKey.Y,
            3: CountMapKey.ID,
        }
        sample_columns = [
            CountMapKey.COUNT,
            CountMapKey.ERR,
            CountMapKey.OWNER_DIST,
            CountMapKey.SQERR,
        ]
        result_grp_key = self.provider.group_key()
        result_index_order = self.provider.index_order()
        result_default_index = self.provider.default_index_key()
        result_columns = self.provider.columns()

        self.assertEqual(result_grp_key, sample_grp_key)
        self.assertEqual(result_index_order, sample_index_order)
        self.assertEqual(result_default_index, sample_default_index)
        self.assertEqual(result_columns, sample_columns)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._build_exact_condition"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._select_where"
    )
    def test_select_id_exact(
        self, mock_select_where: MagicMock, mock_build_exact_condition: MagicMock
    ):
        condition = [f"{CountMapKey.ID}{Operation.EQ}{self._id}"]
        mock_build_exact_condition.return_value = condition
        self.provider.select_id_exact(self._id)
        mock_build_exact_condition.assert_called_once_with(
            key=CountMapKey.ID, value=self._id, operation=Operation.EQ
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._build_exact_condition"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._select_where"
    )
    def test_select_simtime_exact(
        self, mock_select_where: MagicMock, mock_build_exact_condition: MagicMock
    ):
        condition = [f"{CountMapKey.SIMTIME}{Operation.EQ}{self.simtime}"]
        mock_build_exact_condition.return_value = condition
        self.provider.select_simtime_exact(self.simtime)
        mock_build_exact_condition.assert_called_once_with(
            key=CountMapKey.SIMTIME, value=self.simtime, operation=Operation.EQ
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._build_exact_condition"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._select_where"
    )
    def test_select_x_exact(
        self, mock_select_where: MagicMock, mock_build_exact_condition: MagicMock
    ):
        condition = [f"{CountMapKey.X}{Operation.EQ}{self.x}"]
        mock_build_exact_condition.return_value = condition
        self.provider.select_x_exact(self.x)
        mock_build_exact_condition.assert_called_once_with(
            key=CountMapKey.X, value=self.x, operation=Operation.EQ
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._build_exact_condition"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._select_where"
    )
    def test_select_y_exact(
        self, mock_select_where: MagicMock, mock_build_exact_condition: MagicMock
    ):
        condition = [f"{CountMapKey.Y}{Operation.EQ}{self.y}"]
        mock_build_exact_condition.return_value = condition
        self.provider.select_y_exact(self.y)
        mock_build_exact_condition.assert_called_once_with(
            key=CountMapKey.Y, value=self.y, operation=Operation.EQ
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._build_exact_condition"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._select_where"
    )
    def test_select_count_exact(
        self, mock_select_where: MagicMock, mock_build_exact_condition: MagicMock
    ):
        condition = [f"{CountMapKey.COUNT}{Operation.EQ}{self.count}"]
        mock_build_exact_condition.return_value = condition
        self.provider.select_count_exact(self.count)
        mock_build_exact_condition.assert_called_once_with(
            key=CountMapKey.COUNT, value=self.count, operation=Operation.EQ
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._build_exact_condition"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._select_where"
    )
    def test_select_err_exact(
        self, mock_select_where: MagicMock, mock_build_exact_condition: MagicMock
    ):
        condition = [f"{CountMapKey.ERR}{Operation.EQ}{self.err}"]
        mock_build_exact_condition.return_value = condition
        self.provider.select_err_exact(self.err)
        mock_build_exact_condition.assert_called_once_with(
            key=CountMapKey.ERR, value=self.err, operation=Operation.EQ
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._build_exact_condition"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._select_where"
    )
    def test_owner_dist_exact(
        self, mock_select_where: MagicMock, mock_build_exact_condition: MagicMock
    ):
        condition = [f"{CountMapKey.OWNER_DIST}{Operation.EQ}{self.owner}"]
        mock_build_exact_condition.return_value = condition
        self.provider.select_owner_dist_exact(self.owner)
        mock_build_exact_condition.assert_called_once_with(
            key=CountMapKey.OWNER_DIST, value=self.owner, operation=Operation.EQ
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._build_exact_condition"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._select_where"
    )
    def test_select_sqerr_exact(
        self, mock_select_where: MagicMock, mock_build_exact_condition: MagicMock
    ):
        condition = [f"{CountMapKey.SQERR}{Operation.EQ}{self.sqerr}"]
        mock_build_exact_condition.return_value = condition
        self.provider.select_sqerr_exact(self.sqerr)
        mock_build_exact_condition.assert_called_once_with(
            key=CountMapKey.SQERR, value=self.sqerr, operation=Operation.EQ
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._build_exact_condition"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._select_where"
    )
    def test_select_simtime_and_node_id_exact(
        self, mock_select_where: MagicMock, mock_build_exact_condition: MagicMock
    ):
        simtime = 42
        ID = 43
        mock_build_exact_condition.side_effect = [
            [f"{CountMapKey.SIMTIME}={simtime}"],
            [f"{CountMapKey.ID}={ID}"],
        ]
        self.provider.select_simtime_and_node_id_exact(simtime, ID)
        calls = [
            call(key=f"{CountMapKey.SIMTIME}", value=simtime, operation=Operation.EQ),
            call(key=f"{CountMapKey.ID}", value=ID, operation=Operation.EQ),
        ]
        mock_build_exact_condition.mock_has_calls(calls)
        condition = [f"{CountMapKey.SIMTIME}={simtime}", f"{CountMapKey.ID}={ID}"]
        mock_select_where.assert_called_once_with(condition=condition)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._build_range_condition"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._select_where"
    )
    def test_select_id_range(
        self, mock_select_where: MagicMock, mock_build_range_condition: MagicMock
    ):
        condition = [
            f"{CountMapKey.ID}{Operation.GREATER_EQ}{self._id}",
            f"{CountMapKey.ID}{Operation.SMALLER_EQ}{self._id + self.index_range}",
        ]
        mock_build_range_condition.return_value = condition
        self.provider.select_id_range(self._id, self._id + self.index_range)
        mock_build_range_condition.assert_called_once_with(
            key=CountMapKey.ID, _min=self._id, _max=self._id + self.index_range
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._build_range_condition"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._select_where"
    )
    def test_select_simtime_range(
        self, mock_select_where: MagicMock, mock_build_range_condition: MagicMock
    ):
        condition = [
            f"{CountMapKey.SIMTIME}{Operation.GREATER_EQ}{self.simtime}",
            f"{CountMapKey.SIMTIME}{Operation.SMALLER_EQ}{self.simtime + self.index_range}",
        ]
        mock_build_range_condition.return_value = condition
        self.provider.select_simtime_range(
            self.simtime, self.simtime + self.index_range
        )
        mock_build_range_condition.assert_called_once_with(
            key=CountMapKey.SIMTIME,
            _min=self.simtime,
            _max=self.simtime + self.index_range,
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._build_range_condition"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._select_where"
    )
    def test_select_x_range(
        self, mock_select_where: MagicMock, mock_build_range_condition: MagicMock
    ):
        condition = [
            f"{CountMapKey.X}{Operation.GREATER_EQ}{self.x}",
            f"{CountMapKey.X}{Operation.SMALLER_EQ}{self.x + self.index_range}",
        ]
        mock_build_range_condition.return_value = condition
        self.provider.select_x_range(self.x, self.x + self.index_range)
        mock_build_range_condition.assert_called_once_with(
            key=CountMapKey.X, _min=self.x, _max=self.x + self.index_range
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._build_range_condition"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._select_where"
    )
    def test_select_y_range(
        self, mock_select_where: MagicMock, mock_build_range_condition: MagicMock
    ):
        condition = [
            f"{CountMapKey.Y}{Operation.GREATER_EQ}{self.y}",
            f"{CountMapKey.Y}{Operation.SMALLER_EQ}{self.y + self.index_range}",
        ]
        mock_build_range_condition.return_value = condition
        self.provider.select_y_range(self.y, self.y + self.index_range)
        mock_build_range_condition.assert_called_once_with(
            key=CountMapKey.Y, _min=self.y, _max=self.y + self.index_range
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._build_range_condition"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._select_where"
    )
    def test_select_count_range(
        self, mock_select_where: MagicMock, mock_build_range_condition: MagicMock
    ):
        condition = [
            f"{CountMapKey.COUNT}{Operation.GREATER_EQ}{self.count}",
            f"{CountMapKey.COUNT}{Operation.SMALLER_EQ}{self.count + self.index_range}",
        ]
        mock_build_range_condition.return_value = condition
        self.provider.select_count_range(self.count, self.count + self.index_range)
        mock_build_range_condition.assert_called_once_with(
            key=CountMapKey.COUNT, _min=self.count, _max=self.count + self.index_range
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._build_range_condition"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._select_where"
    )
    def test_select_err_range(
        self, mock_select_where: MagicMock, mock_build_range_condition: MagicMock
    ):
        condition = [
            f"{CountMapKey.ERR}{Operation.GREATER_EQ}{self.err}",
            f"{CountMapKey.ERR}{Operation.SMALLER_EQ}{self.err + self.index_range}",
        ]
        mock_build_range_condition.return_value = condition
        self.provider.select_err_range(self.err, self.err + self.index_range)
        mock_build_range_condition.assert_called_once_with(
            key=CountMapKey.ERR, _min=self.err, _max=self.err + self.index_range
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._build_range_condition"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._select_where"
    )
    def test_select_owner_dist_range(
        self, mock_select_where: MagicMock, mock_build_range_condition: MagicMock
    ):
        condition = [
            f"{CountMapKey.OWNER_DIST}{Operation.GREATER_EQ}{self.owner}",
            f"{CountMapKey.OWNER_DIST}{Operation.SMALLER_EQ}{self.owner + self.index_range}",
        ]
        mock_build_range_condition.return_value = condition
        self.provider.select_owner_dist_range(self.owner, self.owner + self.index_range)
        mock_build_range_condition.assert_called_once_with(
            key=CountMapKey.OWNER_DIST,
            _min=self.owner,
            _max=self.owner + self.index_range,
        )
        mock_select_where.assert_called_once_with(condition=condition)

    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._build_range_condition"
    )
    @patch(
        "roveranalyzer.simulators.opp.provider.hdf.IHdfProvider.IHdfProvider._select_where"
    )
    def test_select_sqerr_range(
        self, mock_select_where: MagicMock, mock_build_range_condition: MagicMock
    ):
        condition = [
            f"{CountMapKey.SQERR}{Operation.GREATER_EQ}{self.sqerr}",
            f"{CountMapKey.SQERR}{Operation.SMALLER_EQ}{self.sqerr + self.index_range}",
        ]
        mock_build_range_condition.return_value = condition
        self.provider.select_sqerr_range(self.sqerr, self.sqerr + self.index_range)
        mock_build_range_condition.assert_called_once_with(
            key=CountMapKey.SQERR, _min=self.sqerr, _max=self.sqerr + self.index_range
        )
        mock_select_where.assert_called_once_with(condition=condition)


if __name__ == "__main__":
    unittest.main()
