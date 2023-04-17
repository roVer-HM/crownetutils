import os
import unittest
from unittest import mock
from unittest.mock import MagicMock, call, patch

import pandas as pd
from fs.tempfs import TempFS

from crownetutils.analysis.dpmm.hdf.dpmm_provider import DpmmKey, DpmmProvider
from crownetutils.analysis.dpmm.metadata import DpmmMetaData
from crownetutils.analysis.dpmm.tests.utils import (
    create_dcd_csv_dataframe,
    create_tmp_fs,
    make_dirs,
)
from crownetutils.analysis.hdf.groups import HdfGroups
from crownetutils.analysis.hdf.provider import ProviderVersion


class DpmmProviderTest(unittest.TestCase):
    # create tmp fs. (use fs.root_path to access as normal path)
    fs: TempFS = create_tmp_fs("DcdMapCountProviderTest")
    test_out_dir: str = os.path.join(fs.root_path, "unittest")
    sample_file_dir: str = os.path.join(test_out_dir, "sample.hdf5")
    provider: DpmmProvider = DpmmProvider(sample_file_dir, version=ProviderVersion.V0_1)

    @classmethod
    def setUpClass(cls):
        make_dirs(cls.test_out_dir)

    @classmethod
    def tearDownClass(cls):
        # close temporary filesystem
        cls.fs.close()

    def test_DcdMapProperties_v1(self):
        sample_grp_key = HdfGroups.DCD_MAP
        sample_default_index = DpmmKey.SIMTIME
        sample_index_order = {
            0: DpmmKey.SIMTIME,
            1: DpmmKey.X,
            2: DpmmKey.Y,
            3: DpmmKey.SOURCE,
            4: DpmmKey.NODE,
        }
        sample_columns = [
            DpmmKey.COUNT,
            DpmmKey.MEASURE_TIME,
            DpmmKey.RECEIVED_TIME,
            DpmmKey.SELECTION,
            DpmmKey.OWN_CELL,
            DpmmKey.X_OWNER,
            DpmmKey.Y_OWNER,
            DpmmKey.OWNER_DIST,
            DpmmKey.DELAY,
            DpmmKey.MEASURE_AGE,
            DpmmKey.UPDATE_AGE,
        ]
        p: DpmmProvider = DpmmProvider(
            self.sample_file_dir, version=ProviderVersion.V0_1
        )
        result_grp_key = p.group_key()
        result_index_order = p.index_order()
        result_default_index = p.default_index_key()
        result_columns = p.columns()

        self.assertEqual(result_grp_key, sample_grp_key)
        self.assertEqual(result_index_order, sample_index_order)
        self.assertEqual(result_default_index, sample_default_index)
        self.assertEqual(result_columns, sample_columns)

    def test_DcdMapProperties_v2(self):
        sample_grp_key = HdfGroups.DCD_MAP
        sample_default_index = DpmmKey.SIMTIME
        sample_index_order = {
            0: DpmmKey.SIMTIME,
            1: DpmmKey.X,
            2: DpmmKey.Y,
            3: DpmmKey.SOURCE,
            4: DpmmKey.NODE,
        }
        sample_columns = [
            DpmmKey.COUNT,
            DpmmKey.MEASURE_TIME,
            DpmmKey.RECEIVED_TIME,
            DpmmKey.SELECTION,
            DpmmKey.OWN_CELL,
            DpmmKey.SOURCE_HOST,
            DpmmKey.SOURCE_ENTRY,
            DpmmKey.HOST_ENTRY,
            DpmmKey.SELECTION_RANK,
            DpmmKey.X_OWNER,
            DpmmKey.Y_OWNER,
            DpmmKey.OWNER_DIST,
            DpmmKey.DELAY,
            DpmmKey.MEASURE_AGE,
            DpmmKey.UPDATE_AGE,
        ]
        p: DpmmProvider = DpmmProvider(
            self.sample_file_dir, version=ProviderVersion.V0_2
        )
        result_grp_key = p.group_key()
        result_index_order = p.index_order()
        result_default_index = p.default_index_key()
        result_columns = p.columns()

        self.assertEqual(result_grp_key, sample_grp_key)
        self.assertEqual(result_index_order, sample_index_order)
        self.assertEqual(result_default_index, sample_default_index)
        self.assertEqual(result_columns, sample_columns)

    @patch(
        "crownetutils.analysis.dpmm.hdf.dpmm_provider.DpmmProvider.build_dcd_dataframe"
    )
    @patch(
        "crownetutils.analysis.dpmm.hdf.dpmm_provider.DpmmProvider.set_selection_mapping_attribute"
    )
    @patch(
        "crownetutils.analysis.dpmm.hdf.dpmm_provider.DpmmProvider.set_used_selection_attribute"
    )
    @patch("pandas.HDFStore")
    def test_create_from_csv(
        self,
        mock_store: MagicMock,
        mock_set_attribute: MagicMock,
        mock_set_attribute2: MagicMock,
        mock_build_dataframe: MagicMock,
    ):
        dcd_path_1 = "any/path/dcdMap_42.csv"
        dcd_path_2 = "any/path/dcdMap_43.csv"
        ret_df_1 = pd.DataFrame(data=["a"])
        ret_df_2 = pd.DataFrame(data=["b"])
        mock_build_dataframe.side_effect = [ret_df_1, ret_df_2]

        self.provider.create_from_csv([dcd_path_1, dcd_path_2])

        mock_build_dataframe.assert_has_calls([call(dcd_path_1), call(dcd_path_2)])
        mock_store.return_value.append.assert_has_calls(
            [
                call(
                    key=self.provider.group_key(),
                    index=True,
                    value=ret_df_1,
                    format="table",
                    data_columns=True,
                ),
                call(
                    key=self.provider.group_key(),
                    index=True,
                    value=ret_df_2,
                    format="table",
                    data_columns=True,
                ),
            ]
        )
        mock_set_attribute.assert_called_once()
        mock_set_attribute2.assert_called_once()

    def test_parse_node_id(self):
        node_id = 42
        correct_string = f"any/path/dcdMap_{node_id}.csv"
        result_correct = self.provider.parse_node_id(correct_string)
        self.assertEqual(result_correct, node_id)

        wrong_paths = [
            "any/path/dcdMap361.csv",
            "any/path/dcdMap_full.csv",
            "any/path/dcdMap.csv",
            "any/path/dcdMap_361",
            "any/path/dcdMap_.csv",
            "any/path/dcdMap.csv",
        ]
        for wrong_path in wrong_paths:
            with self.assertRaises(ValueError):
                self.provider.parse_node_id(wrong_path)

    @patch("crownetutils.analysis.dpmm.hdf.dpmm_provider.DpmmProvider.parse_node_id")
    @patch("crownetutils.utils.dataframe.LazyDataFrame.read_meta_data")
    @patch("crownetutils.analysis.dpmm.hdf.dpmm_provider.read_csv")
    def test_build_dcd_dataframe(
        self,
        mock_read_csv: MagicMock,
        mock_meta: MagicMock,
        mock_parse_node_id: MagicMock,
    ):
        own_node_id = 42
        csv_path = f"/any/path/dcdMap_{own_node_id}.csv"
        mock_parse_node_id.return_value = own_node_id
        mock_read_csv.return_value = create_dcd_csv_dataframe(
            number_entries=50, node_id=45
        )
        mock_meta.return_value = dict(
            CELLSIZE=5.0, XSIZE=5, YSIZE=5, NODE_ID=0, version=ProviderVersion.V0_1
        )
        result = self.provider.build_dcd_dataframe(csv_path)
        self.assertEqual(
            [
                DpmmKey.SIMTIME,
                DpmmKey.X,
                DpmmKey.Y,
                DpmmKey.SOURCE,
                DpmmKey.NODE,
            ],
            result.index.names,
        )
        self.assertTrue(
            result["selection"].isin(self.provider.selection_mapping.values()).all()
        )
        self.assertTrue(result.reset_index()[DpmmKey.NODE].isin([own_node_id]).all())

    def test_get_dcd_file_paths(self):
        with mock.patch("os.walk") as mockwalk:
            base_path = "/any/path"
            dcd_map_1 = "dcdMap_42.csv"
            dcd_map_2 = "dcdMap_43.csv"
            mockwalk.return_value = [
                (
                    base_path,
                    (),
                    (
                        "dcdMap_full.csv",
                        "dcdMap123.csv",
                        "345.csv",
                        "dcdMap_123",
                        dcd_map_1,
                        dcd_map_2,
                    ),
                ),
            ]
            result = self.provider.get_dcd_file_paths(base_path)
            self.assertEqual(
                [os.path.join(base_path, p) for p in [dcd_map_1, dcd_map_2]], result
            )
            self.assertEqual(len(result), 2)

    @patch("crownetutils.analysis.hdf.provider.IHdfProvider.get_attribute")
    def test_get_selection_mapping_attribute(self, mock_get_attribute: MagicMock):
        self.provider.get_selection_mapping_attribute()
        mock_get_attribute.assert_called_once_with("selection_mapping")

    @patch("crownetutils.analysis.hdf.provider.IHdfProvider.set_attribute")
    def test_set_selection_mapping_attribute(self, mock_set_attribute: MagicMock):
        self.provider.set_selection_mapping_attribute()
        mock_set_attribute.assert_called_once_with(
            "selection_mapping", self.provider.selection_mapping
        )


if __name__ == "__main__":
    unittest.main()
