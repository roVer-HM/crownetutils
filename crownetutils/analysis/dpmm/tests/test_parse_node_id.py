import re
import unittest

from crownetutils.analysis.dpmm.builder import parse_node_id
from crownetutils.analysis.dpmm.dpmm_cfg import DpmmCfg


class PareNodeTest(unittest.TestCase):
    def test_parse_node_id(self):
        default_cfg = DpmmCfg.default_density_beacon_map_cfg("/tmp/base_dir")
        node_id = 42
        correct_string = f"any/path/dcdMap_{node_id}.csv"
        p = re.compile(default_cfg.node_map_csv_id_regex)
        result_correct = parse_node_id(path=correct_string, regex=p)
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
                parse_node_id(wrong_path, p)
