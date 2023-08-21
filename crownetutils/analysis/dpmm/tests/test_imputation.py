import re
import unittest

import numpy as np
import pandas as pd

from crownetutils.analysis.dpmm.builder import parse_node_id
from crownetutils.analysis.dpmm.dpmm_cfg import DpmmCfg
from crownetutils.analysis.dpmm.imputation import ArbitraryValueImputationWithRsd


class ImputationTest(unittest.TestCase):
    def test_arbitaryValueWithRsd(self):
        enb_dist = pd.DataFrame.from_records(
            [
                [0.0, 100, "eNB[0]", 0.0, 0.0, 0],
                [0.0, 101, "eNB[0]", 50.0, 50.0, 1],
                [0.0, 102, "eNB[0]", 100.0, 100.0, 2],
            ],
            columns=["time", "hostId", "host", "x", "y", "rsd_id"],
        )
        data = pd.DataFrame.from_records(
            [
                [1, np.nan, np.nan, np.nan, np.nan, True, np.nan],
                [1, np.nan, np.nan, np.nan, np.nan, True, np.nan],
                [1, np.nan, np.nan, np.nan, np.nan, True, np.nan],
                [1, np.nan, np.nan, np.nan, np.nan, True, np.nan],
                [2, np.nan, np.nan, np.nan, np.nan, True, np.nan],
                [1, 5, np.nan, np.nan, 8, False, 42],
                [1, 6, np.nan, np.nan, 9, False, 42],
                [1, 7, np.nan, np.nan, 10, False, 42],
            ],
            index=pd.MultiIndex.from_tuples(
                [
                    (1.0, 98.0, 98.0),
                    (1.0, 51.0, 55.0),
                    (1.0, 48.0, 48.0),
                    (1.0, 5.0, 5.0),
                    (2.0, 98.0, 98.0),
                    (1.0, 445.0, 444.0),
                    (1.0, 446.0, 444.0),
                    (1.0, 447.0, 444.0),
                ],
                names=["simtime", "x", "y"],
            ),
            columns=[
                "glb_count",
                "count",
                "x_owner",
                "y_owner",
                "rsd_id",
                "missing_value",
                "owner_rsd_id",
            ],
        )

        inp = ArbitraryValueImputationWithRsd(
            rsd_origin_position=enb_dist, rsd_col="rsd_id", fill_value=0
        )
        out = inp(data, data_column="count")

        # keep line count equal
        self.assertEqual(out.shape[0], 8)

        # only update missing values in the rsd.
        # ensure same cells get same rsd index 0 and 4
        self.assertListEqual([2, 1, 1, 0, 2, 8, 9, 10], out["rsd_id"].to_list())
        self.assertListEqual(
            [42.0, 42.0, 42.0, 42.0, 999.0, 42.0, 42.0, 42.0],
            out["owner_rsd_id"].fillna(999.0).to_list(),
        )
        # only update missing values and leave the rest alone
        self.assertListEqual([0, 0, 0, 0, 0, 5, 6, 7], out["count"].to_list())


if __name__ == "__main__":
    unittest.main()
