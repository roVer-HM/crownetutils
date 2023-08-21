import unittest

import numpy as np
import pandas as pd

from crownetutils.analysis.dpmm.dpmm import DpmMap


class DpmmTest(unittest.TestCase):
    def test_crate_full_time_index(self):
        times = np.array([1, 2, 5])
        xy = pd.MultiIndex.from_arrays(
            np.array([[10, 20, 30], [100, 200, 300]]), names=["x", "y"]
        )
        out = DpmMap.create_full_time_index(times=times, xy_index=xy)

        val = np.array(
            [
                [1, 10, 100, 0],
                [2, 10, 100, 0],
                [5, 10, 100, 0],
                [1, 20, 200, 0],
                [2, 20, 200, 0],
                [5, 20, 200, 0],
                [1, 30, 300, 0],
                [2, 30, 300, 0],
                [5, 30, 300, 0],
            ]
        )
        val_idx = pd.MultiIndex.from_arrays(val.T, names=["simtime", "x", "y", "ID"])
        self.assertEqual(
            out.difference(val_idx).shape,
            (0,),
            "expected full overlap with no difference in index. ",
        )
