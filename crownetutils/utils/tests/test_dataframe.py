import unittest

import numpy as np
import pandas as pd

from crownetutils.utils.dataframe import flatten_record_column
from crownetutils.utils.plot import percentiles_dict


class DataframeTest(unittest.TestCase):
    def test_flatten_record(self):
        x = np.arange(100)
        y = np.ones(100)
        idx = y
        idx[0:30] += 1
        idx[30:60] += 2
        idx[60:] += 3
        df = pd.DataFrame(
            np.concatenate([x, y]).reshape((2, -1)).T,
            columns=["x", "y"],
            index=pd.Index(idx, name="idx"),
        )

        ret = (
            df.groupby("idx")
            .agg(["mean", "count", percentiles_dict(25, 50, 75)])
            .stack(0)
        )

        out1 = flatten_record_column(
            ret, cols="percentile_records", replace_columns=True
        )
        self.assertListEqual(
            ["count", "mean", "p25", "p50", "p75"],
            list(out1.columns),
            "record column should be removed",
        )
        out2 = flatten_record_column(
            ret, cols="percentile_records", replace_columns=False
        )
        self.assertListEqual(
            ["count", "mean", "percentile_records", "p25", "p50", "p75"],
            list(out2.columns),
            "record column should be present",
        )

        ret["col2"] = ret["percentile_records"]
        out3 = flatten_record_column(
            ret, cols="percentile_records", replace_columns=True
        )
        self.assertListEqual(
            ["count", "mean", "col2", "p25", "p50", "p75"],
            list(out3.columns),
            "only one record column should be flattened and col2 must be still there",
        )
        out4 = flatten_record_column(
            ret, cols=["percentile_records", "col2"], replace_columns=True
        )
        self.assertListEqual(
            ["count", "mean", "p25", "p50", "p75", "p25", "p50", "p75"],
            list(out4.columns),
            "both record columns should be flattened",
        )

        out5 = flatten_record_column(ret["col2"], replace_columns=True)
        self.assertListEqual(
            ["p25", "p50", "p75"],
            list(out5.columns),
            "only col2 should be flattened, replace_columns has no effect.",
        )
        out6 = flatten_record_column(ret["col2"], cols="xxx", replace_columns=False)
        self.assertListEqual(
            ["p25", "p50", "p75"],
            list(out6.columns),
            "only col2 should be flattened, replace_columns has no effect.",
        )

        out7 = flatten_record_column(ret["col2"].values)
        self.assertListEqual(
            ["p25", "p50", "p75"],
            list(out7.columns),
            "should work with array of records.",
        )
