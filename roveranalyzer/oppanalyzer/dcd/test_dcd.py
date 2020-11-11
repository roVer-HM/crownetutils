import os
import unittest

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.testing as pdt
from oppanalyzer.dcd.dcd_map import DcdMap2DMulti

from roveranalyzer.oppanalyzer.dcd import DcdMap2D, DcdMetaData
from roveranalyzer.tests.utils import TestDataHandler
from roveranalyzer.uitls import PathHelper
from roveranalyzer.uitls.misc import intersect


class DcdMapTests:
    @staticmethod
    def create_index(ids, times, x, y):
        _idx = [ids, times, x, y]
        return pd.MultiIndex.from_product(_idx, names=("ID", "simtime", "x", "y"))

    @staticmethod
    def regular_grid(cell_count, data, id=1):
        meta = DcdMetaData(1, [cell_count, cell_count], [cell_count, cell_count], id)
        idx = DcdMapTests.create_index(
            [1], [1.0], np.arange(cell_count), np.arange(cell_count)
        )
        lines = cell_count * cell_count
        for k, v in data.items():
            if len(v) != lines:
                data[k] = np.zeros(lines)
        df = pd.DataFrame(data, index=idx)
        return DcdMap2D(meta, {1: 1}, df)


class DcdMapSimpleTest(DcdMapTests):
    def load(self):
        global_map_path = self.handler.data_dir.glob(
            "global.csv", recursive=False, expect=1
        )
        node_map_paths = self.handler.data_dir.glob("0a:*.csv")
        self.dcd = DcdMap2DMulti.from_paths(
            global_data=global_map_path,
            node_data=node_map_paths,
            real_coords=True,
            scenario_plotter=self.scenario_path,
        )


class DcdMapTutorialTests(DcdMapSimpleTest, unittest.TestCase):
    def tearDown(self):
        self.handler.remove_data()

    def setUp(self) -> None:
        # load test data from url and save to /tmp
        test_data_001 = TestDataHandler.tar(
            url="https://sam.cs.hm.edu/samcloud/index.php/s/7RAg26eB3JmTKsX/download",
            file_name="tutorialTest",
            archive_base_dir="2020-10-21_densityMap_test001",
            keep_archive=True,  # keep archive to reduce unnecessary downloads
        )
        self.handler: TestDataHandler = test_data_001
        self.handler.download_test_data(override=True)
        self.scenario_path = self.handler.data_dir.glob(
            "vadere.d/*.scenario", recursive=False, expect=1
        )
        self.load()

    def test_dcd_count_sum_all(self):
        count_per_id_is = self.dcd.raw2d.groupby(level=["ID"]).sum()["count"]
        count_per_id_should = pd.Series(
            data=[105, 388, 460, 439, 28, 442, 440, 445], name="count"
        )
        count_per_id_should.index.rename("ID", inplace=True)
        pdt.assert_series_equal(count_per_id_is, count_per_id_should)


@unittest.skipIf(
    "ROVER_LOCAL" not in os.environ,
    "Local test with local test data. Do not run on CI",
)
class DcdMapTestLocal(DcdMapSimpleTest, unittest.TestCase):
    def setUp(self) -> None:
        # load test data from url and save to /tmp
        simulation = "mucFreiNetdLTE2dMulticast"
        # run_name = "0_vadere00_geo_20201026-ymf_map"
        run_name = "0_vadere00_geo_20201103-10:32:09_all"
        test_data_local_001 = TestDataHandler.local(
            path=PathHelper.rover_sim(simulation, run_name).abs_path()
        )
        self.handler = test_data_local_001
        self.handler.download_test_data(override=True)
        self.scenario_path = self.handler.data_dir.glob(
            "vadere.d/*.scenario", recursive=False, expect=1
        )
        self.load()


    def test_intersect(self):
        a = np.array([1.0, 1.0])
        b = np.array([3.0, 2.0])
        c = np.array([2.0, 0.0])
        d = np.array([2.0, 4.0])

        ab = np.append(d, b).reshape(-1, 2)
        cd = np.append(c, a).reshape(-1, 2)
        print(intersect(ab, cd))


def plot_wrap(func, _self, *args, **kwargs):
    if "ax" not in kwargs:
        print("create axes...")
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        fig.canvas.toolbar_position = "bottom"
        kwargs.setdefault("ax", ax)
    ret = func(_self, *args, **kwargs)
    return ret


if __name__ == "__main__":
    unittest.main()
