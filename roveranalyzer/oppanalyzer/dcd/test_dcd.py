import unittest

import numpy as np
import pandas as pd
import pandas.testing as pdt

from roveranalyzer.oppanalyzer.dcd import (
    DcdMap2D,
    DcdMetaData,
    build_global_density_map,
    build_local_density_map,
)
from roveranalyzer.tests.utils import TestDataHandler
from roveranalyzer.vadereanalyzer.plots.scenario import VaderScenarioPlotHelper


class DcdMapTests(unittest.TestCase):
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


class DcdMapTutorialTests(DcdMapTests):
    def tearDown(self):
        self.handler.remove_data()

    def setUp(self) -> None:
        # load test data from url and save to /tmp
        self.handler = TestDataHandler.tar(
            url="https://sam.cs.hm.edu/samcloud/index.php/s/7RAg26eB3JmTKsX/download",
            file_name="tutorialTest",
            archive_base_dir="2020-10-21_densityMap_test001",
            keep_archive=True,  # keep archive to reduce unnecessary downloads
        )
        self.handler.download_test_data(override=True)

        scenario_path = self.handler.abs_path("vadere.d/mf_2peds.scenario")

        node_paths = [
            "0a:aa:00:00:00:02",
            "0a:aa:00:00:00:03",
            "0a:aa:00:00:00:04",
            "0a:aa:00:00:00:05",
            "0a:aa:00:00:00:06",
            "0a:aa:00:00:00:07",
            "0a:aa:00:00:00:08",
        ]

        global_path = self.handler.abs_path("global.csv")
        s_plotter = VaderScenarioPlotHelper(scenario_path)

        node_data = []
        for node in node_paths:
            path = self.handler.abs_path(f"{node}.csv")
            node_data.append(
                build_local_density_map(path, real_coords=True, full_map=False)
            )

        global_data = build_global_density_map(
            global_path, real_coords=True, with_id_list=True, full_map=False
        )

        self.dcd = DcdMap2D.from_separated_frames(global_data, node_data)
        self.dcd.set_scenario_plotter(s_plotter)

    def test_dcd_count_sum_all(self):
        count_per_id_is = self.dcd.raw2d.groupby(level=["ID"]).sum()["count"]
        count_per_id_should = pd.Series(
            data=[105, 388, 460, 439, 28, 442, 440, 445], name="count"
        )
        count_per_id_should.index.rename("ID", inplace=True)
        pdt.assert_series_equal(count_per_id_is, count_per_id_should)


if __name__ == "__main__":
    unittest.main()
