import unittest

import numpy as np
import pandas as pd
from oppanalyzer.dcd import (
    DcdMap2D,
    DcdMetaData,
    build_global_density_map,
    build_local_density_map,
)
from uitls import PathHelper
from vadereanalyzer.plots.scenario import VaderScenarioPlotHelper


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


class DcdMapPlotTests(DcdMapTests):
    def test_map(self):
        map = self.regular_grid(
            cell_count=3,
            data={
                "count": [1, 1, 2, 2, 3, 3, 4, 4, 5],
                "measured_t": [],
                "received_t": [],
                "delay": [],
                "m_aoi": [],
                "r_aoi": [],
            },
        )
        ax = map.plot_map(1.0, 1)


class DcdMapTutorialTests(DcdMapTests):
    def test_dcd(self):
        scenario_path = PathHelper.rover_sim(
            "mucFreiNetdLTE2dMulticast/",
            "vadere00_geo_20201012_2/vadere.d/mf_2peds.scenario",
        ).abs_path()

        node_paths = [
            "0a:aa:00:00:00:02",
            "0a:aa:00:00:00:03",
            "0a:aa:00:00:00:04",
            "0a:aa:00:00:00:05",
            "0a:aa:00:00:00:06",
            "0a:aa:00:00:00:07",
            "0a:aa:00:00:00:08",
        ]

        global_path = PathHelper.rover_sim(
            "mucFreiNetdLTE2dMulticast/", "vadere00_geo_20201012_2/global.csv",
        ).abs_path()
        s_plotter = VaderScenarioPlotHelper(scenario_path)

        node_data = []
        for node in node_paths:
            path = PathHelper.rover_sim(
                "mucFreiNetdLTE2dMulticast/", f"vadere00_geo_20201012_2/{node}.csv",
            ).abs_path()
            node_data.append(
                build_local_density_map(path, real_coords=True, full_map=False)
            )

        global_data = build_global_density_map(
            global_path, real_coords=True, with_id_list=True, full_map=False
        )

        dcd = DcdMap2D.from_separated_frames(global_data, node_data)
        dcd.set_scenario_plotter(s_plotter)
        dcd.plot_map(4.0, 3).figure.show()


if __name__ == "__main__":
    unittest.main()
