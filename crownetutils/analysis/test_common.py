import os
import shutil
import tempfile
import unittest
from datetime import datetime
from os import path

import pandas as pd

from crownetutils.analysis.common import Simulation


class SimulationTest(unittest.TestCase):
    def setUp(self):
        # Write control stats log files to temporary directory

        self.test_dir = tempfile.mkdtemp()
        self.container_opp_stats_out_file = os.path.join(
            self.test_dir, "container_opp_stats.out"
        )
        self.container_vadere_stats_out_file = os.path.join(
            self.test_dir, "container_vadere_stats.out"
        )
        self.container_control_stats_out_file = os.path.join(
            self.test_dir, "container_control_stats.out"
        )

        vadere_stats_out = "Timestamp,DockerStatsCPUPerc,DockerStatsRamGByte\n \
                            Thu Jun  1 17:30:49 2023,87.45742310856288,0.08413184\n \
                            Thu Jun  1 17:30:50 2023,0.04513008595988539,0.083554304"

        opp_stats_out = "Timestamp,DockerStatsCPUPerc,DockerStatsRamGByte\n \
                            Thu Jun  1 17:31:31 2023,4.961055033557046,0.052297728\n \
                            Thu Jun  1 17:31:32 2023,0.0,0.051904512\n \
                            Thu Jun  1 17:31:34 2023,0.0,0.051777536\n \
                            Thu Jun  1 17:31:36 2023,0.0,0.051777536\n \
                            Thu Jun  1 17:31:38 2023,0.0,0.051777536\n \
                            Thu Jun  1 17:31:40 2023,0.0,0.051777536\n \
                            Thu Jun  1 17:31:42 2023,0.0,0.051777536\n \
                            Thu Jun  1 17:31:44 2023,0.0,0.051777536\n \
                            Thu Jun  1 17:31:46 2023,0.0,0.051777536\n \
                            Thu Jun  1 17:31:48 2023,0.0,0.051777536\n \
                            Thu Jun  1 17:31:50 2023,0.0,0.051777536\n \
                            Thu Jun  1 17:31:52 2023,0.0,0.051777536"

        control_stats_out = "Timestamp,DockerStatsCPUPerc,DockerStatsRamGByte\n \
                            Thu Jun  1 17:30:55 2023,17.583686541568287,0.049242112\n \
                            Thu Jun  1 17:30:57 2023,0.0,0.049119232\n \
                            Thu Jun  1 17:30:59 2023,0.0,0.048873472\n \
                            Thu Jun  1 17:31:01 2023,0.0,0.048750592\n \
                            Thu Jun  1 17:31:03 2023,0.0,0.048750592\n \
                            Thu Jun  1 17:31:05 2023,0.0,0.048750592\n \
                            Thu Jun  1 17:31:07 2023,0.0,0.048750592\n \
                            Thu Jun  1 17:31:09 2023,0.0,0.048750592\n \
                            Thu Jun  1 17:31:11 2023,0.0,0.04859904\n \
                            Thu Jun  1 17:31:13 2023,0.0,0.04847616\n \
                            Thu Jun  1 17:31:15 2023,0.0,0.04847616"

        with open(self.container_opp_stats_out_file, "w") as f:
            f.write(opp_stats_out)

        with open(self.container_vadere_stats_out_file, "w") as f:
            f.write(vadere_stats_out)

        with open(self.container_control_stats_out_file, "w") as f:
            f.write(control_stats_out)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_get_docker_stats(self):
        # compare values in vadere_stats_out variable in the setUp method

        df_expected = pd.DataFrame(
            data={
                "Timestamp": [
                    datetime(2023, 6, 1, 17, 30, 49, 0),
                    datetime(2023, 6, 1, 17, 30, 50, 0),
                ],
                "DockerStatsCPUPerc": [87.45742310856288, 0.04513008595988539],
                "DockerStatsRamGByte": [0.08413184, 0.083554304],
                "container": ["vadere", "vadere"],
            },
            index=[0, 1],
        )

        sim = Simulation.from_output_dir(self.test_dir)
        df_actual = sim.get_docker_stats(self.container_vadere_stats_out_file)
        pd.testing.assert_frame_equal(df_actual, df_expected)

    def test_get_docker_stats_all(self):
        sim = Simulation.from_output_dir(self.test_dir)
        # only get omnetpp stats
        df1 = sim.get_docker_stats_all(required=["container_opp_stats.out"])
        assert all(df1["container"] == "omnetpp")

        # get omnetpp and flowcontrol stats
        df2 = sim.get_docker_stats_all(
            required=["container_opp_stats.out", "container_control_stats.out"]
        )
        assert set(df2["container"]) == {"flowcontrol", "omnetpp"}

        df3 = sim.get_docker_stats_all()
        assert set(df3["container"]) == {"flowcontrol", "omnetpp", "vadere"}


if __name__ == "__main__":
    unittest.main()
