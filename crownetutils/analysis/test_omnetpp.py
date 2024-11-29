import os
import shutil
import tempfile
import unittest
from os import path

import pandas as pd

from crownetutils.analysis.omnetpp import OppAnalysis


class OmnetppTest(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

        omnetpp_container_log_content = "Welcome to the CrowNet OMNeT++ Docker Container. \n \
        Running simulation... \n \
        ** Event #0   t=0   Elapsed: 2.1e-05s (0m 00s)  0% completed  (0% total) \n \
        Speed:     ev/sec=0   simsec/sec=0   ev/simsec=0 \n \
        Messages:  created: 100   present: 100   in FES: 17 \n \
        ** Event #256   t=0.018   Elapsed: 43.2395s (0m 43s)  0% completed  (0% total) \n \
        Speed:     ev/sec=5.94364   simsec/sec=0.000416286   ev/simsec=14277.8 \n \
        Messages:  created: 371   present: 222   in FES: 38 \n \
        ** Event #193792   t=10.002   Elapsed: 45.3246s (0m 45s)  71% completed  (71% total) \n \
        Speed:     ev/sec=92821.2   simsec/sec=4.7884   ev/simsec=19384.6 \n \
        Messages:  created: 99188   present: 502   in FES: 86 \
        <!> Simulation time limit reached -- at t=14s, event #344910 \n \
        Container terminated."

        f = open(path.join(self.test_dir, "container_opp.out"), "w")
        f.write(omnetpp_container_log_content)
        f.close()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_get_sim_real_time_ratio(self):
        # compare values from temporary file in setUp
        df_expected = pd.DataFrame(
            data={
                "event_number": [0, 256, 193792],
                "simtime": [0.000, 0.018, 10.002],
                "realtime": [0.000021, 43.239500, 45.324600],
                "events_per_sec": [0, 5.94364, 92821.2],
                "simsec_per_realsec": [0.0000, 0.000416286, 4.7884],
                "events_per_simsec": [0, 14277.8, 19384.6],
            },
            index=[0, 1, 2],
        )

        file_path = os.path.join(self.test_dir, "container_opp.out")
        df_actual = OppAnalysis.get_sim_real_time_ratio(omnetpp_log_file_path=file_path)

        pd.testing.assert_frame_equal(df_actual, df_expected)


if __name__ == "__main__":
    unittest.main()
