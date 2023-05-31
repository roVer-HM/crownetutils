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
        ** Event #344910   t=14   Elapsed: 46.7698s (0m 46s)  100% completed  (100% total) \n \
        Speed:     ev/sec=104564   simsec/sec=2.76638   ev/simsec=37798.1 \n \
        Messages:  created: 176996   present: 563   in FES: 90 \n \
        <!> Simulation time limit reached -- at t=14s, event #344910 \n \
        Container terminated."

        f = open(path.join(self.test_dir, 'container_opp.out'), 'w')
        f.write(omnetpp_container_log_content)
        f.close()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_get_sim_real_time_ratio(self):
        # compare values from temporary file in setUp
        df_expected = pd.DataFrame(data={"sim_time": [0.000, 0.018, 10.002, 14.000],
                                         "real_time": [0.000021, 43.239500, 45.324600, 46.769800],
                                         "ratio_sim_real": [0.0000, 0.000416286, 4.7884, 2.76638]},
                                   index=[0, 1, 2, 3])

        file_path = os.path.join(self.test_dir, 'container_opp.out')
        df_actual = OppAnalysis.get_sim_real_time_ratio(omnetpp_log_file_path=file_path)

        pd.testing.assert_frame_equal(df_actual, df_expected)


if __name__ == "__main__":
    unittest.main()
