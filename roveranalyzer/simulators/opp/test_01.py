import multiprocessing
import os
import time
from typing import List

from roveranalyzer.simulators.opp.provider.hdf.DcdMapProvider import DcdMapProvider

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.realpath(__file__))
    # simulation_base_path = '/home/mweidner/data/vadere00_60_20210214-21:51:11'
    # csv_file = os.path.join(simulation_base_path, "dcdMap_4807.csv")
    simulation_base_path = "/home/mweidner/data/test"
    hdf_file = os.path.join(file_path, "dcdmap_test_small.hdf")
    # dcd_df = build_dcd_dataframe(csv_file)
    provider = DcdMapProvider(hdf_file)
    # provider.create_from_csv(simulation_base_path)

    print("asd")
