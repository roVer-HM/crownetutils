import os

from pandas import IndexSlice as _I

from roveranalyzer.simulators.opp.provider.hdf.CountMapProvider import (
    CountMapHdfProvider,
)

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.realpath(__file__))
    hdf_dir = os.path.join(file_dir, "hdf_slice.hdf")

    provider = CountMapHdfProvider(hdf_dir)
    # df = provider.get_dataframe()
    # print(provider[slice(0, 5, 2)])

    # delete_me_1 = provider[2]
    # delete_me_2 = provider[0:8]
    # [5:-5] = 5:95 # throw error not implemented
    # delete_me_3 = provider[slice(0, 8, 2)]
    # data = _I[0:2, [2, 4, 5], 12.0, 6.0], ["ups_error", "rel_error"] # todo maybe later
    data = _I[0:2, [2, 4, 5], 12.0, 6.0]
    delete_me_4 = provider[data]
    # delete_me_3 = provider[[1,2]]
    # delete_me_3 = provider[1:5]
    test_tuple = {0: "simtime", 1: "x", 2: "y", 3: "ID"}
    print("")
    # todo umbauen mit handle_xx-functions
