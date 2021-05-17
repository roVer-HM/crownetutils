import os

from pandas import IndexSlice as _I

from roveranalyzer.simulators.opp.provider.hdf.CountMapProvider import (
    CountMapHdfProvider,
)

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.realpath(__file__))
    hdf_dir = os.path.join(file_dir, "hdf_slice.hdf")

    provider = CountMapHdfProvider(hdf_dir)

    # resulting condition array
    # case_1 = provider[1]  # ['ID=1']
    # case_2 = provider[1:10]  # ['ID<=10', 'ID>=1']
    # case_3 = provider[1, 2]  # ['simtime=1', 'x=2']
    # case_4 = provider[1, 2, 3]  # ['simtime=1', 'x=2', 'y=3']
    # case_5 = provider[1, 2, 3, 4]  # ['simtime=1', 'x=2', 'y=3', 'ID=4']
    # case_6 = provider[1, 2, 3, 4, 5]  # ValueError: To many values in tuple. Got: 5 expected: <=4
    # case_7 = provider[1:5, None, None, 1:5]  # ['simtime<=5', 'simtime>=1', 'ID<=5', 'ID>=1']
    # case_8 = provider[_I[2]]  # ['ID=2']
    # case_9 = provider[_I[2, None, 4]]  # ['simtime=2', 'y=4']
    # case_10 = provider[_I[1, 2, 3, 4]]  # ['simtime=1', 'x=2', 'y=3', 'ID=4']
    # case_11 = provider[_I[1, 2, 3, 4, 5]]  # ValueError: To many values in tuple. Got: 5 expected: <=4
    # case_12 = provider[[1, 5, 10]]  # ['ID=1', 'ID=5', 'ID=10']
    # case_13 = provider[_I[[1, 5, 10]]]  # ['ID=1', 'ID=5', 'ID=10']
    # case_14 = provider[_I[[1, 5, 10], None, [1, 4]]]  # ['simtime=1', 'simtime=5', 'simtime=10', 'y=1', 'y=4']

    dataframe_1 = provider[2, 1.0:50.0, None, 10:20]  # ['simtime=2', 'x=6', 'y=12.0']
    print("")
