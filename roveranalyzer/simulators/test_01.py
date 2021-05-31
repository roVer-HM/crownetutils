import os

from pandas import IndexSlice as _I

from roveranalyzer.simulators.opp.provider.hdf.CountMapProvider import CountMapProvider

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.realpath(__file__))
    hdf_dir = os.path.join(file_dir, "hdf_slice.hdf")

    provider = CountMapProvider(hdf_dir)

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
    # case15 = provider[_I[[1, 5, 10], None, [1, 4]], _I["err"]]  # condition: ['simtime=2', 'x=6.0', 'y=12.0', 'ID<=30', 'ID>=10'], columns: ['err']
    # case16 = provider[_I[[1, 5, 10], None, [1, 4]], _I["not_existing"]]  # condition: ['simtime=2', 'x=6.0', 'y=12.0', 'ID<=30', 'ID>=10'], columns: ['err']
    # case17 = provider[_I[0:10, 6.0], ["err", "sqerr"]]  # condition: ['simtime<=10', 'simtime>=0', 'x=6.0'], columns: ['err', 'sqerr']
    # case18 = provider[_I[0:10], ["err", "sqerr"]]  # throwing error

    # TODO: conditions
    #       [✓] 1. p[2] -> ID (single) (✓)
    #       [✓] 2. p[0:5] -> ID (range 0-5)
    #       [✓] 3. p[slice(0,5,4)] -> ID (range 0-5)  + warning for step_size != 0
    #       [✓] 4. p[I[1,2,3,4]] -> simtime (single), x (single), y (single), ID (single)
    #       [✓] 5. p[I[1,None,None,4]] -> simtime (single), x (ignore), y (ignore), ID (single) + handle None
    #       [✓] 6. p[I[1,2]] -> simtime (single), x(single), y (ignore), ID(ignore) + fill
    #       [✓] 7. p[I[1,2,3,4,5,6,7,8]] -> to many values error
    #  TODO:
    #       [✓] 0. Check if  moving Handlers and Dispatcher to Parent is a good idea ?!? -> yes it is
    #   ->  [✓] 1. function to fill None values to missing tuple elements (implement me)
    #       [✓] 2. Check for Tuple in Tuple -> Case for index and column slicer
    #           -> allow column select for hdf columns=["err"]
    #       [✓] 2.5 check if column exists -> if returned dataframe is empty -> raise ValueError
    #       [✓] 2.75 fix bug for provider[_I[0:10], ["err"]]
    #       [x] 3. Unittests for CountMapProvider

    # row_slice = _I[2, 6.0, 12.0, 10:30]
    # col_slice = ["err"]
    # dataframe = provider[row_slice, col_slice]
    # dataframe = provider[_I[0:10], ["err"]]

    # dataframe = provider[_I[0:10, 6.0, 12.0, 4]]
    # dataframe = provider[_I[0:10, 6.0, 12.0, 4], _I["sqerr"]]
    # dataframe = provider[_I[0:10, 6.0, 12.0, 4], _I["sqerr"]]
    # dataframe = provider[_I[0:10, 6.0, 12.0, 4], "sqerr"]
    # dataframe = provider[_I[0:10, 6.0, 12.0, 4], _I["err", "sqerr", "owner_dist", "count"]]
    # dataframe = provider[_I[0:10, 6.0, 12.0, 4], _I["err", "sqerr"]]
    # dataframe = provider[_I[0:10, 6.0], _I["err", "sqerr"]]
    # dataframe = provider[_I[4], _I["err", "sqerr"]]
    # dataframe = provider[_I[1], _I["err"]]
    dataframe = provider[2, ("err",)]
    print("")
