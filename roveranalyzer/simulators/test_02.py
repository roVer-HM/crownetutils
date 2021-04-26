import os

import numpy as np
import pandas as pd

file_dir = os.path.dirname(os.path.realpath(__file__))

# -----basic provider with both index and columns-----
# basic_index_file = os.path.join(file_dir, 'DataFrameBasicIndex')
# df = pd.DataFrame(np.random.rand(5,5),index=list('ABCDE'), columns=list('abcde'))
# df.to_hdf(basic_index_file, 'df_key', format='t', data_columns=True)
# df = pd.read_hdf(basic_index_file, 'df_key')
# df = pd.read_hdf(basic_index_file, 'df_key', where='a > 0.2')
# df = pd.read_hdf(basic_index_file, 'df_key', where='a > 0.2 & index==A')

# -----basic provider with multi-index index and columns-----
multi_index_file = os.path.join(file_dir, "DataFrameMultIndex")
# # index=pd.MultiIndex.from_product([list('abc'),date_range('20140721',periods=3)],names=['symbol','date'])
# mult_idx = pd.MultiIndex.from_arrays([
#     ['Mario', 'Thomas', 'Markus', 'Manuel', 'Alexander'],
#     ['Weidner', 'Muster', 'Mann', 'Gross', 'Stein']
# ], names=['Vorname', 'Nachname'])
# df = pd.DataFrame(np.random.rand(5, 5), index=mult_idx, columns=list('abcde'))
# # df.to_hdf(multi_index_file, 'df_key', format='t', data_columns=True)
# df = pd.read_hdf(multi_index_file, 'df_key')
# df = pd.read_hdf(multi_index_file, 'df_key', where='a > 0.2')
# df = pd.read_hdf(multi_index_file, 'df_key', where='a > 0.2 & Vorname=Mario')

# -----basic provider with multi-index index and multi-index columns-----
dataframe_dir = os.path.join(file_dir, "DataFrame_Crownet_SQERR.h5")
# index=pd.MultiIndex.from_product([list('abc'),date_range('20140721',periods=3)],names=['symbol','date'])
index_count = 60
idxs = [i for i in range(index_count)]
xs = [i for i in range(index_count)]
ys = [i for i in range(index_count)]
ids = [i for i in range(index_count)]
mult_idx = pd.MultiIndex.from_arrays(
    [idxs, xs, ys, ids], names=["simtime", "x", "y", "id"]
)

# mul_cols = pd.MultiIndex.from_product([
#     [0, 1, 2, 3, 4],
#     ['count', 'err', 'owner_dist']
# ], names=['ID', 'values'])

df = pd.DataFrame(
    np.random.rand(index_count, 4),
    index=mult_idx,
    columns=["count", "err", "owner_dist", "sqerr"],
)
# df.to_hdf(dataframe_dir, 'sqerr', format='t', data_columns=True)

df_readed = pd.read_hdf(dataframe_dir, "sqerr")  # read whole dataframe form h5-file
df_id_eq_42 = pd.read_hdf(
    dataframe_dir, "sqerr", where="id=42"
)  # query by row index equls
df_id_gr_42 = pd.read_hdf(
    dataframe_dir, "sqerr", where="id>42"
)  # query by row index greater
df_id_eq_42_and_50 = pd.read_hdf(
    dataframe_dir, "sqerr", where="id=[42,50]"
)  # query multiple rows
df_id_42_to_50 = pd.read_hdf(
    dataframe_dir, "sqerr", where="id > 42 & id <= 50"
)  # select rows in range
df_id_eq_42_to_50_only_sqerr = pd.read_hdf(
    dataframe_dir, "sqerr", where="id=[42]", columns=["sqerr", "owner_dist"]
)  # select row and column
df_sqerr_range = pd.read_hdf(
    dataframe_dir, "sqerr", where=["sqerr>0.5", "sqerr<=0.7"]
)  # select row and column
even_ids = []
for i in range(index_count):
    if i % 2 == 0:
        even_ids.append(i)
df_sqerr_complex = pd.read_hdf(
    dataframe_dir, "sqerr", where="id in {} & sqerr<0.5".format(even_ids)
)  # select row and column

# using the store
store = pd.HDFStore(dataframe_dir, mode="r")
store.select("sqerr", where=["id=5"])

huge_file_dir = os.path.join(file_dir, "Huge_File_SQERR.h5")
# create huge file
# huge_number = 10000000  # 10.000.000
# batch_size = 10000
# store = pd.HDFStore(huge_file_dir, mode='a')
# for i in range(0, huge_number, batch_size):
#     idxs = [n for n in range(i, i + batch_size)]
#     xs = [n for n in range(i, i + batch_size)]
#     ys = [n for n in range(i, i + batch_size)]
#     ids = [n for n in range(i, i + batch_size)]
#     mult_idx = pd.MultiIndex.from_arrays(
#         [idxs, xs, ys, ids],
#         names=['simtime', 'x', 'y', 'id'])
#     df = pd.DataFrame(np.random.rand(batch_size, 4), index=mult_idx, columns=['count', 'err', 'owner_dist', 'sqerr'])
#     store.put("huge_dataframe", df, format='t', append=True, data_columns=True)
# store.close()

# read huge file
import time

start = time.time()
print("hello")
end = time.time()
print(end - start)
store = pd.HDFStore(huge_file_dir, mode="r")
# store.select('huge_dataframe')  # selects whole file takes ages
start = time.time()
store.select("huge_dataframe", where=["id=2"])
end = time.time()
print("single entry: %s" % (end - start))
start = time.time()
store.select("huge_dataframe", where=["sqerr>0.9"])
end = time.time()
print("sqerr > 0.9: %s" % (end - start))
start = time.time()
huge_df_350k_to_500k_simtime = pd.read_hdf(
    huge_file_dir, "huge_dataframe", where=["simtime>350000", "simtime<=500000"]
)
end = time.time()
print("350k-500k: %s" % (end - start))

##############################################################################
################## Crap code i dont know if i use it anymore##################
##############################################################################
# df_id_42_to_50 = pd.read_hdf(dataframe_dir, 'sqerr', where='id=42')
# also an great example but needs further reading into pandas indexes, maybe something like an SimulationAreaIndex could be done
# to select tile [0,0] f.e.
# c = store.select_column('df','index')
# where = pd.DatetimeIndex(c).indexer_between_time('12:30','4:00')
# store.select('df',where=where)

# df = pd.DataFrame(np.random.rand(5, 5), index=mult_idx, columns=[0, 1, 2, 3, 4])
# stacked = df.stack()
# df.to_hdf(double_multi_index_file, 'df_key', format='t', data_columns=True)
# stacked.to_hdf(double_multi_index_file, 'df_key', format='t', data_columns=True)

# df = pd.read_hdf(double_multi_index_file, 'df_key')
# df = pd.read_hdf(double_multi_index_file, 'df_key', where='a > 0.2')
# df = pd.read_hdf(double_multi_index_file, 'df_key', where='a > 0.2 & Vorname=Mario')
print("asdasd")

# COMPLETE DATAFRAME
# df_complete = pd.read_hdf(h5_dir)
# sim_times_2 = list(map(lambda x: x[0], df_complete.index.values))
# x_2 = list(map(lambda x: x[1], df_complete.index.values))
# y_2 = list(map(lambda x: x[2], df_complete.index.values))
# arrays_2 = [sim_times_2, x_2, y_2]
# index = pd.MultiIndex.from_arrays(arrays_2, names=['simtime', 'x', 'y'])
# data = df_complete.stack().astype(float)


# stack up columns and add them to as value
# df_test = pd.read_hdf('/home/mweidner/data/count_map_tabled_3.h5', 'output', where='simtime=2')
# with pd.HDFStore('/home/mweidner/data/count_map_tabled_3.h5', 'w') as store:
#     store.append('output', df_complete.stack().astype(float))

# df_2 = pd.DataFrame(df_complete.stack().astype(float),
#                     index=index)
# df_complete.to_hdf(h5_table_dir, 'df', mode='w', format='table')

# dummy multi index dataframe
# index = pd.MultiIndex.from_arrays(arrays, names=['simtime', 'x', 'y'])
# df = pd.DataFrame(np.arange(6).reshape(6, -1),
#                   index=index,
#                   columns=['value'])
# df.to_hdf(h5_table_dir, 'df', mode='w', format='table')

# some reads with conditions
# df_asad = pd.read_hdf(h5_table_dir, 'df')
# df_simtimes = pd.read_hdf(h5_table_dir, 'df', where='simtime=2')

# bullshit stackoverflow (not working)
# df2 = pd.DataFrame(np.arange(9).reshape(9, -1),
#                    index=pd.MultiIndex.from_product([list('abc'),
#                                                      pd.date_range('20140721', periods=3)], names=['symbol', 'date']),
#                    columns=['value'])
# df.to_hdf(h5_table_dir, 'df', mode='w', format='table')

# df1 = pd.read_hdf(h5_table_dir, 'df', where='index=2')
# df1 = pd.read_hdf(h5_table_dir, 'df')
# df2 = pd.read_hdf(h5_table_dir, 'df', where='y=12')

# df2 = pd.DataFrame(np.arange(9).reshape(9, -1),
#                    index=pd.MultiIndex.from_product(
#                        [list('abc'),
#                         pd.date_range('20140721', periods=3)],
#                        names=['symbol', 'date']),
#                    columns=['value'])

#
# length = df.shape[0]
# df2 = pd.DataFrame(np.arange(length).reshape(length, -1),
#                   index=pd.MultiIndex.from_frame(df),
#                   columns=['values'])
#
# df.to_hdf(h5_table_dir, 'results_table', mode='w', data_columns=['simtime'],
#           format='table')
# df = pd.DataFrame(np.random.randn(9),
#                   index=["simtime", "x", "y"],
#                   names=['symbol', 'date']), columns = ['value'])
# df = pd.DataFrame(np.random.randn(3, 8), index=["A", "B", "C"])

print("done")

# df = DataFrame(np.arange(9).reshape(9,-1),
# index=pd.MultiIndex.from_product([
# list('abc'),
# date_range('20140721',periods=3)],
# names=['symbol','date']),
# columns=['value'])
# df.to_hdf('test.h5','df',mode='w',format='table')
# pd.read_hdf('test.h5','df',where='date=20140722')
# pd.read_hdf('test.h5','df',where='symbol="a"')
