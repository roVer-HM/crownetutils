import os

import cv2
import h5py
import numpy as np
import pandas as pd

from roveranalyzer.simulators.opp.provider.hdf.CountMapProvider import (
    CountMapHdfProvider,
)
from roveranalyzer.simulators.opp.provider.hdf.IHdfProvider import IHdfProvider
from roveranalyzer.simulators.opp.provider.hdf.Operation import Operation

# dits
file_dir = os.path.dirname(os.path.realpath(__file__))
hdf_dir = os.path.join(file_dir, "pandas_to_hdf.h5")

# create test dataframe
index_count = 60
idxs = [i for i in range(index_count)]
xs = [i for i in range(index_count)]
ys = [i for i in range(index_count)]
ids = [i for i in range(index_count)]
mult_idx = pd.MultiIndex.from_arrays(
    [idxs, xs, ys, ids], names=["simtime", "x", "y", "ID"]
)
df = pd.DataFrame(
    np.random.rand(index_count, 4),
    index=mult_idx,
    columns=["count", "err", "owner_dist", "sqerr"],
)
# transform to array with indexes seperated
# sa = df_to_sarray(df.reset_index())
# safe this array
# with h5py.File(hdf_dir, 'w') as hf:
#     hf['count_map'] = sa

# load the h5 dataset
# with h5py.File(hdf_dir) as hf:
#     sa2 = hf['count_map'][:]

# Extract the xxx column
# todo

print("check new array breakpoint")

# todo new provider for h5py and viewability with panoply
# load provider-provider

panoply_test_dir = os.path.join(file_dir, "panoply_test_2.h5")
lenna_img = cv2.imread(os.path.join(file_dir, "lenna.png"))

# df = pd.read_hdf(panoply_test_dir, "dataset/something.png")
# df = pd.read_hdf(panoply_test_dir, "dataset/count_map")
# print("")
# f = h5py.File(panoply_test_dir, 'r')
# dat = CountMapHdfProvider(panoply_test_dir).get_data("lenna")
# f.close()
# show image
# cv2.imshow("new_window",dat)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

hdf_provider = CountMapHdfProvider(panoply_test_dir)
hdf_provider.write_dataframe(df)
hdf_provider.add_data(lenna_img, "lenna")

# h = h5py.File(panoply_test_dir, 'a')
# # h.create_group("Lenna_Image")
# # h.create
# h.create_dataset("dataset/something.png", data=lenna_img, dtype='uint8')
# h.close()

# print("asda")

# image_hdf_path = os.path.join(file_dir, "img_file.h5")
# h2 = h5py.File(image_hdf_path)
# attrs = h2["Photos"]["Image 1"].attrs.items()
# for a in attrs:
#     print(a)
# print("asdasd")

# hdf = CountMapHdfProvider(hdf_path=hdf_dir)
# h = h5py.File(hdf_dir)
# attrs = h["count_map"]["table"].attrs.items()
# for a in h["count_map"]["table"].attrs.items():
#     print(a)
# hdf.test()
# hdf.write_dataframe(df)
# hdf.addColumn()

# print("asdasdas")

# todo
# lenna_img = cv2.imread('D:\\Sources\\roveranalyzer\\roveranalyzer\\simulators\\lenna.png')
# lenna_img = cv2.imread(os.path.join(file_dir, "lenna.png"))
# img_data = [lenna_img, lenna_img]

# hdf_file = h5py.File(hdf_dir, 'w')
# hdf_file.create_dataset("Image_1", data=lenna_img, dtype='uint8')
# hdf_file.close()
# data = self.get_dataframe()
# data["LennaImg"] = img_data
# print("")
# with self.ctx(mode='w') as store:
#     store.put(key=self.group, value=data, format='t', data_columns=True)


# i5 = hdf.select_id_exact(value=5)
# i5_gr = hdf.select_id_exact(value=5, operation=Operation.GREATER_EQ)
# i5_10 = hdf.select_id_exact(value=[5, 6, 7, 8, 9, 10])
# i_range = hdf.select_id_range(_min=5, _max=12)
# s_5 = hdf.select_simtime_exact(value=5)
# s_5_8_20 = hdf.select_simtime_exact(value=[5, 8, 20])
# s_range = hdf.select_id_range(_min=5, _max=12)
# sqerr_range = hdf.select_sqerr_range(0.1, 0.5)
print("asd")

# class SimTimeEvent():
#  simtime:
#  ...
#
# wrapper = Wrapper(path,table_name)
# wrapper.append(simtime_event: SimTimeEvent);

##############################################################################
################## TODOS ##################
##############################################################################

# 0. Schritt
# Tool zum anschauen
# - https://www.giss.nasa.gov/tools/panoply/download/
# - HDFView
# - sudo apt-get install hdfview
# evlt .png einlesen abspeichern etc.
# https://portal.hdfgroup.org/display/support/HDFView+3.1.2#files
# https://support.hdfgroup.org/ftp/HDF5/releases/HDF-JAVA/hdfview-3.1.2/bin/ (Works on WIndows)
# This link -> https://www.hdfgroup.org/downloads/hdfview/
# register with tmp email and download
# also needs the hdfview.bat for windows
# (https://support.hdfgroup.org/ftp/HDF5/releases/HDF-JAVA/hdfview-3.1.2/hdfview_scripts/hdfview.bat)

# 1. Schritt
# HDFStoreWrapper
# wrapper.selectById list []  single or multiple
# wrapper.selectBySimtime
# selectBySimtime -> selectByKey mit 'simtime'
# CountMapProvider-Klasse und IHdfProvider
# in roveranalyzer/simulators/opp/privider/hdf

# 2. Schritt
# pickle -> provider-files
# schauen ob man mit h5 einfach so spalten hinzufügen kann
