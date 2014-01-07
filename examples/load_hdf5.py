from __future__ import print_function
from nilmtk.dataset import DataSet

H5_DIR = '/home/jack/Dropbox/Data/nilmtk_datasets/redd/low_freq/'

dataset = DataSet()
print('loading', H5_DIR)
dataset.load_hdf5(H5_DIR)

electric = dataset.buildings[1].utility.electric
