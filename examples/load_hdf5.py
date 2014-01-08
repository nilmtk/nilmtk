from __future__ import print_function
from nilmtk.dataset import DataSet
from os.path import expanduser

H5_DIR = expanduser('~/Dropbox/Data/nilmtk_datasets/redd/low_freq/')

dataset = DataSet()
print('loading', H5_DIR)
dataset.load_hdf5(H5_DIR)

electric = dataset.buildings[1].utility.electric
