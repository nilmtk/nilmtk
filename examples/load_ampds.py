from __future__ import print_function
from nilmtk.dataset import ampds
from nilmtk.cross_validation import train_test_split
import time


PATH = '/home/nipun/Desktop/AMPds/'
EXPORT_PATH = '/home/nipun/Desktop/temp/ampds/'

'''
# Loading data for Home 01
ampds.load_electricity(PATH)
ampds.load_water(PATH)
ampds.load_gas(PATH)
'''

# Load everything
dataset = ampds.AMPDS()
t1 = time.time()
dataset.load(PATH)
t2 = time.time()
print("Runtime to load from AMPDS format = {:.2f}".format(t2 - t1))


# Storing data in HDF5
t1 = time.time()
dataset.export(EXPORT_PATH)
t2 = time.time()
print("Runtime to export to HDF5 = {:.2f}".format(t2 - t1))


# Loading data from HDF5 store
dataset = ampds.AMPDS()
t1 = time.time()
dataset.load_hdf5(EXPORT_PATH)
t2 = time.time()
print("Runtime to import from HDF5 = {:.2f}".format(t2 - t1))

# Dividing the data into train and test
b = dataset.buildings['Building_1']
train, test = train_test_split(b)


