from __future__ import print_function
from nilmtk.dataset import ampds

#dataset = ampds.AMPDS()

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
dataset.load(PATH)

# Storing data in HDF5
dataset.export(EXPORT_PATH)

# Loading data from HDF5 store
dataset=ampds.AMPDS()
dataset.load_hdf5(EXPORT_PATH)
