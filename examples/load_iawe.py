from __future__ import print_function
from nilmtk.dataset import iawe

EXPORT_LOCATION = '/home/nipun/Desktop/temp/iawe/'

# Load everything
dataset = iawe.IAWE()
dataset.add_mains()
dataset.add_appliances()

# Exporting the dataset into HDF5
dataset.export(EXPORT_LOCATION)

# Importing dataset from the exported location
dataset = iawe.IAWE()
dataset.load_hdf5(EXPORT_LOCATION)

# First building
building = dataset.buildings[1]
