from __future__ import print_function
from nilmtk.dataset import iawe, DataSet
from nilmtk.stats.electricity.building import top_k_appliances
from nilmtk.stats.electricity.single import get_sample_period

EXPORT_LOCATION = '/home/nipun/Desktop/temp/iawe/'

# Load everything
dataset = iawe.IAWE()
dataset.add_mains()
dataset.add_appliances()

# Exporting the dataset into HDF5
dataset.export(EXPORT_LOCATION)

# Importing dataset from the exported location
dataset = DataSet()
dataset.load_hdf5(EXPORT_LOCATION)

# First building
building = dataset.buildings[1]

# Doing stats
# Finding the sampling period of mains and appliances
for mains_name, mains in building.utility.electric.mains.iteritems():
    print ('{}: {:.2f} seconds'.format(mains_name, get_sample_period(mains)))

for appliance_name, appliance in building.utility.electric.appliances.iteritems():
    print ('{}: {:.2f} seconds'.format(
        appliance_name, get_sample_period(appliance)))

# Fixing implausible voltage values

# Finding appliance usage
