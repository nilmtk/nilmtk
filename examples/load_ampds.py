from __future__ import print_function
from nilmtk.dataset import ampds
from nilmtk.cross_validation import train_test_split
from nilmtk.disaggregate.co_1d import CO_1d
from nilmtk.metrics import rms_error_power
from nilmtk.metrics import mean_normalized_error_power
import time
import pandas as pd


PATH = '/home/nipun/Desktop/AMPds/'
EXPORT_PATH = '/home/nipun/Desktop/temp/ampds/'
DISAGG_FEATURE = Measurement('power', 'active')
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

# Initializing CO 1D Disaggregator
disaggregator = CO_1d()
train_mains = train.utility.electric.mains[
    train.utility.electric.mains.keys()[0]][DISAGG_FEATURE]

# Get appliances data
app = train.utility.electric.appliances
train_appliances = pd.DataFrame({appliance: app[appliance][DISAGG_FEATURE] for appliance in app})

# Train
disaggregator.train(train_mains, train_appliances)

# Disaggregate
disaggregator.disaggregate(test.utility.electric.mains[
    test.utility.electric.mains.keys()[0]][DISAGG_FEATURE])

# Metrics

predicted_power = disaggregator.predictions
app_ground = test.utility.electric.appliances
ground_truth_power = pd.DataFrame({appliance: app_ground[appliance][DISAGG_FEATURE] for appliance in app_ground})

# RMS Error
re = rms_error_power(predicted_power, ground_truth_power)

# Mean Normalized Error
mne = mean_normalized_error_power(predicted_power, ground_truth_power)
