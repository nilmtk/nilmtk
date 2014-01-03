from __future__ import print_function
from nilmtk.dataset import pecan
from nilmtk.cross_validation import train_test_split
from nilmtk.metrics import *
import time
from nilmtk.sensors.electricity import Measurement
from nilmtk.disaggregate.co_1d import CO_1d

import pandas as pd

# Feature to perform disaggregation on
DISAGG_FEATURE = Measurement('power', 'active')

EXPORT_PATH = '/home/nipun/Desktop/temp/pecan_1min/'

pecan_1min = pecan.Pecan_1min()

# Name of buildings
pecan_1min.load_building_names('/home/nipun/Desktop/PECAN/')

# Loading data for Home 01
pecan_1min.load_building('/home/nipun/Desktop/PECAN/', 'Home 01')

# Accessing Home 01 Building
building = pecan_1min.buildings[1]

# Loading all buildings data
dataset = pecan.Pecan_1min()
dataset.load('/home/nipun/Desktop/PECAN/')

# Storing data in HDF5
t1 = time.time()
dataset.export(EXPORT_PATH)
t2 = time.time()
print("Runtime to export to HDF5 = {:.2f}".format(t2 - t1))

# Now importing the data from HDF5 store
t1 = time.time()
dataset = pecan.Pecan_1min()
dataset.load_hdf5(EXPORT_PATH)
t2 = time.time()
print("Runtime to importing from HDF5 = {:.2f}".format(t2 - t1))

# Doing analysis on Home_10
b = dataset.buildings[Home_10]

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
