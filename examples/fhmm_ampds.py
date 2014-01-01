from __future__ import print_function
from nilmtk.dataset import ampds
from nilmtk.cross_validation import train_test_split
from nilmtk.disaggregate.fhmm_exact import FHMM
from nilmtk.metrics import rms_error_power
from nilmtk.metrics import mean_normalized_error_power
from nilmtk.sensors.electricity import Measurement
from nilmtk.stats.electricity.building import top_k_appliances
from nilmtk.preprocessing import filter_top_k_appliances
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


# Loading data from HDF5 store
dataset = ampds.AMPDS()
t1 = time.time()
dataset.load_hdf5(EXPORT_PATH)
t2 = time.time()
print("Runtime to import from HDF5 = {:.2f}".format(t2 - t1))

# Experiment on first (and only) building
b = dataset.buildings[1]

# Filtering to include only top 6 appliances
b = filter_top_k_appliances(b, 6)

# Dividing the data into train and test
train, test = train_test_split(b)

# Again subdivide data into train, test for testing on even smaller data
#train, test = train_test_split(train, test_size=.5)


# Initializing FHMM Disaggregator
disaggregator = FHMM()
train_mains = train.utility.electric.mains[
    train.utility.electric.mains.keys()[0]][DISAGG_FEATURE]

# Get appliances data
app = train.utility.electric.appliances
train_appliances = pd.DataFrame({appliance: app[appliance][DISAGG_FEATURE] for appliance in app if DISAGG_FEATURE in app[appliance]})

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
