from nilmtk.dataset import ampds
from nilmtk.disaggregate.co_1d import CO_1d
from nilmtk.metrics
import json
import pandas as pd

dataset = ampds.AMPDS()

PATH = '/home/nipun/Desktop/AMPds/'

# Feature to perform disaggregation on
DISAGG_FEATURE = 'power_active'


# Load data
dataset.load(PATH)

# Get data of Home_01
building = dataset.buildings['Home_01']

print('Loading data....')
# Get mains data
mains = building.utility.electric.mains

# Get appliances data
app = building.utility.electric.appliances

# Finding number of data points
row = mains.index.size

print('Dividing data into test and train')
# Dividing the data into train and test (ratio 1:9)
train_aggregate, test_aggregate = mains[:row / 10], mains[row / 10:]

# Finding the active component of power for train and test aggregate (mains)
train_power_aggregate = train_aggregate[DISAGG_FEATURE]
test_power_aggregate = test_aggregate[DISAGG_FEATURE]

# Loading train and test appliance data
train_appliance = {}
test_appliance = {}
for appliance in app:
    train_appliance[appliance] = app[appliance][:row / 10]
    test_appliance[appliance] = app[appliance][row / 10:]

# Finding the active component of power for appliances
train_power_appliance = {}
test_power_appliance = {}
for appliance in app:
    train_power_appliance[appliance] = train_appliance[appliance][DISAGG_FEATURE]
    test_power_appliance[appliance] = test_appliance[appliance][DISAGG_FEATURE]

# Creating appliances dataframe
df_train_appliances = pd.DataFrame(train_power_appliance)

# Instantiating the CO Disaggregator
disaggregator = CO_1d()

# Train

# Load learnt model from disk
with open('ampds_model_co.json') as f:
    model = json.load(f)

print(model)
disaggregator.model = model
print("*"*80)

print('Disaggregate!')
# Predicting with the learnt model
disaggregator.disaggregate(test_power_aggregate)

# Accessing diaggregated data
for appliance in disaggregator.predictions:
    # Do something with self.predictions[appliance]
    pass


