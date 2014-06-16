from nilmtk.dataset import ampds
from nilmtk.disaggregate.fhmm_exact import FHMM
import json
import pandas as pd
from pandas import HDFStore
from nilmtk.dataset.ampds import Measurement
import nilmtk.preprocessing.electricity.building as prepb
from nilmtk.cross_validation import train_test_split


dataset = ampds.AMPDS()

PATH = '/home/nipun/study/datasets/AMPds/'

# Feature to perform disaggregation on
DISAGG_FEATURE = Measurement('power', 'active')

# Load data
dataset.load(PATH)

# Get data of Home_01
building = dataset.buildings[1]

# Let us filter out appliances contributing less than 5%
building = prepb.filter_contribution_less_than_x(building, x=5)


print('Dividing data into test and train')
train, test = train_test_split(building, train_size = 0.5) 

# Train
disaggregator = FHMM()
disaggregator_name = "FHMM"
t1 = time.time()
disaggregator.train(train, disagg_features=[DISAGG_FEATURE])
t2 = time.time()
print("Runtime to train for {} = {:.2f} seconds".format(disaggregator_name, t2 - t1))
train_time=t2-t1
    
# Disaggregate
t1 = time.time()
disaggregator.disaggregate(test)
t2 = time.time()
print("Runtime to disaggregate for {}= {:.2f} seconds".format(disaggregator_name, t2 - t1))
disaggregate_time=t2-t1   
        
# Predicted power and states
# Predicted power is a DataFrame containing the predicted power of different 
# appliances
predicted_power = disaggregator.predictions
app_ground = test.utility.electric.appliances
ground_truth_power = pd.DataFrame({appliance: app_ground[appliance][DISAGG_FEATURE] for appliance in app_ground})
        




