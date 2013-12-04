# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:42:30 2013

@author: nipun

Shows example method calls to perform diaggregation
"""
import nilmtk

# Convert HES dataset to REDD+

hes = nilmtk.dataset.hes.HES()
hes.load('hes.csv')

# Export HES dataset to REDD+ CSV
hes.export('hes_converted.csv')

# Print summary stats of dataset (number of houses; uptime; etc)
hes.print_summary_stats() 

# Extract aggregate and appliance data for house1 as Pandas DataFrame
building = hes.get_building('house1')

# Filter data using by downsampling 

# Divide data into test and train
# Modelled on scikit-learn's cross_validation:
# http://scikit-learn.org/stable/modules/cross_validation.html
[train, test] = nilmtk.cross_validation.train_test_split(building, test_size=0.4)

# Initialize disaggregator
disaggregator = nilmtk.disaggregate.ms_nalm.MS_NALM()

# Train
for appliance in ['train_fridge1.h5', 'train_fridge2.h5', 'train_kettle1.h5', ...]:
    disaggregator.train_on_appliances(appliance)

# Export the trained model
disaggregator.save_model('model.json')

# Load learnt model
disaggregator.load_model('model.json')

# Load ground truth..Should it not be a function of test dataset
ground_truth = nilmtk.load_ground_truth('individual_appliance_data.h5')

# Predict
prediction= disaggregator.predict(test)
# Jack: I think I'd prefer using 'disaggregate' rather than 'predict'.
# scikit-learn uses 'predict' because it has to use a single interface to
# loads of different ML techinques, and so has to use a very general term
# like 'predict'.  But we know that our disaggregator object will always do
# disaggregation so we can be more concrete in our method namings!

# Compute score
score={}
for metric in ['f1_score', 'roc' ,'precision' ,'MNE']:
    score[metric]= nilmtk.metrics[metric](prediction, ground_truth)

# Plot results
fig=nilmtk.metrics.std_performance_metrics(appliance_estimates, 
                                             ground_truth)
plt.plot()
