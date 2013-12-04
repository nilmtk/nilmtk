# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:42:30 2013

@author: nipun

Shows example method calls to perform diaggregation
"""
# Convert HES dataset to REDD++
hes_redd =nilmtk.format_convertors('hes.csv')

# Export HES dataset to REDD+ CSV
nilmtk.export(hes_redd, 'hes.csv')

# Load standardized HES dataset 
dataset = nilmtk.dataset.load('hes.csv') 

# Print summary stats of dataset (number of houses; uptime; etc)
dataset.print_summary_stats() 

# Extract aggregate and appliance data for house1 as Pandas DataFrame
building = dataset['house1'] 

# Filter data using by downsampling 

# Divide data into test and train
[train, test]= nilmtk.preprocess.divide( ,how={"ratio":[4,1]})

# Initialize disaggregator
disaggregator = nilmtk.disaggregate.ms_nalm.MS_NALM()

# Train
for appliance in ['train_fridge1.h5', 'train_fridge2.h5', 'train_kettle1.h5', ...]:
    disaggregator.train(appliance)
    
# Export the trained model
disaggregator.save('model.json')

# Load learnt model
disaggregator= nilmtk.load.load_model('model.json')

# Load ground truth..Should it not be a function of test dataset
ground_truth = nilmtk.load_ground_truth('individual_appliance_data.h5')

# Predict
prediction= disaggregator.predict(test)

# Compute score
score={}
for metric in ['f1_score', 'roc' ,'precision' ,'MNE']:
    score[metric]= nilmtk.metrics[metric](prediction, ground_truth)

# Plot results
fig=nilmtk.metrics.std_performance_metrics(appliance_estimates, 
                                             ground_truth)
plt.plot()
