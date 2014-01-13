"""
Produces a LaTeX table summarising the datasets
"""
from __future__ import print_function, division
from os.path import expanduser, join
import pandas as pd
from nilmtk.dataset import DataSet
from nilmtk.sensors.electricity import ApplianceName
from collections import OrderedDict
import matplotlib.pyplot as plt

"""Notes
For building # 1 in each dataset
SMART* : Washing machine was not used during the instrumentation
iAWE   : 6:25 to 6:55; 21 June
AMPds  : 4:30 to 6:00; 2 April
REDD   : 23:15 to 23:59; 24 April
PECAN  : 08:00 to 11:00; 5 Sept
UKPD   : 10:24 to 11:44 10 Nov, 2012
"""


DATASET_PATH = expanduser('~/Dropbox/nilmtk_datasets/')

# Maps from human-readable name to path
DATASETS = OrderedDict()
DATASETS['REDD'] = 'redd/low_freq'
DATASETS['Pecan'] = 'pecan_1min'
DATASETS['AMPds'] = 'ampds'
#DATASETS['iAWE'] = 'iawe'

wm_names = {'REDD': ApplianceName('washer dryer', 1),
            'iAWE': ApplianceName('washing machine', 1),
            'UKPD': ApplianceName('washing_machine', 1),
            'AMPds': ApplianceName('washer', 1),
            'Pecan': ApplianceName('washer', 1),
            }

time_datasets = {
    'Pecan': [pd.Timestamp("2012-09-05 08:00"), pd.Timestamp("2012-09-05 11:00")],
    'REDD': [pd.Timestamp("2011-04-24 08:40"), pd.Timestamp("2011-04-24 09:55")],
    'AMPds': [pd.Timestamp("2012-04-02 04:30"), pd.Timestamp("2012-04-02 06:00")],
    'iAWE': [pd.Timestamp("2013-06-21 06:25"), pd.Timestamp("2013-06-21 06:55")],
    'UKPD': [pd.Timestamp("2012-11-10 10:24"), pd.Timestamp("2013-11-10 11:44")]

}

count = -1
fig, axes = plt.subplots(ncols=5)
for dataset_name, dataset in DATASETS.iteritems():
    count += 1
    dataset = DataSet()
    full_path = join(DATASET_PATH, DATASETS[dataset_name])
    print("Loading", full_path)
    dataset.load_hdf5(full_path)
    start, end = time_datasets[dataset_name]
    building = dataset.buildings[1]
    electric = building.utility.electric
    electric = electric.sum_split_supplies()
    electric.appliances[wm_names[dataset_name]][
        ('power', 'active')][start:end].plot(title=dataset_name,
                                             ax=axes[count])

count += 1
# Plotting UKPD which is on other path
dataset = DataSet()
dataset.load_hdf5("/home/nipun/Desktop/temp/ukpd")
start, end = time_datasets["UKPD"]
dataset.buildings[1].utility.electric.appliances[wm_names['UKPD']][
    ('power', 'active')][start:end].plot(title='UKPD',
                                         ax=axes[count])
plt.show()
