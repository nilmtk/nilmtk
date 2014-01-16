"""
Produces a plot comparing washing machines across different datasets
"""
from __future__ import print_function, division
from os.path import expanduser, join
import pandas as pd
from nilmtk.dataset import DataSet
from nilmtk.sensors.electricity import ApplianceName
from nilmtk.plots import plot_series, format_axes, latexify
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

DATE_FORMAT = '%H:%M'

DATASET_PATH = expanduser('~/Dropbox/nilmtk_datasets/')

# Maps from human-readable name to path
DATASETS = OrderedDict()
DATASETS['REDD'] = 'redd/low_freq'
#DATASETS['Pecan'] = 'pecan_1min'
#DATASETS['AMPds'] = 'ampds'
#DATASETS['iAWE'] = 'iawe'
DATASETS['UKPD'] = 'ukpd'

wm_names = {'REDD': ApplianceName('washer dryer', 1),
            'iAWE': ApplianceName('washing machine', 1),
            'UKPD': ApplianceName('washing_machine', 1),
            'AMPds': ApplianceName('washer', 1),
            'Pecan': ApplianceName('washer', 1),
            }

time_datasets = {
    'Pecan': [pd.Timestamp("2012-09-05 08:00"), pd.Timestamp("2012-09-05 11:00")],
    'REDD': [pd.Timestamp("2011-04-24 08:30"), pd.Timestamp("2011-04-24 09:55")],
    'AMPds': [pd.Timestamp("2012-04-02 04:30"), pd.Timestamp("2012-04-02 06:00")],
    'iAWE': [pd.Timestamp("2013-07-10 07:22"), pd.Timestamp("2013-07-15 07:30")],
    'UKPD': [pd.Timestamp("2013-04-14 14:00"), pd.Timestamp("2014-04-14 16:00")]

}

count = -1
latexify(columns=1)
fig, axes = plt.subplots(ncols=len(DATASETS), sharey=True)


for dataset_name, dataset_path in DATASETS.iteritems():
    count += 1
    try:
        del dataset
    except:
        pass
    dataset = DataSet()
    full_path = join(DATASET_PATH, dataset_path)
    print("Loading", full_path)
    dataset.load_hdf5(full_path, [1])
    start, end = time_datasets[dataset_name]
    building = dataset.buildings[1]
    electric = building.utility.electric
    electric = electric.sum_split_supplies()
    ax = plot_series(electric.appliances[wm_names[dataset_name]][
        ('power', 'active')][start:end],
        ax=axes[count], date_format=DATE_FORMAT, color='k')
    ax.set_title(dataset_name)
    ax.set_ylabel("")
axes[0].set_ylabel("Active Power (Watts)")

"""
count += 1
# Plotting UKPD which is on other path
dataset = DataSet()
dataset.load_hdf5("/home/nipun/Desktop/temp/ukpd", [1])
start, end = time_datasets["UKPD"]
ax = plot_series(dataset.buildings[1].utility.electric.appliances[wm_names['UKPD']][
    ('power', 'active')][start:end], ax=axes[count], date_format=DATE_FORMAT, color='k')
ax.set_title("UKPD")
ax.set_ylabel("Active Power (Watts)")
"""
for ax in axes:
    format_axes(ax)

fig.tight_layout()
fig.savefig("/home/nipun/Desktop/wm.pdf")
