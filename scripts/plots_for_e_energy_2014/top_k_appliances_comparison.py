"""
Produces a plot comparing the top k appliances and their 
contribution across a sample home each in Canada, India, UK and US
"""
from __future__ import print_function, division
from os.path import expanduser, join
import pandas as pd
from nilmtk.dataset import DataSet
from nilmtk.sensors.electricity import ApplianceName
from nilmtk.plots import plot_series, format_axes, latexify
from nilmtk.stats.electricity.building import top_k_appliances
from collections import OrderedDict
import matplotlib.pyplot as plt


DATE_FORMAT = '%H:%M'

DATASET_PATH = expanduser('~/Dropbox/nilmtk_datasets/')

# Maps from human-readable name to path
DATASETS = OrderedDict()
#DATASETS['REDD'] = 'redd/low_freq'
DATASETS['Pecan'] = 'pecan_1min'
#DATASETS['AMPds'] = 'ampds'
#DATASETS['iAWE'] = 'iawe'


def pretty_name_appliance_names(appliance_name_list):
    return [name + str(instance) for (name, instance) in appliance_name_list]

count = -1
fig, axes = plt.subplots(ncols=len(DATASETS) + 1)
latexify(columns=2)

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
    building = dataset.buildings[1]
    electric = building.utility.electric
    electric = electric.sum_split_supplies()
    top_k = top_k_appliances(electric, k=5)
    fracs = (top_k.values * 100).tolist()
    labels = pretty_name_appliance_names(top_k.index.tolist())
    ax = axes[count]
    ax.pie(fracs,  labels=labels,
           autopct='%1.1f%%', shadow=True, startangle=90)
    ax.set_title(dataset_name)
    


count += 1
# Plotting UKPD which is on other path
dataset = DataSet()
dataset.load_hdf5("/home/nipun/Desktop/temp/ukpd", [1])
building = dataset.buildings[1]
electric = building.utility.electric
top_k = top_k_appliances(electric, k=5)
fracs = (top_k.values * 100).tolist()
labels = pretty_name_appliance_names(top_k.index.tolist())
ax = axes[count]
ax.pie(fracs,  labels=labels,
       autopct='%1.1f%%', shadow=True, startangle=90)
ax.set_title("UKPD")

for ax in axes:
    format_axes(ax)

fig.tight_layout()
fig.savefig("/home/nipun/Desktop/pie.pdf")
