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
from nilmtk.stats.electricity.building import proportion_per_appliance
from collections import OrderedDict
import matplotlib.pyplot as plt

"""
TODO:
* are we doing all the pre-processing we need to do?
* sum multiple instances of same class of appliance
"""

K = 5
LOAD_DATASETS = False
DATASET_PATH = expanduser('~/Dropbox/Data/nilmtk_datasets/')
FIGURE_PATH = expanduser('~/PhD/writing/papers/e_energy_2014/figures/')

# From http://colorbrewer2.org
# number of data classes = 7
# qualitative
COLORS = [(141,211,199),
(255,255,179),
(190,186,218),
(251,128,114),
(128,177,211),
(253,180,98),
(179,222,105)]

COLORS = ['#8dd3c7',
'#ffffb3',
'#bebada',
'#fb8072',
'#80b1d3',
'#fdb462',
'#b3de69']

COLORS = [
'#a6cee3',
'#1f78b4',
'#b2df8a',
'#33a02c',
'#fb9a99',
'#e31a1c',
'#fdbf6f'
]
# Maps from human-readable name to path
DATASETS = OrderedDict()
DATASETS['REDD'] = join(DATASET_PATH, 'redd/low_freq')
# DATASETS['Smart*'] = join(DATASET_PATH, 'smart')
#DATASETS['Pecan Street'] = join(DATASET_PATH, 'pecan_1min')
# DATASETS['AMPds'] = join(DATASET_PATH, 'ampds')
DATASETS['iAWE'] = join(DATASET_PATH, 'iawe')
DATASETS['UKPD'] = '/data/mine/vadeec/h5_cropped'

if LOAD_DATASETS:
    electrics = {}
    proportions = {}
    for dataset_name, dataset_path in DATASETS.iteritems():
        dataset = DataSet()
        print("Loading", dataset_path)
        dataset.load_hdf5(dataset_path, [1])
        building = dataset.buildings[1]
        electric = building.utility.electric
        electrics[dataset_name] = electric
        proportions[dataset_name] = proportion_per_appliance(electric)

def pretty_name_appliance_names(appliance_name_list):
    return [name.replace('_', ' ') + str(instance) for (name, instance) in appliance_name_list]

plt.close('all')
latexify(columns=2, fig_height=3)
fig, axes = plt.subplots(ncols=3)

count = 0
for dataset_name, dataset_path in DATASETS.iteritems():
    electric = electrics[dataset_name]
    top_k = proportions[dataset_name][:K]
    print("\n------------------", dataset_name, "-----------------")
    print(top_k)
    fracs = top_k.tolist()
    fracs.append(1.0 - sum(fracs))
    labels = pretty_name_appliance_names(top_k.index.tolist())
    labels.append("Others")
    explode = (0.03, 0.03, 0.03, 0.03, 0.03, 0.03)
    ax = axes[count]
    ax.pie(fracs,  labels=labels,  labeldistance=1.0, colors=COLORS,
#           autopct='%1.0f%%', explode=explode,
           shadow=False, startangle=90)
    ax.set_title(dataset_name)
    count += 1
    del electric

"""
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
"""

for ax in axes:
    format_axes(ax)

fig.tight_layout()
fig.savefig(join(FIGURE_PATH, "top_k_appliances_pie.pdf"))
