"""
Produces a plot comparing the top k appliances and their 
contribution across a sample home each in India, UK and US
Using a stacked bars.
"""
from __future__ import print_function, division
from os.path import expanduser, join
import pandas as pd
import numpy as np
from nilmtk.dataset import DataSet
from nilmtk.sensors.electricity import ApplianceName
from nilmtk.plots import plot_series, format_axes, latexify
from nilmtk.stats.electricity.building import proportion_per_appliance
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.pyplot import Rectangle

K = 5
LOAD_DATASETS = False
DATASET_PATH = expanduser('~/Dropbox/Data/nilmtk_datasets/')
FIGURE_PATH = expanduser('~/PhD/writing/papers/e_energy_2014/'
                         'nilmtk_e_energy_2014/figures/')
BAR_WIDTH = 0.8 # bar BAR_WIDTH
FONTSIZE = 8 # for appliance names

# for plotting special iAWE text:
HSEP = 0.2
VSEP = 0.06

# From http://colorbrewer2.org
# COLORS = [
# '#a6cee3',
# '#1f78b4',
# '#b2df8a',
# '#33a02c',
# '#fb9a99',
# '#e31a1c',
# '#fdbf6f',
# '#ff7f00',
# '#cab2d6',
# '#6a3d9a',
# '#ffff99'
# ]

COLORS = [
'#8dd3c7',
'#ffffb3',
'#bebada',
'#fb8072',
'#80b1d3',
'#fdb462',
'#b3de69',
'#fccde5',
'#d9d9d9',
'#bc80bd',
'#ccebc5'
]

# List of tuples
# Each tuple is a list of synonyms
label_to_color = [
    ('others'),
    ('lighting'), 
    ('fridge'), 
    ('washer dryer','washing_machine'),
    ('kitchen outlets'),
    ('boiler'),
    ('htpc'),
    ('air conditioner'),
    ('laptop computer'),
    ('dishwasher'),
    ('entertainment unit')
]

map_label_to_color = {}
for i, label_tuple in enumerate(label_to_color):
    if isinstance(label_tuple, tuple):
        for label in label_tuple:
            map_label_to_color[label] = COLORS[i]
    else:
        map_label_to_color[label_tuple] = COLORS[i]

# Maps from human-readable name to path
DATASETS = OrderedDict()
DATASETS['REDD'] = join(DATASET_PATH, 'redd/low_freq')
# DATASETS['Smart*'] = join(DATASET_PATH, 'smart')
#DATASETS['Pecan Street'] = join(DATASET_PATH, 'pecan_1min')
# DATASETS['AMPds'] = join(DATASET_PATH, 'ampds')
DATASETS['UKPD'] = '/data/mine/vadeec/h5_cropped'
DATASETS['iAWE'] = join(DATASET_PATH, 'iawe')

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
        proportions[dataset_name] = proportion_per_appliance(electric, 
                                                             merge=True)

def prettify_name(name):
    name = name.replace('_', ' ')
    name = name[0].upper() + name[1:]
    for old, new in [('Air conditioner', 'AC'),
                     ('Htpc', 'AV'),
                     ('Boiler', 'Gas boiler'),
                     ('Washing machine', 'Clothes w.'),
                     ('Washer dryer','Clothes w.'),
                     ('Dishwasher', 'Dishwasher'),
                     ('Laptop computer', 'Laptop'),
                     ('Entertainment unit', 'AV'),
                     ('Lighting', 'Lights')]:
        if name == old:
            name = new
    if name not in ['Clothes w.', 'Gas boiler']:
        name = name.replace(' ', '\n')
    return name

plt.close('all')
latexify(columns=1, fig_height=2.5)
fig, ax = plt.subplots(ncols=1)
dataset_names = DATASETS.keys()

# Code based on http://stackoverflow.com/a/19060351/732596
prop_matrix = np.array([proportions[dataset][:K] for dataset in dataset_names])
prop_matrix = prop_matrix.transpose()
others = 1 - prop_matrix.sum(axis=0)
prop_matrix = np.vstack([others, prop_matrix])
bottoms = np.cumsum(prop_matrix, axis=0)
n = len(DATASETS)
x = np.arange(n)

def labels_for_row(j):
    if j == 0:
        labels = ['others'] * n
    else:
        labels = [proportions[dataset].index[j-1] for dataset in dataset_names]
    return labels

def colors_for_row(j):
    return [map_label_to_color[label] for label in labels_for_row(j)]

# Bottom row of bars and text:
rects = ax.bar(x, prop_matrix[0], BAR_WIDTH, color=colors_for_row(0),
#               edgecolor=colors_for_row(0))
               edgecolor='white', linewidth=0.5)
for rect in rects:
    text_x = rect.get_x()+(BAR_WIDTH/2)
    text_y = rect.get_y()+(rect.get_height()/2)
    ax.text(text_x, text_y, 'Others', ha='center', va='center', fontsize=FONTSIZE)

# Other rows of bars and text:
for j in xrange(1, K+1):
    rects = ax.bar(x, prop_matrix[j], BAR_WIDTH, bottom=bottoms[j-1], 
                   color=colors_for_row(j), 
#                   edgecolor=colors_for_row(j))
                   edgecolor='white', linewidth=0.5)
    labels = labels_for_row(j)
    for label, rect, dataset in zip(labels, rects, dataset_names):
        text_x = rect.get_x()+(BAR_WIDTH/2)
        text_y = rect.get_y()+(rect.get_height()/2)
        pretty_label = prettify_name(label)
        ha = 'center'
        special = False
        if dataset == 'iAWE':
            if label == 'kitchen outlets':
                special = True
                pretty_label = 'Kitchen'
                text_y = 1.0 + VSEP
            elif label == 'entertainment unit':
                special = True
                text_y = 1.0
            elif label == 'laptop computer':
                special = True
                text_y = 1.0 - VSEP

            if special:
                text_x = rect.get_x()+BAR_WIDTH+HSEP
                ha='left'
                ax.plot([rect.get_x()+BAR_WIDTH, text_x-0.05],
                        [rect.get_y()+(rect.get_height()/2), text_y],
                        color='k', linewidth=0.5)
                
        ax.text(text_x, 
                text_y, 
                pretty_label,
                ha=ha,
                va='center', fontsize=FONTSIZE)

ax.set_yticks([0, 0.5, 1])
ax.set_ylim([0,1+VSEP])
ax.set_xticks(x+(BAR_WIDTH/2))
ax.set_xticklabels(dataset_names)
format_axes(ax)

for spine in ['bottom', 'left']:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
plt.subplots_adjust(right=0.87, top=0.98, bottom=0.08)

ax.set_ylabel('Proportion of energy')

fig.savefig(join(FIGURE_PATH, "top_k_appliances_bar.pdf"))
print("done")
