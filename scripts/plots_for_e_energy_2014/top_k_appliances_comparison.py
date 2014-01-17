"""
Produces a plot comparing the top k appliances and their 
contribution across a sample home each in India, UK and US
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
from matplotlib.pyplot import Rectangle

K = 5
LOAD_DATASETS = True
DATASET_PATH = expanduser('~/Dropbox/Data/nilmtk_datasets/')
FIGURE_PATH = expanduser('~/PhD/writing/papers/e_energy_2014/latex/figures/')

# From http://colorbrewer2.org
COLORS = [
'#a6cee3',
'#1f78b4',
'#b2df8a',
'#33a02c',
'#fb9a99',
'#e31a1c',
'#fdbf6f',
'#ff7f00',
'#cab2d6',
'#6a3d9a',
'#ffff99'
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

def get_colors_for_labels(labels):
    colors = []
    for label in labels:
        for i, labs in enumerate(label_to_color):
            if label in labs:
                colors.append(COLORS[i])
                print(label, i)
                break
    return colors

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
        proportions[dataset_name] = proportion_per_appliance(electric, 
                                                             merge=True)

def pretty_name_appliance_names(appliance_name_list):
    names = [name.replace('_', ' ')  for name in appliance_name_list]
    names = [name[0].upper() + name[1:] for name in names]
    for old, new in [('Htpc', 'Home theatre PC'), 
                     ('Boiler', 'Gas boiler'),
                     ('Air conditioner', 'Air conditioning')]:
        try:
            names[names.index(old)] = new
        except ValueError:
            pass
    return names

plt.close('all')
latexify(columns=2, fig_height=2)
fig, axes = plt.subplots(ncols=4, subplot_kw={'aspect':1})

count = 0
for dataset_name, dataset_path in DATASETS.iteritems():
    print("\n------------------", dataset_name, "-----------------")
    top_k = proportions[dataset_name][:K]
    print(top_k)
    fracs = top_k.tolist()
    fracs.append(1.0 - sum(fracs))
    labels_raw = top_k.index.tolist()
    labels_raw.append('others')
    colors = get_colors_for_labels(labels_raw)
    pretty_labels = pretty_name_appliance_names(labels_raw)
    explode = (0.03, 0.03, 0.03, 0.03, 0.03, 0.03)
    ax = axes[count]
    ax.pie(fracs, 
           labeldistance=0.9, 
           colors=colors,
#           autopct='%1.0f%%', explode=explode, labels=pretty_labels,
           radius=1.1,
           shadow=False, startangle=90)
    ax.set_title(dataset_name)
    count += 1

# draw legend
names = []
for tup in label_to_color:
    if isinstance(tup, tuple):
        names.append(tup[0])
    else:
        names.append(tup)
names = pretty_name_appliance_names(names)
proxy_artists = [Rectangle((0,0),1,1,fc=color) for color in COLORS]
ax = axes[-1]
ax.axis('off')
ax.legend(proxy_artists, names, loc='center')
fig.tight_layout()
fig.subplots_adjust(wspace=.001)
fig.savefig(join(FIGURE_PATH, "top_k_appliances_pie.pdf"))
print("done")
