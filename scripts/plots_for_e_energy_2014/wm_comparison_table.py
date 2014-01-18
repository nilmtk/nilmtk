"""
Produces a plot comparing washing machines across different datasets
"""
from __future__ import print_function, division
from os.path import expanduser, join
import pandas as pd
from nilmtk.dataset import DataSet
from nilmtk.sensors.electricity import ApplianceName
from nilmtk.plots import plot_series, format_axes, latexify, _to_ordinalf_np_vectorized
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

LOAD_DATASETS = False
DATASET_PATH = expanduser('~/Dropbox/Data/nilmtk_datasets/')
FIGURE_PATH = expanduser('~/PhD/writing/papers/e_energy_2014/latex/figures/')

# Maps from human-readable name to path
DATASETS = OrderedDict()
DATASETS['REDD'] = join(DATASET_PATH,'redd/low_freq')
#DATASETS['Pecan'] = 'pecan_1min'
#DATASETS['AMPds'] = 'ampds'
#DATASETS['iAWE'] = 'iawe'
DATASETS['UKPD'] = '/data/mine/vadeec/h5_cropped'

wm_names = {'REDD': ApplianceName('washer dryer', 1),
            'iAWE': ApplianceName('washing machine', 1),
            'UKPD': ApplianceName('washing_machine', 1),
            'AMPds': ApplianceName('washer', 1),
            'Pecan': ApplianceName('washer', 1),
            }

time_datasets = {
    'Pecan': [pd.Timestamp("2012-09-05 08:00"), pd.Timestamp("2012-09-05 11:00")],
    'REDD': [pd.Timestamp("2011-05-01 17:43"), pd.Timestamp("2011-05-01 23:00")],
    'AMPds': [pd.Timestamp("2012-04-02 04:30"), pd.Timestamp("2012-04-02 06:00")],
    'iAWE': [pd.Timestamp("2013-07-10 07:22"), pd.Timestamp("2013-07-15 07:30")],
    'UKPD': [pd.Timestamp("2013-10-29 10:47"), pd.Timestamp("2013-10-29 12:17")]

}

count = 0
latexify(columns=2, fig_height=2.5)
fig, axes = plt.subplots(ncols=len(DATASETS), sharey=True)

if LOAD_DATASETS:
    electrics = {}
    for dataset_name, dataset_path in DATASETS.iteritems():
        dataset = DataSet()
        print("Loading", dataset_path)
        dataset.load_hdf5(dataset_path, [1])
        building = dataset.buildings[1]
        electric = building.utility.electric
        electric = electric.sum_split_supplies()
        electrics[dataset_name] = electric

for dataset_name, dataset_path in DATASETS.iteritems():
    electric = electrics[dataset_name]
    start, end = time_datasets[dataset_name]
    data = electric.appliances[wm_names[dataset_name]][('power', 'active')][start:end]
    if dataset_name == 'UKPD':
        data = data[data < 2300]
    ax = axes[count]
    x = _to_ordinalf_np_vectorized(data.index.to_pydatetime())
    ax.plot((x - x[0])*mdates.MINUTES_PER_DAY, data.values / 1000, 
            color='k', linewidth=0.1)
#    ax.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FORMAT))
    ax.set_title(dataset_name)
    ax.set_ylabel("")
    format_axes(ax)
    ax.set_ylim([0,3.500])
    ax.set_yticks([0,1,2,3])
    ax.set_xlim([0,90])
    count += 1

axes[0].set_ylabel("Active Power (kW)")
fig.text(0.5, 0.01, "time (minutes)", ha='center', fontsize=8)
fig.tight_layout()

fig.savefig(join(FIGURE_PATH, "wm.pdf"))
