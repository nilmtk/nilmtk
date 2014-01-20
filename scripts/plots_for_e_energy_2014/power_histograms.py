from __future__ import print_function, division
from os.path import expanduser, join
import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.pyplot import Rectangle
from matplotlib.ticker import MaxNLocator, FuncFormatter
from nilmtk.dataset import DataSet
from nilmtk.sensors.electricity import ApplianceName
from nilmtk.plots import plot_series, format_axes, latexify
from nilmtk.stats.electricity.building import proportion_per_appliance
from nilmtk.preprocessing.electricity.single import normalise_power

LOAD_DATASETS = False
DATASET_PATH = expanduser('~/Dropbox/Data/nilmtk_datasets/')
FIGURE_PATH = expanduser('~/PhD/writing/papers/e_energy_2014/'
                         'nilmtk_e_energy_2014/figures/')

NORMALISED_BAR_COLOR = 'gray'
UNNORMALISED_LINE_COLOR = 'k'
RESAMPLE_RULE = '30S'
UK_NOMINAL_VOLTAGE = 242.0
INDIA_NOMINAL_VOLTAGE = 230
TITLE_Y = 1.0
MINIMUM_BIN_COUNT = 100

# Maps from human-readable name to path
DATASETS = OrderedDict()
#DATASETS['REDD'] = join(DATASET_PATH, 'redd/low_freq')
# DATASETS['Smart*'] = join(DATASET_PATH, 'smart')
#DATASETS['Pecan Street'] = join(DATASET_PATH, 'pecan_1min')
# DATASETS['AMPds'] = join(DATASET_PATH, 'ampds')
DATASETS['iAWE'] = join(DATASET_PATH, 'iawe')
DATASETS['UKPD'] = '/data/mine/vadeec/h5'

if LOAD_DATASETS:
    electrics = {}
    for dataset_name, dataset_path in DATASETS.iteritems():
        dataset = DataSet()
        print("Loading", dataset_path)
        dataset.load_hdf5(dataset_path, [1])
        building = dataset.buildings[1]
        electric = building.utility.electric
        electrics[dataset_name] = electric

ukpd_voltage = electrics['UKPD'].mains[(1,1)][('voltage', '')]
ukpd_voltage = ukpd_voltage.resample(RESAMPLE_RULE)
ukpd_apps = electrics['UKPD'].appliances

iawe_ac = electrics['iAWE'].appliances[('air conditioner', 1)]
iawe_ac = iawe_ac[(iawe_ac.index > '7-13-2013') & (iawe_ac.index < '8-4-2013')]
iawe_voltage = iawe_ac[('voltage','')]
iawe_voltage = iawe_voltage[(iawe_voltage > 160) & (iawe_voltage < 260)]

labels = ['Washer dryer', 'Toaster', 'Air conditioning']
chans = [(ukpd_apps[('washing_machine', 1)][('power', 'active')], 
          ukpd_voltage,
          UK_NOMINAL_VOLTAGE),
          (ukpd_apps[('toaster', 1)][('power', 'active')], 
           ukpd_voltage,
           UK_NOMINAL_VOLTAGE),
         (iawe_ac[('power', 'active')],
          iawe_voltage,
          INDIA_NOMINAL_VOLTAGE)]

plt.close('all')
latexify(columns=2, fig_height=1.8)
fig, axes = plt.subplots(ncols=3)
for i, (chan, voltage, nominal_voltage) in enumerate(chans):
    name = labels[i]
    if name != 'Air conditioning':
        chan = chan.resample(RESAMPLE_RULE)

    if name == 'Toaster':
        min_power = 1490
        max_power = 1640
    elif name == 'Washer dryer': 
        min_power = 2
        max_power = 2050
    elif name == 'Air conditioning':
        min_power = 1450
        max_power = 2350

    chan = chan[(chan > min_power) & (chan < max_power)]
    normed = normalise_power(chan, voltage, nominal_voltage)

    ax = axes[i]
    
    # First get unconstrained histogram from which we will 
    # automatically find a sensible range
    # hist, bin_edges = np.histogram(chan, bins=100)
    # above_threshold = np.where(hist > MINIMUM_BIN_COUNT)[0]

    # if len(above_threshold) < 1:
    #     print(name, "does not have enough data above threshold")
    #     continue

    # min_power = int(round(bin_edges[above_threshold[0]]))
    # max_power = int(round(bin_edges[above_threshold[-1]+1]))

    n_bins = int(round((max_power-min_power)/2))

    # Draw histogram for normalised values
    n, bins, patches = ax.hist(normed.values, 
                               facecolor=NORMALISED_BAR_COLOR,
                               edgecolor=NORMALISED_BAR_COLOR,
                               range=(min_power, max_power), 
                               bins=n_bins)

    # Draw histogram for unnormalised values
    ax.hist(chan.values,
            histtype='step',
            color=UNNORMALISED_LINE_COLOR,
            alpha=0.5,
            range=(min_power, max_power), 
            bins=n_bins)

    # format plot
#    ax.set_axis_bgcolor('#eeeeee')
    yticks = ax.get_yticks()
    ax.set_yticks([])
    format_axes(ax)
    ax.spines['left'].set_visible(False)
    title_x = 0.5
    ax.set_xlim([min_power, max_power])
    if name == 'Washer dryer':
        ax.set_ylim([0, np.max(n)*0.7])
        ax.set_xlim([0,max_power])
        ax.set_xticks([0,1000,2000])
    elif name == 'Toaster':
        ax.set_ylim([0, np.max(n)*1.0])
#        ax.set_xlim([1500, 1640])
        ax.set_xticks([1500,1600])
    elif name == 'Air conditioning':
        ax.set_xticks([1500,1900,2300])
    else:
        ax.set_ylim([0, np.max(n)*1.2])

    if i==0:
        ax.set_ylabel('Frequency')
    elif i==1:
        ax.set_xlabel('Power (kW)')

    def watts_to_kw(x, pos):
        return '{:.1f}'.format(x/1000)

#    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(watts_to_kw))
    ax.set_title(name, x=title_x, y=TITLE_Y, ha='center')

fig.tight_layout()
fig.subplots_adjust(hspace=0.1, wspace=0.2)
fig.savefig(join(FIGURE_PATH, "power_histograms.pdf"))
print("done")
