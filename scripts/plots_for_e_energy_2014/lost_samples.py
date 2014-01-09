from __future__ import print_function, division
from os.path import expanduser, join
import matplotlib.pyplot as plt
from copy import deepcopy
from nilmtk.dataset import DataSet
import nilmtk.stats.electricity.building as bstats
from nilmtk.plots import latexify, format_axes

# Attempt to automatically figure out if we need to load dataset
try:
    print(dataset)
except:
    LOAD_DATASET = True
else:
    LOAD_DATASET = False

DATASET_PATH = expanduser('~/Dropbox/Data/nilmtk_datasets/redd/low_freq')
N_APPLIANCES = 5
LATEX_PDF_OUTPUT_FILENAME = expanduser('~/Dropbox/MyWork/imperial/PhD/writing'
                                       '/papers/e_energy_2014/figures/'
                                       'lost_samples.pdf')
fig = plt.figure()
ax = fig.add_subplot(111)

if LOAD_DATASET:
    dataset = DataSet()
    print('Loading', DATASET_PATH)
    dataset.load_hdf5(DATASET_PATH)

electric = dataset.buildings[1].utility.electric

latexify(fig_height=1, fig_width=3.39)
electric_cropped = deepcopy(electric)
electric_cropped.appliances = {k:v for k,v in electric.appliances.items()[:N_APPLIANCES]}
bstats.plot_missing_samples_using_bitmap(electric_cropped, ax=ax)
format_axes(ax)

plt.savefig(LATEX_PDF_OUTPUT_FILENAME)
plt.show()
