from __future__ import print_function, division
from os.path import expanduser, join
import matplotlib.pyplot as plt
from copy import deepcopy
from nilmtk.dataset import DataSet
import nilmtk.stats.electricity.building as bstats
from nilmtk.plots import latexify, format_axes, SPINE_COLOR

# Attempt to automatically figure out if we need to load dataset
try:
    print(dataset)
except:
    LOAD_DATASET = True
else:
    LOAD_DATASET = False

DATASET_PATH = expanduser('~/Dropbox/Data/nilmtk_datasets/redd/low_freq')
N_APPLIANCES = 3
LATEX_PDF_OUTPUT_FILENAME = expanduser('~/PhD/writing/papers/e_energy_2014/'
                                       'nilmtk_e_energy_2014/figures/'
                                       'lost_samples.pdf')

plt.close('all')
latexify(columns=1, fig_height=1.5)
fig = plt.figure()
ax = fig.add_subplot(111)

if LOAD_DATASET:
    dataset = DataSet()
    print('Loading', DATASET_PATH)
    dataset.load_hdf5(DATASET_PATH)

electric = dataset.buildings[1].utility.electric
electric_cropped = deepcopy(electric)
electric_cropped.appliances = {k:v for k,v in electric.appliances.items()[:N_APPLIANCES]}
bstats.plot_missing_samples_using_bitmap(electric_cropped, ax=ax, cmap=plt.cm.Greys, gain=5)
format_axes(ax)
xlim = ax.get_xlim()
ax.set_title('')
plt.tight_layout()

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color(SPINE_COLOR)
    ax.spines[spine].set_linewidth(0.5)

# format appliance labels
ytext = [t.get_text() for t in ax.get_yticklabels()]
formatted_ytext = []
for text in ytext:        
    if 'mains' not in text:
        text = text[:-2]
    formatted_ytext.append(text[0].upper() + text[1:])

ax.set_yticklabels(formatted_ytext)

fig.text(0.95,0.6,'''Dropout rate (\%)''', fontsize=8, rotation=90, va='center')
plt.subplots_adjust(bottom=0.3, left=0.27, top=0.96)
plt.savefig(LATEX_PDF_OUTPUT_FILENAME)
plt.show()
