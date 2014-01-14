from __future__ import print_function
from nilmtk.dataset import DataSet
from nilmtk.cross_validation import train_test_split
from nilmtk.disaggregate.fhmm_exact import FHMM
from nilmtk.disaggregate.co_1d import CO_1d
from nilmtk.metrics import rms_error_power
from nilmtk.metrics import mean_normalized_error_power, fraction_energy_assigned_correctly, f_score
from nilmtk.sensors.electricity import Measurement
from nilmtk.stats.electricity.building import top_k_appliances
from nilmtk.stats.electricity.building import find_appliances_contribution
import nilmtk.preprocessing.electricity.building as prepb
from nilmtk.dataset import DataSet
from nilmtk.plots import latexify, format_axes, plot_series

from copy import deepcopy
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import resource
import os


megs = 5000
resource.setrlimit(resource.RLIMIT_AS, (megs * 1048576L, -1L))

# <codecell>

import warnings
warnings.filterwarnings('ignore')


dataset_names = ['iawe']

# <codecell>

metrics = ['mne', 'fraction', 'f_score']

# <codecell>

base_path = "/home/nipun/Dropbox/nilmtk_datasets/"

# <codecell>

DISAGG_FEATURE = Measurement('power', 'active')

# <codecell>

disaggregators = {}
disaggregators['fhmm'] = FHMM()
disaggregators['co'] = CO_1d()

# <codecell>

actual_names_datasets = {'redd/low_freq': 'REDD',
                         'iawe': 'iAWE',
                         'pecan_1min': 'PECAN',
                         'ampds': 'AMPds'
                         }


# <codecell>

metric_function = {'mne': mean_normalized_error_power,
                   'fraction': fraction_energy_assigned_correctly,
                   'f_score': f_score
                   }

# <codecell>

train_time = {}
disaggregate_time = {}
results = {}
predicted_power = {'co': {}, 'fhmm': {}}


DATE_FORMAT = '%H:%M'


frequencies = ['1T']
for freq in frequencies:
    train_time[freq] = {}
    disaggregate_time[freq] = {}
    results[freq] = {}
    for algorithm in ['fhmm', 'co']:
        train_time[freq][algorithm] = {}
        disaggregate_time[freq][algorithm] = {}
        results[freq][algorithm] = {}
    for metric in metrics:
        results[freq][algorithm][metric] = {}


def preprocess_iawe(building, freq):
    building.utility.electric = building.utility.electric.sum_split_supplies()
    building = prepb.filter_out_implausible_values(
        building, Measurement('voltage', ''), 160, 260)
    building = prepb.filter_datetime(building, '7-13-2013', '8-4-2013')
    building = prepb.downsample(building, rule=freq)
    building = prepb.fill_appliance_gaps(building)
    building = prepb.prepend_append_zeros(
        building, '7-13-2013', '8-4-2013', freq, 'Asia/Kolkata')
    building = prepb.drop_missing_mains(building)
    building = prepb.make_common_index(building)
    building = prepb.filter_top_k_appliances(building, k=6)
    return building


def preprocess_redd(building, freq):
    building.utility.electric = building.utility.electric.sum_split_supplies()
    building = prepb.downsample(building, rule=freq)
    building = prepb.fill_appliance_gaps(building)
    building = prepb.drop_missing_mains(building)
    building = prepb.make_common_index(building)
    building.utility.electric.mains[(1, 1)].rename(
        columns={Measurement('power', 'apparent'): Measurement('power', 'active')}, inplace=True)
    building = prepb.filter_top_k_appliances(building, k=6)

    return building


def preprocess_ampds(building, freq):
    building = prepb.downsample(building, rule=freq)
    building = prepb.filter_top_k_appliances(building, k=6)
    return building


def preprocess_pecan(building, freq):
    building = prepb.downsample(building, rule=freq)
    building = prepb.filter_top_k_appliances(building, k=6)
    return building

preprocess_map = {'iawe': preprocess_iawe, 'redd/low_freq': preprocess_redd,
                  'ampds': preprocess_ampds, 'pecan_1min': preprocess_pecan}

# <codecell>

dataset_name = "iawe"
dataset = DataSet()
dataset.load_hdf5(os.path.join(base_path, dataset_name))
print("Loaded {}".format(dataset_name))
for freq in frequencies:
    print("*" * 80)
    print("Loading {}".format(freq))
    building = dataset.buildings[1]
    building = preprocess_map[dataset_name](building, freq)
    print("Number of appliance left = {}".format(
        len(building.utility.electric.appliances.keys())))
    print("Dividing data into test and train")
    train, test = train_test_split(building, train_size=0.5)
    for disaggregator_name, disaggregator in disaggregators.iteritems():
        # Train
        t1 = time.time()
        disaggregator.train(train, disagg_features=[DISAGG_FEATURE])
        t2 = time.time()
        print("Runtime to train for {} = {:.2f} seconds".format(
            disaggregator_name, t2 - t1))
        train_time[freq][disaggregator_name] = t2 - t1

        # Disaggregate
        t1 = time.time()
        disaggregator.disaggregate(test)
        t2 = time.time()
        print("Runtime to disaggregate for {}= {:.2f} seconds".format(
            disaggregator_name, t2 - t1))
        disaggregate_time[freq][disaggregator_name] = t2 - t1

        # Predicted power and states
        predicted_power[disaggregator_name] = disaggregator.predictions
        app_ground = test.utility.electric.appliances
        ground_truth_power = pd.DataFrame({appliance: app_ground[appliance][DISAGG_FEATURE] for appliance in app_ground})
        for metric in metrics:
            results[freq][disaggregator_name][metric] = metric_function[
                metric](predicted_power[disaggregator_name], ground_truth_power)


results = results['1T']
start = pd.Timestamp("2013-07-27 21:35")
end = pd.Timestamp("2013-07-27 22:35")
fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True)
latexify(columns=1)
plot_series(predicted_power['fhmm']
            [('air conditioner', 2)][start:end], ax=axes[0], date_format=DATE_FORMAT)
plot_series(predicted_power['co']
            [('air conditioner', 2)][start:end], ax=axes[1], date_format=DATE_FORMAT)
plot_series(ground_truth_power[('air conditioner', 2)]
            [start:end], ax=axes[2], date_format=DATE_FORMAT)


for ax in axes:
    ax.set_ylim((0, 2000))
    format_axes(ax)


axes[0].set_title("Predicted power\nFHMM")
axes[1].set_title("Predicted power\nCO")
axes[2].set_title("Ground truth power")

axes[0].set_ylabel("Power (W)")
fig.tight_layout()
fig.savefig("/home/nipun/Desktop/ac_2.pdf")
plt.show()
