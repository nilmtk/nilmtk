"""Statistics for applying to an entire building"""

from __future__ import print_function, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from nilmtk.sensors.electricity import Measurement
from single import DEFAULT_MAX_DROPOUT_RATE, usage_per_period
import single

def find_common_measurements(electricity):
    """Finds common measurement contained in all electricity streams

    Parameters
    ----------
    electricity : nilmtk.sensors.electricity

    Returns
    -------
    list of common measurements
    """

    # Measurements in first mains
    measurements = set(electricity.mains.values()[0].columns)

    # Finding intersection with other mains
    for main in electricity.mains.itervalues():
        measurements = measurements.intersection(
            main.columns)

    # Finding intersection with appliances
    for appliance in electricity.appliances.itervalues():
        measurements = measurements.intersection(
            appliance.columns)
    return list(measurements)


def proportion_of_energy_submetered(electricity,
                                    max_dropout_rate=DEFAULT_MAX_DROPOUT_RATE,
                                    require_matched_measurements=True):
    """Reports the proportion of energy in a building that is submetered.


    Parameters
    ----------
    electricity : nilmtk.sensors.electricity.Electricity

    max_dropout_rate : float [0,1], optional

    require_matched_measurements : boolean, optional, default=True
        If True then raise an exception if there is not at least one shared
        Measurement (e.g. ('power', 'active')) across all channels.
        If False then continue even if measurements do not match.

    Returns
    -------
    float
        0 = no energy submetered
        1 = all energy submetered
       >1 = more energy submetered than is recorded on the mains channels!
    """

    # TODO: Handle circuits.
    # TODO: Check if all channels share at least one Measurement (e.g. ('power', 'active'))
    #       and handle `require_matched_measurements`
    # TODO: handle dataframes with more than one column (don't use df.icol(0))

    # for each channel, find set of 'good_days' where dropout_rate <
    # max_dropout_rate
    good_days_list = []

    def get_kwh_per_day_per_chan(dictionary):
        """Helper function.  Returns a list of pd.Series of kWh per day."""
        chan_kwh_per_day = []
        for label, df in dictionary.iteritems():
            kwh_per_day = usage_per_period(df.icol(0), freq='D',
                                           max_dropout_rate=max_dropout_rate)['kwh']
            kwh_per_day = kwh_per_day.dropna()
            print(kwh_per_day)
            chan_kwh_per_day.append(kwh_per_day)
            good_days_list.append(set(kwh_per_day.index))
        return chan_kwh_per_day

    mains_kwh_per_day = get_kwh_per_day_per_chan(electricity.mains)
    appliances_kwh_per_day = get_kwh_per_day_per_chan(electricity.appliances)

    # find intersection of all these sets (i.e. find all good days in common)
    good_days_set = good_days_list[0]
    for good_days in good_days_list[1:]:
        good_days_set = good_days_set.intersection(good_days)

    # for each day in intersection, get kWh
    proportion_per_day = []
    for good_day in good_days_set:
        mains_kwh = 0
        for kwh_per_day in mains_kwh_per_day:
            mains_kwh += kwh_per_day[good_day]

        appliances_kwh = 0
        for kwh_per_day in appliances_kwh_per_day:
            appliances_kwh += kwh_per_day[good_day]

        proportion = appliances_kwh / mains_kwh
        proportion_per_day.append(proportion)

    return np.mean(proportion_per_day)


def start_end_datetimes_channels(electricity):
    """Finds the start and end datetimes of all electricity channels. 
    It can be used to find common start and end times for different 
    channels to be used in preprocessing filters

    Parameters
    ----------
    electricity : nilmtk.utility.electricity

    Returns
    -------
    start_end_datetimes_dict : dictionary  
        {channel: [start_timestamp, end_timestamp]}
    """
    # also see nilmtk.sensors.electricity.Electricity.get_start_and_end_dates()
    raise NotImplementedError


def average_energy(electricity,
                   max_dropout_rate=DEFAULT_MAX_DROPOUT_RATE):
    """
    Returns
    -------
    float
       Average energy usage for this building in kWh per day.
    """
    raise NotImplementedError


def average_energy_per_appliance(electricity,
                                 max_dropout_rate=DEFAULT_MAX_DROPOUT_RATE):
    """Reports the average energy consumed by each appliance.

    For each appliance, we ignore any days which have a dropout rate
    above `max_dropout_rate`.

    Parameters
    ----------
    electricity:
        nilmtk.sensors.electricity.Electricity

    Returns
    -------
    av_energy:
        pd.Series
        Each element of the index is an ApplianceName
        Values are average energy in kWh per day
    """
    raise NotImplementedError


def find_appliances_contribution(electricity, how=np.mean):
    """Reports dataframe of form (appliance : contribution) type

    Parameters
    ----------
    electricity : nilmtk.sensors.Elictricity

    Returns
    -------
    series_contribution: pandas.DataFrame
    """
    # Finding number of mains
    num_mains = len(electricity.mains.keys())

    # If more than 1 mains exists, add them up
    combined_mains = electricity.mains[electricity.mains.keys()[0]]
    if num_mains > 1:
        for i in xrange(1, num_mains):
            combined_mains += electricity.mains.keys()[electricity.mains.keys()[i]]

    # Finding common measurements
    common_measurements = find_common_measurements(electricity)
    if len(common_measurements) == 0:
        raise Exception('Cannot proceed further; no common attribute')

    if Measurement('power', 'active') in common_measurements:
        common_measurement = Measurement('power', 'active')
    else:
        # Choose the first attribute for comparison
        common_measurement = common_measurements[0]

    print("Common Measurement: ", common_measurement)

    # Applying function over all appliances
    series_appliances = {}
    for appliance in electricity.appliances:
        series_appliances[appliance] = electricity.appliances[
            appliance][common_measurement].mean()

    series_appliances = pd.Series(series_appliances)

    # Applying function over all mains summed up
    series_mains = combined_mains[common_measurement].mean()

    # Contribution per appliance
    series_appliances_contribution = series_appliances / series_mains

    return series_appliances_contribution


def top_k_appliances(electricity, k=3, how=np.mean, order='desc'):
    """Reports the top k appliances by 'how' attribute

    Parameters
    ----------
    electricity:
        nilmtk.sensors.electricity.Electricity
    k:
        Number of results to be returned, int
        Default value:
            3
    how:
        Function by which to order top k appliances
        Default:
            numpy.mean
    order:
        Order whether top k from highest(desc) or from lowest(asc)

    Returns
    -------
    top_k:
        pd.Series
        appliance:
            contribution

    # TODO: Allow arbitrary functions
    # TODO: Handle case when number of appliances is less than default k=3
    """
    series_appliances_contribution = find_appliances_contribution(electricity)

    if order == 'asc':
        # Sorting
        series_appliances_contribution.sort()
    else:
        series_appliances_contribution.sort(ascending=False)

    return series_appliances_contribution.head(k)


def _plot_missing_sample_using_rectanges(electricity, ax=None, fig=None):
    # TODO: docstrings!
    # TODO: better default date format

    n = len(electricity.appliances) + len(electricity.mains)
    colours = [plt.cm.Blues(c) for c in np.linspace(0.3, 0.9, n)]
    ylabels = []
    i = 0
    for appliance_name, appliance_df in electricity.appliances.iteritems():
        ax, fig = single.plot_missing_samples(
            appliance_df, ax, fig, bottom=i + 0.1, color=colours[i])
        ylabels.append((appliance_name.name, appliance_name.instance))
        i += 1

    for mains_name, mains_df in electricity.mains.iteritems():
        ax, fig = single.plot_missing_samples(
            mains_df, ax, fig, bottom=i + 0.1, color=colours[i])
        ylabels.append(('mains', mains_name.split, mains_name.meter))
        i += 1

    i -= 1

    ax.set_yticks(np.arange(0.5, i + 1.5))
    ax.set_xlim(electricity.get_start_and_end_dates())

    def formatter(x, pos):
        x = int(x)
        return ylabels[x]

    ax.yaxis.set_major_formatter(FuncFormatter(formatter))
    for item in ax.get_yticklabels():
        item.set_fontsize(10)


def _plot_missing_sample_using_bitmap(electricity, ax=None, fig=None):
    # Alternative strategy for doing this as a bitmap image:
    # find start and end dates of whole dataset (write a sep method for this)
    # resolution = how much time each pixel on the plot represents.
    # for each channel:
    #    implement stats.location_of_missing_samples(). Returns a Series where
    #    index is the datetime of each missing sample and all values are True
    #    Then just resample(resolution, how='sum').  This should give us what we need to plot
    #
    raise NotImplementedError

def plot_missing_samples(electricity, ax=None, fig=None, how='rectangles'):
    """
    Arguments
    ---------
    how : {'bitmap', 'rectangles'}
    """
    # TODO: docstring!

    map_how_to_functions = {
        'bitmap': _plot_missing_sample_using_bitmap,
        'rectangles': _plot_missing_sample_using_rectanges
    }

    return map_how_to_functions[how](electricity, ax, fig)
