"""Statistics for applying to an entire building"""

from __future__ import print_function, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from collections import OrderedDict
from nilmtk.sensors.electricity import Measurement
from nilmtk.stats.electricity.single import DEFAULT_MAX_DROPOUT_RATE, usage_per_period
import nilmtk.stats.electricity.single as single
from nilmtk.exceptions import NoSuitableMeasurementError, NoCommonMeasurementError
from nilmtk.utils import apply_func_to_values_of_dicts

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
                                    sample_period_multiplier=20):
#                                    require_common_measurements=True):
    """Reports the proportion of energy in a building that is submetered.

    Assumes that missing appliance data means that the appliance is off.

    Parameters
    ----------
    electricity : nilmtk.sensors.electricity.Electricity

    sample_period_multiplier : float, optional
        Defines a 'gap'.  Any gap > sample_period x sample_period_multiplier
        is counted as a gap.  Default = 4.

    require_common_measurements : boolean, optional, default=True
        If True then raise a NoCommonMeasurementsError exception if 
        there is not at least one shared
        Measurement (e.g. ('power', 'active')) across all channels.
        If False then continue even if measurements do not match.

    Returns
    -------
    float
        0 = no energy submetered
        1 = all energy submetered
       >1 = more energy submetered than is recorded on the mains channels!
    """
    # TODO: get common measurements!
    # TODO: Handle circuits.
    # TODO: Handle wiring diagram.

    print("Calculating proportion of energy submetered...")
    total_mains_energy, totals_per_appliance = energy_per_dataframe(
        electricity, sample_period_multiplier)
    return totals_per_appliance.sum() / total_mains_energy


def proportion_per_appliance(electricity, merge=False):
    """
    Parameters
    ----------
    merge : boolean, optional
        If True then merge multiple instances of the same appliance name
        and merge all appliances with 'light' in their name.

    Returns
    -------
    Series, sorted by value
       Keys are ApplianceNames.names if `merge` else ApplianceNames.names
       Values are proportion of total energy expressed as a float
       between 0 and 1.
    """
    total_mains_energy, totals_per_appliance = energy_per_dataframe(
        electricity)
    prop_per_appliance = totals_per_appliance / total_mains_energy

    if merge:
        names = set([appliance.name for appliance in prop_per_appliance.index])

        # Split names into lighting and non-lighting
        names_excl_lights = []
        names_of_lights = []
        for name in names:
            if np.any([light_name in name for light_name in ['light','lamp']]):
                names_of_lights.append(name)
            else:
                names_excl_lights.append(name)

        # Merge non-lighting appliances
        ppa_merged = {}
        for name in names_excl_lights:
            i = 1
            ppa_merged[name] = 0
            while True:
                try:
                    ppa_merged[name] += prop_per_appliance[(name, i)]
                except KeyError:
                    break
                else:
                    i += 1

        # Merge lighting
        ppa_merged['lighting'] = 0
        for name in names_of_lights:
            i = 1
            while True:
                try:
                    ppa_merged['lighting'] += prop_per_appliance[(name, i)]
                except KeyError:
                    break
                else:
                    i += 1

        prop_per_appliance = pd.Series(ppa_merged)

    prop_per_appliance.sort(ascending=False)
    return prop_per_appliance


def energy_per_dataframe(electricity, sample_period_multiplier=20, unit='kwh'):
    """pre-processes electricity and then gets total energy per channel, 
    after masking out all gaps in mains.

    Returns
    -------
    mains_total_energy, totals_per_appliance

    total_mains_energy : float
    totals_per_appliance : pd.Series
        each key is an ApplianceName
        each value is total energy
    """

    # TODO: this might be an ugly hack to resolve circular dependencies.
    from nilmtk.preprocessing.electricity.building import mask_appliances_with_mains
    from nilmtk.preprocessing.electricity.single import insert_zeros

    # remove 'unmetered' and 'subpanels' from appliances
    electricity.appliances = electricity.remove_channels_from_appliances()

    # Sum split mains and DualSupply appliances
    electricity = electricity.sum_split_supplies()

    # TODO: Select common measurements. Maybe use electricity.select_common_measurements?
    # MEASUREMENT_PREFERENCES = [Measurement('power', 'active'),
    #                            Measurement('power', 'apparent')]

    # # Check if all channels share at least one Measurement (e.g. ('power', 'active'))
    # common_measurements = find_common_measurements(electricity)
    # common_measurement = None
    # for measurement_preference in MEASUREMENT_PREFERENCES:
    #     if measurement_preference in common_measurements:
    #         common_measurement = measurement_preference
    #         print("Using common_measurement:", common_measurement)
    #         break
    # if common_measurements is None and require_common_measurements:
    #     raise NoCommonMeasurementError

    # Find large gaps in mains data and ignore those gaps for all appliance channels
    electricity = mask_appliances_with_mains(electricity, 
                                             sample_period_multiplier)

    # Drop NaNs on all channels
    electricity = apply_func_to_values_of_dicts(electricity, 
                                                lambda df: df.dropna(),
                                                ['appliances', 'mains'])

    # Insert_zeros on appliance data.
    print("Inserting zeros... may take a little while...", end='')
    sys.stdout.flush()
    single_insert_zeros = lambda df: insert_zeros(
        df, sample_period_multiplier=sample_period_multiplier)
    electricity = apply_func_to_values_of_dicts(electricity, single_insert_zeros,
                                                ['appliances', 'mains'])
    print("done inserting zeros")

    # Total energy used for mains
    total_mains_energy = get_total_energy_per_dict(electricity, 'mains', unit)

    totals_per_appliance = {}
    for name, df in electricity.appliances.iteritems():
        totals_per_appliance[name] = single.energy(df, unit=unit)
    
    return total_mains_energy, pd.Series(totals_per_appliance)


def get_total_energy_per_dict(electricity, dict_='mains',  unit='kwh'):
    total_energy = 0.0
    for df in electricity.__dict__[dict_].values():
        energy_for_df = single.energy(df, unit)
        total_energy += energy_for_df
    return total_energy


def average_energy(electricity,
                   max_dropout_rate=DEFAULT_MAX_DROPOUT_RATE):
    """
    Returns
    -------
    float
       Average energy usage for this building in kWh per day.

    .. warning:: NOT IMPLEMENTED YET
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

    .. warning:: NOT IMPLEMENTED YET
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
    combined_mains = electricity.mains.values()[0]
    if num_mains > 1:
        for i in xrange(1, num_mains):
            combined_mains += electricity.mains.values()[i]

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


def plot_missing_samples_using_rectangles(electricity, ax=None, fig=None,
                                          color='k'):
    # TODO: docstrings!
    # TODO: better default date format

    n = len(electricity.appliances) + len(electricity.mains)
    ylabels = []
    i = 0
    appliances = electricity.remove_channels_from_appliances()
    for appliance_name, appliance_df in appliances.iteritems():
        ax, fig = single.plot_missing_samples(
            appliance_df, ax, fig, bottom=i + 0.1, color=color)
        ylabels.append((appliance_name.name, appliance_name.instance))
        i += 1

    for mains_name, mains_df in electricity.mains.iteritems():
        ax, fig = single.plot_missing_samples(
            mains_df, ax, fig, bottom=i + 0.1, color=color)
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


def plot_missing_samples_using_bitmap(electricity, ax=None, fig=None,
                                      fig_width=800, add_colorbar=True,
                                      cmap=plt.cm.Blues):
    """
    Parameters
    ----------

    fig_width : int, default=800
        The width of the plotted figure, in pixels
    """
    # TODO: docstring!!!

    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()

    dataset_start, dataset_end = electricity.get_start_and_end_dates()
    sec_per_pixel = (dataset_end - dataset_start).total_seconds() / fig_width
    rule_code = '{:d}S'.format(int(round(sec_per_pixel)))

    missing_samples_per_period = OrderedDict()
    for dict_of_dfs in [electricity.remove_channels_from_appliances(),
                        electricity.circuits,
                        electricity.mains]:
        for name, df in dict_of_dfs.iteritems():
            try:
                name_str = '{} {}'.format(name.name, name.instance)
            except:
                name_str = 'mains {}'.format(name.split)

            missing_samples_per_period[name_str] = (
                single.dropout_rate_per_period(
                    data=df, rule=rule_code,
                    window_start=dataset_start, window_end=dataset_end))

    df = pd.DataFrame(missing_samples_per_period)
    img = np.transpose(df.values)
    start_datenum = mdates.date2num(df.index[0])
    end_datenum = mdates.date2num(df.index[-1])
    im = ax.imshow(img, aspect='auto', interpolation='none', origin='lower',
                   extent=(start_datenum, end_datenum, 0, df.columns.size),
                   cmap=cmap)

    if add_colorbar:
        plt.colorbar(im)

    ax.set_yticks(np.arange(0.5, len(df.columns) + 0.5))
    ax.set_xlim([start_datenum, end_datenum])

    def formatter(x, pos):
        x = int(x)
        return df.columns[x]

    ax.yaxis.set_major_formatter(FuncFormatter(formatter))
    ax.set_title('Proportion of lost samples')
    for item in ax.get_yticklabels():
        item.set_fontsize(8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y',
                                                      tz=df.index.tzinfo))

    fig.autofmt_xdate()
    # Plot horizontal lines separating appliances
    for i in range(1, img.shape[0]):
        ax.plot([start_datenum, end_datenum],
                [i, i], color='grey', linewidth=1)

    return ax


def get_dropout_rates(electricity, ignore_gaps=False):
    """
    Parameters
    ----------
    electricity : Electricity object
    ignore_gaps : boolean

    Returns
    -------
    list of dropout rates, one per dataframe
    """
    print("Calculating dropout rates...")
    dropout_func = (single.get_dropout_rate_ignore_gaps if ignore_gaps 
                    else single.get_dropout_rate)
    dropout_rates = []
    for attribute in ['appliances', 'circuits', 'mains']:
        for name, df in electricity.__dict__[attribute].iteritems():
            print(name)
            try:
                dropout_rates.append(dropout_func(df.index))
            except:
                print("Error occurred when processing attribute={}, name={}"
                      .format(attribute, name), file=sys.stderr)
                raise
    print("done calculating dropout rates")
    return dropout_rates
    

def proportion_of_time_where_more_energy_submetered(building, 
                                                    min_proportion_submetered=0.7,
                                                    require_common_indicies=True):
    """Report the proportion of time slices where the sum of all the appliances
    submetered is greater than the mains * `min_proportion_submetered`

    Parameters
    ----------
    building : Building
    min_proportion_submetered : float [0,1], optional
        default = 0.7
    require_common_indicies : boolean, optional
        default = True.  Decides what to use for the 'total duration' when 
        calculating the proportion of time.  If False then use the total
        duration between the first and last samples. If True then only
        use the non-NaN timeslices after finding the intersection of the
        appliance and mains indicies.

    Returns
    -------
    float [0,1] proportion of time
    """

    # mask appliance with mains & remove large gaps in mains
    # (take a look at proportion_of_time_where_more_energy_submetered)
    # put mains and appliances into one big DF (make sure we take the correct appliances!)
    # downsample to 10 minute chunks, using mean
    # 
    # OR....
    # ignore large gaps in mains... then...
    # chop into 10 min chunks and pass these to proportion_of_energy_submetered.
    # maybe can do this efficiently by putting everything into a big dataframe
    # and using indicies_of_periods, and then extracting these chunks back
    # into an Electricity object.

    building.utility.electric = building.utility.electric.sum_split_supplies()
    
    import nilmtk.preprocessing.electricity.building as prepb

    # downsample mains, circuits and appliances
    b_downsampled = prepb.downsample(building, '1T')
    
    electric = b_downsampled.utility.electric
    appliance_df = electric.get_dataframe_of_appliances().dropna()
    mains_df = electric.get_dataframe_of_mains().dropna()
    common_index = mains_df.index & appliance_df.index
    appliance_df = appliance_df.ix[common_index]
    mains_df = mains_df.ix[common_index]
    appliances_summed = appliance_df.sum(axis=1)
    timeslices_above_thresh = appliances_summed > (mains_df * min_proportion_submetered)

    mins_above_thresh = timeslices_above_thresh.sum().values[0]
    secs_above_thresh = mins_above_thresh * 60
    
    # Calc total duration
    if require_common_indicies:
        total_duration_secs = len(common_index) * 60
    else:
        start, end = building.utility.electric.get_start_and_end_dates()
        total_duration_secs = (end - start).total_seconds()

    return secs_above_thresh / total_duration_secs
