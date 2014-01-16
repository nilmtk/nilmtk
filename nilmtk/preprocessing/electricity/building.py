"""Contains preprocessing functions."""
from __future__ import print_function, division
import pandas as pd
import numpy as np
import sys

from nilmtk.stats.electricity.building import find_appliances_contribution
from nilmtk.stats.electricity.building import top_k_appliances
from nilmtk.stats.electricity.single import get_sample_period, get_gap_starts_and_gap_ends

from nilmtk.preprocessing.electricity.single import remove_implausible_entries
from nilmtk.preprocessing.electricity.single import filter_datetime_single
from nilmtk.preprocessing.electricity import single

from nilmtk.utils import apply_func_to_values_of_dicts

from copy import deepcopy

# Define all the dicts to which we want to apply functions within Buildings
BUILDING_ELECTRICITY_DICTS = ['utility.electric.appliances',
                              'utility.electric.mains',
                              'utility.electric.circuits']


def filter_contribution_less_than_x(building, x=5):
    """Filters out appliances which contribute less than x%

    Parameters
    ----------
    building : nilmtk.Building
    x : float [0,100], default :5

    Returns
    -------
    building_copy : nilmtk.Building
    """
    electricity = building.utility.electric
    contribution_df = find_appliances_contribution(electricity)
    more_than_x_df = contribution_df[contribution_df > (x * 1.0 / 100)]
    building_copy = deepcopy(building)
    appliances_dict = building.utility.electric.appliances
    appliances_filtered = {appliance_name: appliance_df
                           for appliance_name, appliance_df in appliances_dict.iteritems()
                           if appliance_name in more_than_x_df.keys()}
    building_copy.utility.electric.appliances = appliances_filtered
    return building_copy


def filter_top_k_appliances(building, k=5):
    """Filters and keeps only the top k appliance data

    Parameters
    ----------
    building : nilmtk.Building
    k : int, default: 5
        Top 'k' appliances to keep

    Returns
    -------
    building_copy : nilmtk.Building
    """

    top_k = top_k_appliances(building.utility.electric, k=k).index
    building_copy = deepcopy(building)
    appliances_dict = building.utility.electric.appliances
    appliances_filtered = {appliance_name: appliance_df
                           for appliance_name, appliance_df in appliances_dict.iteritems()
                           if appliance_name in top_k}
    building_copy.utility.electric.appliances = appliances_filtered
    return building_copy


def downsample(building, rule='1T', how='mean', dropna=False):
    """Downsample all electrical data

    Parameters
    ----------
    building : nilmtk.Building
    rule : string
        refer to pandas.resample docs for rules; default '1T' or 1 minute
    how : string
        refer to pandas.resample docs for how; default 'mean'
    dropna : boolean, optional
        default = False.  Whether to drop NaNs after resampling.

    Returns
    --------
    building_copy: nilmtk.Building

    """
    # Define a resample function
    if dropna:
        resample = lambda df: pd.DataFrame.resample(
            df, rule=rule, how=how).dropna()
    else:
        resample = lambda df: pd.DataFrame.resample(df, rule=rule, how=how)

    return apply_func_to_values_of_dicts(building, resample,
                                         BUILDING_ELECTRICITY_DICTS)


def drop_missing_mains(building):
    MAINS = ['utility.electric.mains']
    return apply_func_to_values_of_dicts(
        building, lambda df: df.dropna(),
        MAINS)


def prepend_append_zeros(building, start_datetime, end_datetime, freq, timezone):
    """Fill zeros from `start` to `appliance`.index[0] and from 
    `appliance`.index[-1] to end at `frequency`"""

    # TODO: can this function be merged with or make use of
    # preprocessing.building.single.reframe_index ?

    APPLIANCES = ['utility.electric.appliances']
    idx = pd.DatetimeIndex(start=start_datetime, end=end_datetime, freq=freq)
    idx = idx.tz_localize('GMT').tz_convert(timezone)

    def reindex_fill_na(df):
        df_copy = deepcopy(df)
        df_copy = df_copy.reindex(idx)

        power_columns = [
            x for x in df.columns if x.physical_quantity in ['power']]
        non_power_columns = [x for x in df.columns if x not in power_columns]

        for power in power_columns:
            df_copy[power].fillna(0, inplace=True)
        for measurement in non_power_columns:
            df_copy[measurement].fillna(
                df[measurement].median(), inplace=True)

        return df_copy

    new_building = apply_func_to_values_of_dicts(building, reindex_fill_na,
                                                 APPLIANCES)
    return new_building


def filter_channels_with_less_than_x_samples(building, threshold=100):
    building_copy = deepcopy(building)
    for appliance_name, appliance_df in building.utility.electric.appliances.items():
        print(appliance_name, len(appliance_df.index))
        if len(appliance_df.index) < threshold:
            print(appliance_name)
            building_copy.utility.electric.appliances.pop(appliance_name, None)
    return building_copy


def fill_appliance_gaps(building, sample_period_multiplier=4):
    """Book-ends all large gaps with zeros using
    `nilmtk.preprocessing.electric.single.insert_zeros`
    on all appliances in `building` and then forward fills any remaining NaNs.
    This will result in forward-filling small gaps with
    the recorded value which precedes the gap, and forward-filling zeros
    in large gaps.

    NOTE: This function assumes that any gaps in the appliance data is the
    result of the appliance monitor and the appliance being off.  Do not
    use this function if gaps in appliance data are the result of the
    IAM being broken (and hence the state of the appliance is unknown).

    Parameters
    ----------
    building : nilmtk.Building
    sample_period_multiplier : float or int, optional
        The permissible  maximum sample period expressed as a multiple
        of each dataframe's sample period. Any gap longer
        than the max sample period is assumed to imply that the IAM
        and appliance are off.  If None then will default to
        4 x the sample period of each dataframe.

    Returns
    -------
    building_copy : nilmtk.Building

    See Also
    --------
    nilmtk.preprocessing.electric.single.insert_zeros()
    """

    # TODO: should probably remove any periods where all appliances
    # are not recording (which indicates that things are broken)

    # "book-end" each gap with a zero at each end
    single_insert_zeros = lambda df: single.insert_zeros(
        df, sample_period_multiplier=sample_period_multiplier)

    APPLIANCES = ['utility.electric.appliances']
    new_building = apply_func_to_values_of_dicts(building, single_insert_zeros,
                                                 APPLIANCES)

    # Now fill forward
    ffill = lambda df: pd.DataFrame.fillna(df, method='ffill')
    new_building = apply_func_to_values_of_dicts(new_building, ffill,
                                                 APPLIANCES)

    return new_building


def filter_datetime(building, start_datetime=None, end_datetime=None, copy=False):
    """Filters out all data falling outside the start and the end date

    Parameters
    ----------
    building : nilmtk.Building
    start_datetime :string, 'mm-dd-yyyy hh:mm:ss'
    end_datetime : string, 'mm-mdd-yyyy hh:mm:ss'

    Returns
    -------
    building_copy : nilmtk.Building
    """
    if copy:
        building_copy = deepcopy(building)
    else:
        building_copy = building
    # Filtering appliances
    del_list = []
    for appliance_name, appliance_df in building.utility.electric.appliances.iteritems():
        building_copy.utility.electric.appliances[
            appliance_name] = filter_datetime_single(appliance_df, start_datetime,
                                                     end_datetime)
        if len(building_copy.utility.electric.appliances[
                appliance_name]) == 0:

            del_list.append(appliance_name)
    for appliance_name in del_list:
        print("DELETING {}".format(appliance_name))
        building_copy.utility.electric.appliances.pop(appliance_name)

    # Filtering mains data
    for mains_name, mains_df in building.utility.electric.mains.iteritems():
        building_copy.utility.electric.mains[
            mains_name] = filter_datetime_single(mains_df, start_datetime, end_datetime)
        if len(building_copy.utility.electric.mains[
                mains_name]) == 0:
            print("DELETING {}".format(mains_name))
            del building_copy.utility.electric.mains[
                mains_name]

    return building_copy


def filter_out_implausible_values(building, measurement,
                                  min_threshold=None, max_threshold=None):
    """Filters out values in appliances, circuits and mains which have measurement
    as one of the attribute. Filtering is done by min and max threshold for a
    measurement

    Parameters
    ----------
    building :nilmtk.Building
    measurement :nilmtk.sensor.electricity.Measurement
    min_threshold : float, default=None
    max_threshold :float, deafult=None

    Returns
    -------
    building_copy : nilmtk.Building

    See also
    ---------
    nilmtk.preprocessing.electricity.single.remove_implausible_entries"""

    building_copy = deepcopy(building)
    # Filtering appliances
    for appliance_name, appliance_df in building.utility.electric.appliances.iteritems():
        if measurement in appliance_df.columns:
            building_copy.utility.electric.appliances[
                appliance_name] = remove_implausible_entries(appliance_df, measurement,
                                                             min_threshold, max_threshold)
        else:
            # Copy as it is
            building_copy.utility.electric.appliances[
                appliance_name] = appliance_df

    # Filtering mains data
    for mains_name, mains_df in building.utility.electric.mains.iteritems():
        if measurement in mains_df.columns:
            building_copy.utility.electric.mains[
                mains_name] = remove_implausible_entries(mains_df, measurement,
                                                         min_threshold, max_threshold)
        else:
            building_copy.utility.electric.mains[
                mains_name] = mains_df

    return building_copy


def make_common_index(building):
    building_copy = deepcopy(building)
    appliances_index = building.utility.electric.appliances.values()[0].index
    mains_index = building.utility.electric.mains.values()[0].index
    freq = building.utility.electric.mains.values()[0].index.freq
    # TODO: can the line below be replace with
    # common_index = mains_index & appliances_index
    # This might be a lot faster and as far as I can tell gives the same
    # answer.
    common_index = pd.DatetimeIndex(
        np.sort(list(set(mains_index).intersection(set(appliances_index)))),
        freq=freq)
    take_common_index = lambda df: df.ix[common_index]
    return apply_func_to_values_of_dicts(building, take_common_index,
                                         BUILDING_ELECTRICITY_DICTS)


def mask_appliances_with_mains(electricity, sample_period_multiplier=4):
    """Finds gaps in first mains channel and then removes 
    these gaps from all appliance data. 

    The assumption is that if the mains channel is dead for any
    timeslice then we should ignore this timeslice for all appliance
    channels too.

    Parameters
    ----------
    electricity : Electricity object

    sample_period_multiplier : int, optional
        Default = 4
        max_sample_period = sample_period x sample_period_multiplier
        max_sample_period defines a 'gap'.
    
    Returns
    -------
    copy of electricity
    
    .. warning:: currently only uses gaps from first mains dataframe and ignores
                 all other mains dataframes.

    """

    # TODO: handle multiple mains channels and take intersection of gaps

    print("Masking appliances with mains... may take a little while...", end='')
    sys.stdout.flush()
    mains = electricity.mains.values()[0]
    max_sample_period = get_sample_period(mains) * sample_period_multiplier
    print("Mains sample period = {:.1f}, max_sample_period = {:.1f}"
          .format(get_sample_period(mains), max_sample_period))
    print("Getting gap starts and ends...")
    gap_starts, gap_ends = get_gap_starts_and_gap_ends(mains, max_sample_period)
    print("Found {:d} gap starts and {:d} gap ends.".format(len(gap_starts), len(gap_ends)))

    def mask_appliances(appliance_df):
        """For each appliance dataframe, insert NaNs for any reading inside
        mains gaps.
        """
        print(".", end='')
        sys.stdout.flush()
        for gap_start, gap_end in zip(gap_starts, gap_ends):
            index = appliance_df.index
            try:
                appliance_df[(index >= gap_start) & (index <= gap_end)] = np.NaN
            except ValueError:
                # some DFs are int32, which can't accept NaNs, so convert to float32:
                # TODO: remove this once #105 is fixed
                appliance_df = appliance_df.astype(np.float32)
                appliance_df[(index >= gap_start) & (index <= gap_end)] = np.NaN
        return appliance_df
    
    masked = apply_func_to_values_of_dicts(electricity, mask_appliances, 
                                           ['appliances'])
    print("done")
    return masked
    
