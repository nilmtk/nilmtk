"""Contains preprocessing functions."""
import pandas as pd
import numpy as np

from nilmtk.stats.electricity.building import find_appliances_contribution
from nilmtk.stats.electricity.building import top_k_appliances

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
                           if appliance_name in more_than_x_df.columns}
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


def prepend_append_zeros(building, start_datetime, end_datetime, freq, timezone):
    """Fill zeros from `start` to `appliance`.index[0] and from 
    `appliance`.index[-1] to end at `frequency`"""
    APPLIANCES = ['utility.electric.appliances']
    idx = pd.DatetimeIndex(start=start_datetime, end=end_datetime, freq=freq)
    idx = idx.tz_localize('GMT').tz_convert(timezone)

    def reindex_fill_na(df):
        df_copy = deepcopy(df)
        df_copy.reindex(idx)

        power_columns = [
            x for x in df.columns if x.physical_quantity in ['power']]
        non_power_columns = [x for x in df.columns if x not in power_columns]
        df_copy[power_columns].fillna(0, inplace=True)
        for measurement in non_power_columns:
            df_copy[measurement].fillna(df[measurement].median(), inplace=True)
        print(df_copy.index)
        print(df_copy.describe())
        return df_copy

    new_building = apply_func_to_values_of_dicts(building, reindex_fill_na,
                                                 APPLIANCES)
    return new_building



def fill_appliance_gaps(building, sample_period_multiplier=4):
    """Book-ends all large gaps with zeros using
    `nilmtk.preprocessing.electric.single.insert_zeros`
    and all appliances in `building` and then forward fills any remaining NaNs.
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
    single_insert_zeros = lambda df: single.insert_zeros(df,
                                                         sample_period_multiplier=sample_period_multiplier)

    APPLIANCES = ['utility.electric.appliances']
    new_building = apply_func_to_values_of_dicts(building, single_insert_zeros,
                                                 APPLIANCES)

    # Now fill forward
    ffill = lambda df: pd.DataFrame.fillna(df, method='ffill')
    new_building = apply_func_to_values_of_dicts(new_building, ffill,
                                                 APPLIANCES)

    return new_building


def filter_datetime(building, start_datetime=None, end_datetime=None):
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
    building_copy = deepcopy(building)
    # Filtering appliances
    for appliance_name, appliance_df in building.utility.electric.appliances.iteritems():
        building_copy.utility.electric.appliances[
            appliance_name] = filter_datetime_single(appliance_df, start_datetime,
                                                     end_datetime)
        if len(building_copy.utility.electric.appliances[
                appliance_name]) == 0:
            print "DELETING ", appliance_name
            del building_copy.utility.electric.appliances[
                appliance_name]

    # Filtering mains data
    for mains_name, mains_df in building.utility.electric.mains.iteritems():
        building_copy.utility.electric.mains[
            mains_name] = filter_datetime_single(mains_df, start_datetime, end_datetime)
        if len(building_copy.utility.electric.mains[
                mains_name]) == 0:
            print "DELETING", mains_name
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
    print freq
    common_index = pd.DatetimeIndex(
        np.sort(list(set(mains_index).intersection(set(appliances_index)))),
        freq=freq)
    take_common_index = lambda df: df.ix[common_index]
    return apply_func_to_values_of_dicts(building, take_common_index,
                                         BUILDING_ELECTRICITY_DICTS)
