"""Contains preprocessing functions."""
import pandas as pd
import numpy as np

from nilmtk.stats.electricity.building import find_appliances_contribution
from nilmtk.stats.electricity.building import top_k_appliances

from nilmtk.preprocessing.electricity.single import remove_implausible_entries
from nilmtk.preprocessing.electricity.single import filter_datetime_single
from nilmtk.preprocessing.electricity.single import insert_zeros

from copy import deepcopy


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


def downsample(building, rule='1T', how='mean'):
    """Downsample all electrical data

    Parameters
    ----------
    building : nilmtk.Building
    rule : string
        refer to pandas.resample docs for rules; default '1T' or 1 minute
    how : string
        refer to pandas.resample docs for how; default 'mean'

    Returns
    --------
    building_copy: nilmtk.Building

    """
    building_copy = deepcopy(building)

    # Downsampling appliance data
    for appliance_name, appliance_df in building.utility.electric.appliances.iteritems():
        building_copy.utility.electric.appliances[
            appliance_name] = appliance_df.resample(rule, how).dropna()

    # Downsampling mains data
    for mains_name, mains_df in building.utility.electric.mains.iteritems():
        building_copy.utility.electric.mains[
            mains_name] = mains_df.resample(rule, how).dropna()

    return building_copy


def add_zeros(building):
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
            appliance_name] = insert_zeros(appliance_df)

    # Filtering mains data
    for mains_name, mains_df in building.utility.electric.mains.iteritems():
        building_copy.utility.electric.mains[
            mains_name] = insert_zeros(mains_df)

    return building_copy


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
            del building_copy.utility.electric.appliances[
                appliance_name]

    # Filtering mains data
    for mains_name, mains_df in building.utility.electric.mains.iteritems():
        building_copy.utility.electric.mains[
            mains_name] = filter_datetime_single(mains_df, start_datetime, end_datetime)
        if len(building_copy.utility.electric.mains[
                mains_name]) == 0:
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
