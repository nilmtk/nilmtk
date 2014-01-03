""" Contains preprocessing modules"""
import pandas as pd
import numpy as np

from nilmtk.stats.electricity.building import
from nilmtk.stats.electricity.building import top_k_appliances
from copy import deepcopy


def filter_contribution_less_than_x(building, x=5):
    """Filters out appliances which contribute less than x%

    Parameters
    ----------
    building : nilmtk.Building
    x : float, default :5

    Returns
    -------
    building_copy : nilmtk.Building
    """
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
            appliance_name] = appliance_df.resample(rule, how)

    # Downsampling mains data
    for mains_name, mains_df in building.utility.electric.mains.iteritems():
        building_copy.utility.electric.mains[
            mains_name] = mains_df.resample(rule, how)

    return building_copy


def insert_zeros(single_appliance_dataframe, max_sample_period=None):
    """Some individual appliance monitors (IAM) get turned off occasionally.
    This might happen, for example, in the case where a hoover's IAM is 
    permanently attached to the hoover's power cord, even when the hoover is
    put away in the cupboard.

    Say the hoover was switched on and then both the hoover and the hoover's IAM
    were unplugged.  This would result in the dataset having a gap immediately
    after the on-segment.  This combination of an on-segment followed (without
    any zeros) by a gap might confuse downstream statistics and
    disaggregation functions.

    If, after any reading > 0, there is a gap in the dataset of more than 
    `max_sample_period` seconds then assume the appliance (and 
    individual appliance monitor) have been turned off from the
    mains and hence insert a zero max_sample_period seconds after 
    the last sample of the on-segment.

    TODO: a smarter version of this function might use information from
    the aggregate data to do a better job of estimating exactly when
    the appliance was turned off.

    Parameters
    ----------
    single_appliance_dataframe : pandas.DataFrame
        A DataFrame storing data from a single appliance

    max_sample_period : float or int, optional
        The maximum sample permissible period (in seconds). Any gap longer
        than `max_sample_period` is assumed to mean that the IAM 
        and appliance are off.  If None then will default to
        3 x the sample period of `single_appliance_dataframe`.

    Returns
    -------
    df_with_zeros : pandas.DataFrame
        A copy of `single_appliance_dataframe` with zeros inserted 
        `max_sample_period` seconds after the last sample of the on-segment.
    
    """
    raise NotImplementedError


def replace_nans_with_zeros(multiple_appliances_dataframe, max_sample_period):
    """For a single column, find gaps of > max_sample_period and replace
    all NaNs in these gaps with zeros.  But leave NaNs is gaps <=
    max_sample_period."""
    raise NotImplementedError
