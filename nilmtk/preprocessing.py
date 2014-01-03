""" Contains preprocessing modules"""
import pandas as pd

from nilmtk.stats.electricity.building import top_k_appliances
from copy import deepcopy


def filter_top_k_appliances(building, k=5):
    top_k = top_k_appliances(building.utility.electric, k=k).index
    building_copy = deepcopy(building)
    appliances_dict = building.utility.electric.appliances
    appliances_filtered = {appliance: appliances_dict[appliance] for appliance in appliances_dict if appliance in top_k}
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
