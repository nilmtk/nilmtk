""" Contains preprocessing modules"""

from nilmtk.stats.electricity.building import top_k_appliances
from copy import deepcopy


def filter_top_k_appliances(building, k=5):
    top_k = top_k_appliances(building.utility.electric, k=k).index
    building_copy = deepcopy(building)
    appliances_dict = building.utility.electric.appliances
    appliances_filtered = {appliance: appliances_dict[appliance] for appliance in appliances_dict if appliance in top_k}
    building_copy.utility.electric.appliances = appliances_filtered
    return building_copy
