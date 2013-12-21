"""Statistics for applying to an entire building"""

from __future__ import print_function, division
from single import DEFAULT_MAX_DROPOUT_RATE, usage_per_period
import numpy as np



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

    # for each channel, find set of 'good_days' where dropout_rate < max_dropout_rate
    good_days_list = []

    def get_kwh_per_day_per_chan(dictionary):
        """Helper function.  Returns a list of pd.Series of kWh per day."""
        chan_kwh_per_day = []
        for label, df in dictionary.iteritems():
            kwh_per_day = usage_per_period(df.icol(0), freq='D',
                                           max_dropout_rate=max_dropout_rate)['kwh']
            kwh_per_day = kwh_per_day.dropna()
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
    electricity : nilmtk.sensors.electricity.Electricity

    Returns
    -------
    av_energy : pd.Series
        Each element of the index is an ApplianceName
        Values are average energy in kWh per day
    """
    raise NotImplementedError

