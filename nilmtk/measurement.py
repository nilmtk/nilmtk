from warnings import warn
import pandas as pd
from nilmtk.exceptions import MeasurementError

AC_TYPES = ['active', 'apparent', 'reactive']
# AC is short for 'Alternating Current'.
# AC_TYPES is in order of preference (favourite first).
# Note that 'pf' is deprecated. Please use 'power factor'.  See issue #289.
PHYSICAL_QUANTITIES = ['power', 'energy', 'cumulative energy', 
                       'voltage', 'current', 'pf', 'frequency', 'power factor', 
                       'state', 'phase angle']
PHYSICAL_QUANTITIES_WITH_AC_TYPES = ['power', 'energy', 'cumulative energy']
PHYSICAL_QUANTITIES_TO_AVERAGE = ['voltage', 'pf', 'frequency', 'power factor','phase_angle']
LEVEL_NAMES = ['physical_quantity', 'type']

def check_ac_type(ac_type):
    if ac_type not in AC_TYPES:
        msg = ("'{}' is not a valid AC type."
               " Valid AC types = {}.".format(ac_type, AC_TYPES))
        raise MeasurementError(msg)

def check_physical_quantity(physical_quantity):
    if physical_quantity not in PHYSICAL_QUANTITIES:
        msg = ("'{}' is not a valid physical quantity."
               " Valid physical quantities = {}."
               .format(physical_quantity, PHYSICAL_QUANTITIES))
        raise MeasurementError(msg)

def select_best_ac_type(available_ac_types, mains_ac_types=None):
    """Selects the 'best' alternating current measurement type from 
    `available_ac_types`.

    Parameters
    ----------
    available_ac_types : list of strings
        e.g. ['active', 'reactive']
    mains_ac_types : list of strings, optional
        if provided then will try to select the best AC type from `available_ac_types`
        which is also in `mains_ac_types`.
        If none of the measurements from `mains_ac_types` are 
        available then will raise a warning and will select another ac type.

    Returns
    -------
    best_ac_type : string
    """

    if mains_ac_types is None:
        order_of_preference = AC_TYPES
    else:
        order_of_preference = [ac_type for ac_type in AC_TYPES
                               if ac_type in mains_ac_types]

    for ac_type in order_of_preference:
        if ac_type in available_ac_types:
            return ac_type

    # if we get to here then we haven't found any relevant ac_type in available_ac_types
    if mains_ac_types is None:
        raise KeyError('No relevant measurements in {}'.format(available_ac_types))
    else:
        warn("None of the AC types recorded by Mains are present in `available_ac_types`."
             " Will use try using one of {}.".format(AC_TYPES), RuntimeWarning)
        return select_best_ac_type(available_ac_types)


def measurement_columns(column_tuples):
    """
    Parameters
    ----------
    column_tuples : list of 2-tuples
    
    Returns
    -------
    pd.MultiIndex
    """
    for physical_quantity, ac_type in column_tuples:
        check_physical_quantity(physical_quantity)
        if physical_quantity in ['energy', 'cumulative energy', 'power']:
            check_ac_type(ac_type)
    return pd.MultiIndex.from_tuples(column_tuples, names=LEVEL_NAMES)
