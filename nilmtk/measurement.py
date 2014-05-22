from __future__ import print_function, division
from collections import namedtuple
from warnings import warn

AC_TYPES = ['active', 'apparent', 'reactive']
# AC is short for 'Alternating Current'.
# AC_TYPES is in order of preference (favourite first).

def check_ac_type(ac_type):
    if ac_type not in AC_TYPES:
        msg = "'ac_type' must be one of {}, not '{}'.".format(AC_TYPES, ac_type)
        raise ValueError(msg)

def select_best_ac_type(available_ac_types, mains_ac_types=None):
    """Selects the 'best' alternating current measurement type from `available_ac_types`.

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
             " Will use try using one of {}.".format(AC_TYPES))
        return select_best_ac_type(available_ac_types)

class Power(namedtuple('Power', ['ac_type'])):
    """Mains electricity alternating current (AC) power demand.
    
    Attributes
    ----------
    ac_type : {'active', 'apparent', 'reactive'}
        Type of Alternating Current (AC) measurement.
    """
    def __new__(cls, ac_type):
        check_ac_type(ac_type)
        return super(Power, cls).__new__(cls, ac_type)


class Energy(namedtuple('Energy', ['ac_type', 'cumulative'])):
    """Mains electricity alternating current (AC) energy consumption.
    
    Attributes
    ----------
    ac_type : {'active', 'apparent', 'reactive'}
        Type of Alternating Current (AC) measurement.

    cumulative : bool
    """
    def __new__(cls, ac_type, cumulative=False):
        check_ac_type(ac_type)
        if not isinstance(cumulative, bool):
            msg = "'cumulative' must be a bool."
            raise TypeError(msg)
        return super(Energy, cls).__new__(cls, ac_type, cumulative)


class Voltage(namedtuple('Voltage', [])):
    """Mains electricity alternating current (AC) voltage."""
    def __new__(cls):
        return super(Voltage, cls).__new__(cls)
