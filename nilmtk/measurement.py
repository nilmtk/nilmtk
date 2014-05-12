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

def select_best_ac_type(results, mains=None, physical_quantity='power'):
    """Selects the 'best' alternating current measurement type from `results`.

    Parameters
    ----------
    results : dict or pandas.DataFrame
        keys must be measurement objects (Power, Energy, Voltage)
    mains : nilmtk.Mains object, optional
        if provided then will try to select the best AC type from `results`
        which is also in `mains`.
    physical_quantity : {'power', 'energy'}

    Returns
    -------
    `results[ac_type]` where `ac_type` is the selected AC type.
    """
    physical_quantity_map = {'power': Power, 'energy': Energy}
    try:
        phys_quantity_class = physical_quantity_map[physical_quantity]
    except KeyError:
        raise ValueError("'{}' is not a recognised physical quantity."
                         .format(physical_quantity))
    order_of_preference = [phys_quantity_class(ac_type) for ac_type in AC_TYPES]
    if mains is not None:
        order_of_preference = [ac_type for ac_type in order_of_preference
                               if ac_type in mains.available_measurements()]

    for ac_type in order_of_preference:
        if ac_type in results:
            return results[ac_type]

    # if we get to here then we haven't found any relevant ac_type in results
    if mains is None:
        raise KeyError()
    else:
        warn("None of the AC types recorded by Mains are present in `results`."
             " Will use try using one of {}.".format(AC_TYPES))
        return select_best_ac_type(results)

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
