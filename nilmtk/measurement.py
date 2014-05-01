from __future__ import print_function, division
from collections import namedtuple

AC_TYPES = ['active', 'apparent', 'reactive']
# AC is short for 'Alternating Current'


def check_ac_type(ac_type):
    if ac_type not in AC_TYPES:
        msg = "'ac_type' must be one of {}, not '{}'.".format(AC_TYPES, ac_type)
        raise ValueError(msg)


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
