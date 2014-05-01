from __future__ import print_function, division
from node import Node
from energyresults import EnergyResults
import numpy as np
from nilmtk.utils import timedelta64_to_secs
from nilmtk.consts import JOULES_PER_KWH
from nilmtk.measurement import AC_TYPES
from nilmtk import TimeFrame
from nilmtk.measurement import Power, Energy

def _energy_per_power_series(series):
    timedelta = np.diff(series.index.values)
    timedelta_secs = timedelta64_to_secs(timedelta)
    joules = (timedelta_secs * series.values[:-1]).sum()
    return joules / JOULES_PER_KWH

class EnergyNode(Node):

    requirements = {'preprocessing': {'gaps_bookended_with_zeros': True}}
    postconditions =  {'preprocessing': {'energy_computed': True}}

    def __init__(self, name='energy'):
        super(EnergyNode, self).__init__(name)

    def process(self, df, metadata):
        energy_results = EnergyResults()
        df.results = getattr(df, 'results', {})

        energy = {}
        for measurement, series in df.iteritems():
            if isinstance(measurement, Power):
                _energy = _energy_per_power_series(series)
            elif isinstance(measurement, Energy):
                if measurement.cumulative:
                    _energy = series.iloc[-1] - series.iloc[0]
                else:
                    _energy = series.sum()
            else:
                continue
            energy[measurement.ac_type] = _energy

        energy_results.append(df.timeframe, energy)
        df.results[self.name] = energy_results
        return df

    def required_measurements(self, state):
        """EnergyNode needs all power and energy measurements."""
        available_measurements = state['device']['measurements']
        return [measurement for measurement in available_measurements 
                if isinstance(measurement, (Power, Energy))]
