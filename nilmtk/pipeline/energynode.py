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
        for ac_type in AC_TYPES:
            energy_measurement = Energy(ac_type)
            power_measurement = Power(ac_type)
            if energy_measurement in df.columns:
                energy[ac_type] = df[energy_measurement].sum()
            elif power_measurement in df.columns:
                energy[ac_type] = _energy_per_power_series(
                    df[power_measurement])

        energy_results.append(df.timeframe, **energy)
        df.results[self.name] = energy_results
        return df

    def required_measurements(self, state):
        """EnergyNode needs all power and energy measurements."""
        available_measurements = state['device']['measurements']
        return [measurement for measurement in available_measurements 
                if isinstance(measurement, (Power, Energy))]
