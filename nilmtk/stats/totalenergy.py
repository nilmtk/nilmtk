from __future__ import print_function, division
import numpy as np
from .totalenergyresults import TotalEnergyResults
from ..node import Node
from ..utils import timedelta64_to_secs
from ..consts import JOULES_PER_KWH
from ..measurement import AC_TYPES
from ..timeframe import TimeFrame


class TotalEnergy(Node):

    requirements = {'device': {'max_sample_period': 'ANY VALUE'},
                    'preprocessing_applied': {'clip': 'ANY VALUE'}}
    postconditions =  {'statistics': {'energy': {}}}
    results_class = TotalEnergyResults

    def process(self):
        """
        Preference: Cumulative energy > Energy > Power
        """
        self.check_requirements()
        metadata = self.upstream.get_metadata()
        max_sample_period = metadata['device']['max_sample_period']
        for chunk in self.upstream.process():
            energy = get_total_energy(chunk, max_sample_period)
            self.results.append(chunk.timeframe, energy)
            yield chunk

    def required_measurements(self, state):
        """TotalEnergy needs all power and energy measurements."""
        available_measurements = state['device']['measurements']
        return [(measurement['physical_quantity'], measurement['type']) 
                for measurement in available_measurements 
                if measurement['physical_quantity'] in 
                ['power', 'energy', 'cumulative energy']]


def get_total_energy(df, max_sample_period):
    """Calculate total energy for energy / power data in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
    max_sample_period : float or int

    Returns
    -------
    energy : dict
        With a key for each AC type (reactive, apparent, active) in `df`.
        Values are energy in kWh (or equivalent for reactive and apparent power).
    """

    energy = {}
    data_source_rank = {} # overwrite Power with Energy with Energy(cumulative)
    for (physical_quantity, ac_type), series in df.iteritems():
        if physical_quantity == 'power':
            # Preference is to calculate energy from 
            # native Energy data rather than Power data
            # so don't overwrite with Power data.
            if not energy.has_key(ac_type):
                energy[ac_type] = _energy_for_power_series(
                    series, max_sample_period)
                data_source_rank[ac_type] = 3 # least favourite
        elif physical_quantity == 'cumulative energy':
            energy[ac_type] = series.iloc[-1] - series.iloc[0]
            data_source_rank[ac_type] = 1 # favourite
        elif (physical_quantity == 'energy' and 
              data_source_rank.get(ac_type, 3) > 2):
            energy[ac_type] = series.sum()
            data_source_rank[ac_type] = 2
    return energy


def _energy_for_power_series(series, max_sample_period):
    """
    Parameters
    ----------
    series : pd.Series
    max_sample_period : float or int

    Returns
    -------
    energy : float
        kWh
    """
    series = series.dropna()
    timedelta = np.diff(series.index.values)
    timedelta_secs = timedelta64_to_secs(timedelta)
    timedelta_secs = timedelta_secs.clip(max=max_sample_period)
    joules = (timedelta_secs * series.values[:-1]).sum()
    kwh = joules / JOULES_PER_KWH
    return kwh
