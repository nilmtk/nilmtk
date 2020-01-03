import numpy as np
import gc
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

    # Select a column based on ordered preferences
    PHYSICAL_QUANTITY_PREFS = ["cumulative energy", "energy", "power"]
    selected_columns = []
    for ac_type in AC_TYPES:
        physical_quantities = [physical_quantity 
                               for (physical_quantity, col_ac_type) in df.keys()
                               if col_ac_type == ac_type]
        for pq in PHYSICAL_QUANTITY_PREFS:
            if pq in physical_quantities:
                selected_columns.append((pq, ac_type))
                break

    energy = {}
    for col in selected_columns:
        (physical_quantity, ac_type) = col
        series = df[col]
        if physical_quantity == 'power':
            energy[ac_type] = _energy_for_power_series(series, max_sample_period)
        elif physical_quantity == 'cumulative energy':
            energy[ac_type] = series.iloc[-1] - series.iloc[0]
        elif physical_quantity == 'energy':
            energy[ac_type] = series.sum()

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
    del timedelta
    gc.collect()
    timedelta_secs = timedelta_secs.clip(max=max_sample_period)
    joules = (timedelta_secs * series.values[:-1]).sum()
    kwh = joules / JOULES_PER_KWH
    return kwh
