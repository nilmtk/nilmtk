import pandas as pd
import numpy as np
from collections import Counter
from warnings import warn
from .timeframe import TimeFrame
from .measurement import select_best_ac_type
from nilmtk.utils import offset_alias_to_seconds

class Electric(object):
    """Common implementations of methods shared by ElecMeter and MeterGroup.
    """
    
    def when_on(self, **load_kwargs):
        """Are the connected appliances appliance is on (True) or off (False)?

        Uses `self.min_on_power_threshold()` if `on_power_threshold` not provided.

        Parameters
        ----------
        on_power_threshold : number, optional
        **load_kwargs : key word arguments
            Passed to self.power_series()

        Returns
        -------
        generator of pd.Series
            index is the same as for chunk returned by `self.power_series()`
            values are booleans
        """
        on_power_threshold = load_kwargs.pop('on_power_threshold', 
                                             self.min_on_power_threshold())
        for chunk in self.power_series(**load_kwargs):
            yield chunk > on_power_threshold

            
    def min_on_power_threshold(self):
        """Returns the minimum `on_power_threshold` across all appliances 
        immediately downstream of this meter.  If any appliance 
        does not have an `on_power_threshold` then default to 10 watts."""
        DEFAULT_ON_POWER_THRESHOLD = 10
        on_power_thresholds = [
            appl.metadata.get('on_power_threshold', DEFAULT_ON_POWER_THRESHOLD)
            for appl in self.appliances]
        if on_power_thresholds:
            return min(on_power_thresholds)
        else:
            return DEFAULT_ON_POWER_THRESHOLD

    def matches_appliances(self, key):
        """
        Parameters
        ----------
        key : dict

        Returns
        -------
        True if all key:value pairs in `key` match any appliance
        in `self.appliances`.
        """
        for appliance in self.appliances:
            if appliance.matches(key):
                return True
        return False

    def power_series_all_data(self, **kwargs):
        chunks = []        
        for series in self.power_series(**kwargs):
            if len(series) > 0:
                chunks.append(series)
        if chunks:
            all_data = pd.concat(chunks)
        else:
            all_data = None
        return all_data

    def plot(self, **loader_kwargs):
        all_data = self.power_series_all_data(**loader_kwargs)
        return all_data.plot()
        """ TODO:
        Parameters
        ----------
        stacked : {'stacked', 'heatmap', 'lines', 'snakey'}

        pretty snakey:
        http://www.cl.cam.ac.uk/research/srg/netos/c-aware/joule/V4.00/
        """

    def proportion_of_upstream(self, **load_kwargs):
        """Returns a value in the range [0,1] specifying the proportion of
        the upstream meter's total energy used by this meter.
        """
        upstream = self.upstream_meter()
        upstream_good_sects = upstream.good_sections(**load_kwargs)
        proportion_of_energy = (self.total_energy(sections=upstream_good_sects) /
                                upstream.total_energy(sections=upstream_good_sects))
        if isinstance(proportion_of_energy, pd.Series):
            best_ac_type = select_best_ac_type(proportion_of_energy.keys())
            return proportion_of_energy[best_ac_type]
        else:
            return proportion_of_energy

    def vampire_power(self, **load_kwargs):
        # TODO: this might be a naive approach to calculating vampire power.
        return self.power_series_all_data(**load_kwargs).min()

    
    def uptime(self, **load_kwargs):
        """
        Returns
        -------
        timedelta: total duration of all good sections.
        """
        good_sections = self.good_sections(**load_kwargs)
        if not good_sections or len(good_sections) == 0:
            return
        uptime = good_sections[0].timedelta
        for good_section in good_sections[1:]:
            uptime += good_section.timedelta
        return uptime

  

    def average_energy_per_period(self, offset_alias='D', **load_kwargs):
        """Calculate the average energy per period.  e.g. the average 
        energy per day.

        Parameters
        ----------
        offset_alias : str
            A Pandas `offset alias`.  See:
            pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

        Returns
        -------
        pd.Series
            Keys are AC types.
            Values are energy in kWh per period.
        """
        uptime_secs = self.uptime(**load_kwargs).total_seconds()
        periods = uptime_secs / offset_alias_to_seconds(offset_alias)
        energy = self.total_energy(**load_kwargs)
        return energy / periods
        
    def proportion_of_energy(self, other, **loader_kwargs):
        """
        Parameters
        ----------
        other : nilmtk.MeteGroup or ElecMeter
            Typically this will be mains.

        Returns
        -------
        float [0,1]
        """
        good_other_sections = other.good_sections(**loader_kwargs)
        loader_kwargs['sections'] = good_other_sections
        total_energy = self.total_energy(**loader_kwargs)
        if total_energy.empty:
            return 0.0

        # TODO test effect of setting `sections` for other
        other_total_energy = other.total_energy(**loader_kwargs)
        other_ac_types = other_total_energy.keys()
        self_ac_types = total_energy.keys()
        shared_ac_types = set(other_ac_types).intersection(self_ac_types)
        n_shared_ac_types = len(shared_ac_types)
        if n_shared_ac_types > 1:
            return (total_energy[shared_ac_types] / 
                    other_total_energy[shared_ac_types]).mean()
        elif n_shared_ac_types == 0:
            ac_type = select_best_ac_type(self_ac_types)
            other_ac_type = select_best_ac_type(other_ac_types)
            warn("No shared AC types.  Using '{:s}' for submeter"
                 " and '{:s}' for other.".format(ac_type, other_ac_type))
        elif n_shared_ac_types == 1:
            ac_type = list(shared_ac_types)[0]
            other_ac_type = ac_type
        return total_energy[ac_type] / other_total_energy[other_ac_type]

  #   def activity_distribution(self):
  # * activity distribution:
  #   - use ElecMeter.get_timeframe() to get start and end
  #   - use pd.period_range to make daily period index
  #   - load daily chunks
  #   - downsample using Apply
  #   - when_on
  #   - answers = zeros(n_timeslices_per_chunk)
  #   - add each chunk to answers
  #   - answer is now the histogram!



def align_two_meters(master, slave, func='power_series'):
    """Returns a generator of 2-column pd.DataFrames.  The first column is from
    `master`, the second from `slave`.

    Takes the sample rate and good_periods of `master` and applies to `slave`.

    Parameters
    ----------
    master, slave : ElecMeter or MeterGroup instances
    """
    sample_period = master.sample_period()
    period_alias = '{:d}S'.format(sample_period)
    sections = master.good_sections()
    master_generator = getattr(master, func)(sections=sections)
    for master_chunk in master_generator:
        if len(master_chunk) < 2:
            return
        chunk_timeframe = TimeFrame(master_chunk.index[0],
                                    master_chunk.index[-1])
        slave_generator = getattr(slave, func)(sections=[chunk_timeframe])
        slave_chunk = next(slave_generator)

        # TODO: do this resampling in the pipeline?
        slave_chunk = slave_chunk.resample(period_alias)
        master_chunk = master_chunk.resample(period_alias)

        yield pd.DataFrame({'master': master_chunk, 'slave': slave_chunk})

