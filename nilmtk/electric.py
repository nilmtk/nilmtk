import pandas as pd
from .timeframe import TimeFrame
from .measurement import select_best_ac_type

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
            chunks.append(series)
        return pd.concat(chunks)

    def plot(self, **loader_kwargs):
        all_data = self.power_series_all_data(**loader_kwargs)
        all_data.plot()
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
        return self.power_series_all_data().min()

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
        slave_generator = getattr(slave, func)(sections=[chunk_timeframe],
                                               chunksize=1E9)
        slave_chunk = next(slave_generator)

        # TODO: do this resampling in the pipeline?
        slave_chunk = slave_chunk.resample(period_alias)
        master_chunk = master_chunk.resample(period_alias)

        yield pd.DataFrame({'master': master_chunk, 'slave': slave_chunk})

