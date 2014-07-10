import pandas as pd

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
        return min(
            [appl.metadata.get('on_power_threshold', DEFAULT_ON_POWER_THRESHOLD)
             for appl in self.appliances])


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
    master_generator = getattr(master, func)(periods=sections)
    for master_chunk in master_generator:
        slave_generator = getattr(slave, func)(periods=[master_chunk.timeframe],
                                               chunksize=1E9)
        slave_chunk = next(slave_generator)

        # TODO: do this resampling in the pipeline?
        slave_chunk = slave_chunk.resample(period_alias)
        master_chunk = master_chunk.resample(period_alias)

        yield pd.DataFrame({'master': master_chunk, 'slave': slave_chunk})

