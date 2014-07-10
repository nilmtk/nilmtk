class ElecMeterAndMeterGroup(object):
    """Common implementations of methods shared by ElecMeter and MeterGroup.
    """

    def when_on(self, **load_kwargs):
        """Are the connected appliances appliance is on (True) or off (False)?

        Uses `self.min_on_power_threshold()`.

        Parameters
        ----------
        **load_kwargs : key word arguments
            Passed to self.power_series()

        Returns
        -------
        generator of pd.Series
            index is the same as for chunk returned by `self.power_series()`
            values are booleans
        """
        on_power_threshold = self.min_on_power_threshold()
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
