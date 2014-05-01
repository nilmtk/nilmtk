from results import Results
import pandas as pd
import numpy as np
import copy
from nilmtk import TimeFrame
from nilmtk.measurement import AC_TYPES

class EnergyResults(Results):
    """
    Attributes
    ----------
    _data : pd.DataFrame
        index is start date
        `end` is end date
        `active` is (optional) energy in kWh
        `reactive` is (optional) energy in kVARh
        `apparent` is (optional) energy in kVAh
    """
    
    @property
    def combined(self):
        """Return all results from each chunk combined.
        Either return single float for all periods or a dict where necessary, 
        e.g. if calculating Energy for a meter which records both apparent power and
        active power then we've have energyresults.combined['active']
        """
        return self._data[self._columns_with_end_removed()].sum()

    def append(self, timeframe, **kwargs):
        """Append a single result.
        e.g. append(TimeFrame(start, end), apparent=34, active=43)
        """
        if set(kwargs.keys()) - set(AC_TYPES):
            raise KeyError('kwargs must be a combination of '+
                           str(AC_TYPES))
        super(EnergyResults, self).append(timeframe, **kwargs)

    def update(self, new_result):
        """Update with new result."""
        if not isinstance(new_result, EnergyResults):
            raise TypeError('new_results must be of type EnergyResults')
        super(EnergyResults, self).update(new_result)
