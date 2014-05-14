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
    
    def append(self, timeframe, new_results):
        """Append a single result.
        e.g. append(TimeFrame(start, end), {'apparent': 34, 'active': 43})
        """
        if set(new_results.keys()) - set(AC_TYPES):
            raise KeyError('new_results must be a combination of ' +
                           str(AC_TYPES))
        super(EnergyResults, self).append(timeframe, new_results)

    def unify(self, other):
        super(EnergyResults, self).unify(other)
        for i, row in self._data.iterrows():
            for ac_type in AC_TYPES:
                self._data[ac_type].loc[i] += other._data[ac_type].loc[i]
