import abc
import pandas as pd
import copy
from .timeframe import TimeFrame

class Results(object):
    """Stats results from each node need to be assigned to a specific
    class so we know how to combine results from multiple chunks.  For
    example, Energy can be simply summed; while dropout rate should be
    averaged, and gaps need to be merged across chunk boundaries.  Results
    objects contain a DataFrame, the index of which is the start timestamp for
    which the results are valid; the first column ('end') is the end
    timestamp for which the results are valid.  Other columns are accumulators 
    for the results.

    Attributes
    ----------
    _data : DataFrame
        Index is period start.  
        Columns are: `end` and any columns for internal storage of stats.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._data = pd.DataFrame(columns=['end'])

    def combined(self):
        """Return all results from each chunk combined.  Either return single
        float for all periods or a dict where necessary, e.g. if
        calculating Energy for a meter which records both apparent
        power and active power then get active power with
        energyresults.combined['active']
        """
        return self._data[self._columns_with_end_removed()].sum()

    def per_period(self):
        """return a DataFrame.  Index is period start.  
        Columns are: end and <stat name>
        """
        return copy.deepcopy(self._data)

    def append(self, timeframe, new_results):
        """Append a single result.

        Parameters
        ----------
        timeframe : nilmtk.TimeFrame
        new_results : dict
        """
        assert isinstance(timeframe, TimeFrame), type(timeframe)
        assert isinstance(new_results, dict), type(new_results)

        # check that there is no overlap
        for index, series in self._data.iterrows():
            tf = TimeFrame(index, series['end'])
            intersect = tf.intersect(timeframe)
            if not intersect.empty:
                raise ValueError("Periods overlap" + str(intersect))

        row = pd.DataFrame(index=[timeframe.start],
                           columns=['end'] + new_results.keys())
        row['end'] = timeframe.end
        for key, val in new_results.iteritems():
            row[key] = val
        self._data = self._data.append(row, verify_integrity=True)
        self._data.sort_index(inplace=True)

    def update(self, new_result):
        """Add results from a new chunk.
        
        Parameters 
        ---------- 
        new_result : Results subclass (same
            class as self) from new chunk of data.

        """
        if not isinstance(new_result, self.__class__):
            raise TypeError("new_results must be of type '{}'"
                            .format(self.__class__))

        self._data = self._data.append(new_result._data, verify_integrity=True)
        self._data.sort_index(inplace=True)

    def unify(self, other):
        """Take results from another table of data (another physical meter)
        and merge those results into self.  For example, if we have a dual-split
        mains supply then we want to merge the results from each physical meter.

        Parameters
        ----------
        other : Results subclass (same class as self).
            Results calculated from another table of data.
        """
        assert isinstance(other, self.__class__)
        for i, row in self._data.iterrows():
            assert other._data['end'].loc[i] == row['end']

    def _columns_with_end_removed(self):
        cols = set(self._data.columns)
        cols.remove('end')
        cols = list(cols)
        return cols

    def __repr__(self):
        return str(self._data)
