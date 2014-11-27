import abc
import pandas as pd
import copy
from .timeframe import TimeFrame
from nilmtk.utils import get_tz, tz_localize_naive

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

    Static Attributes
    -----------------
    name : str
        The string used to cache this results object.
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

    def simple(self):
        """Returns the simplest representation of the results."""
        return self.combined()

    def append(self, timeframe, new_results):
        """Append a single result.

        Parameters
        ----------
        timeframe : nilmtk.TimeFrame
        new_results : dict
        """
        if not isinstance(timeframe, TimeFrame):
            raise TypeError("`timeframe` must be of type 'nilmtk.TimeFrame',"
                            " not '{}' type.".format(type(timeframe)))
        if not isinstance(new_results, dict):
            raise TypeError("`new_results` must of a dict, not '{}' type."
                            .format(type(new_results)))
        
        # check that there is no overlap
        for index, series in self._data.iterrows():
            tf = TimeFrame(index, series['end'])
            tf.check_for_overlap(timeframe)

        row = pd.DataFrame(index=[timeframe.start],
                           columns=['end'] + new_results.keys())
        row['end'] = timeframe.end
        for key, val in new_results.iteritems():
            row[key] = val
        self._data = self._data.append(row, verify_integrity=True)
        self._data.sort_index(inplace=True)

    def check_for_overlap(self):
        n = len(self._data)
        index = self._data.index
        for i in range(n):
            row1 = self._data.iloc[i]
            tf1 = TimeFrame(index[i], row1['end'])
            for j in range(i+1, n):
                row2 = self._data.iloc[j]
                tf2 = TimeFrame(index[j], row2['end'])
                tf1.check_for_overlap(tf2)

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
        self.check_for_overlap()

    def unify(self, other):
        """Take results from another table of data (another physical meter)
        and merge those results into self.  For example, if we have a dual-split
        mains supply then we want to merge the results from each physical meter.
        The two sets of results must be for exactly the same timeframes.

        Parameters
        ----------
        other : Results subclass (same class as self).
            Results calculated from another table of data.
        """
        assert isinstance(other, self.__class__)
        for i, row in self._data.iterrows():
            if other._data['end'].loc[i] != row['end']:
                raise RuntimeError("The sections we are trying to merge"
                                   " do not have the same end times so we"
                                   " cannot merge them.")

    def import_from_cache(self, cached_stat, sections):
        """
        Parameters
        ----------
        cached_stat : DataFrame of cached data
        sections : list of nilmtk.TimeFrame objects
            describing the sections we want to load stats for.
        """
        tz = get_tz(cached_stat)
        usable_sections_from_cache = []
        for section in sections:
            try:
                row = cached_stat.loc[section.start]
            except KeyError:
                pass
            else:
                # Save, disable, then re-enable `pd.SettingWithCopyWarning`
                # from http://stackoverflow.com/a/20627316/732596
                chained_assignment = pd.options.mode.chained_assignment
                pd.options.mode.chained_assignment = None
                row['end'] = tz_localize_naive(row['end'], tz)
                pd.options.mode.chained_assignment = chained_assignment

                if row['end'] == section.end:
                    usable_sections_from_cache.append(row)

        self._data = pd.DataFrame(usable_sections_from_cache)
        self._data.sort_index(inplace=True)

    def export_to_cache(self):
        """
        Returns
        -------
        pd.DataFrame

        Notes
        -----
        Objects are converted using `DataFrame.convert_objects()`.
        The reason for doing this is to strip out the timezone
        information from data columns.  We have to do this otherwise
        Pandas complains if we try to put a column with multiple
        timezones (e.g. Europe/London across a daylight saving
        boundary).
        """
        return self._data.convert_objects()

    def timeframes(self):
        """Returns a list of timeframes covered by this Result."""
        return [TimeFrame(index, row['end']) 
                for index, row in self._data.iterrows()]

    def _columns_with_end_removed(self):
        cols = set(self._data.columns)
        cols.remove('end')
        cols = list(cols)
        return cols

    def __repr__(self):
        return str(self._data)
