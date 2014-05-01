from __future__ import print_function, division

class Loader(object):
    """Owns exactly one DataStore. Knows how to chunk up data using masks.
    Responsible for managing the transfer of data from disk (via DataStore)
    and into main system memory.

    Design notes
    ------------
    If we really wanted to, a lot of the responsibility of Loader could be put
    into DataStore and/or Meter.  But Loader feels like the best place
    to store large DataFrames in-memory (i.e. to do persistence) and
    we want to pass a Loader into Pipeline, not a Meter (because 
    Meters use Pipeline so we'd have a horrible two-way interdependence)

    Attributes
    ----------
    store : nilmtk.DataStore
    key : string, , the location of a table within the DataStore
    mask : list of nilmtk.TimeFrame objects or None (to load all data)
    """

    def __init__(self, store, key):
        self.store = store
        self.key = key
        self.mask = None

    def load_chunks(self, cols=None):
        """

        Set `mask` attribute (a list of TimeFrames) before calling
        `load_chunks` to control which chunks are loaded.

        Parameters
        ----------
        cols : list or 'index', optional
            e.g. [('power', 'active'), ('power', 'reactive'), ('voltage', '')]
            if not provided then will return all columns from the table.

        Returns
        -------
        generator of DataFrame objects.  If `self.mask` is not None (i.e. it
        is a list of TimeFrame objects) then `load_chunks` will always return
        the same number of chunks as there are items in `self.mask`.  Any empty
        chunks will be represented as an empty DataFrame.

        """
        
        # TODO: this would be much more efficient 
        # if we first got row indicies for each period,
        # then checked each period will fit into memory,
        # and then iterated over the row indicies.      

        self.store._check_columns(self.key, cols)
        periods = [self.store.timeframe(self.key)] if self.mask is None else self.mask
        for period in periods:
            data = self.store.load(key=self.key, cols=cols, period=period)
            yield data

    def load_metadata(self):
        return self.store.load_metadata(self.key)
