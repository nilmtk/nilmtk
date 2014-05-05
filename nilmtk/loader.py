from __future__ import print_function, division
from timeframe import TimeFrame

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
        self.mask = [TimeFrame()]

    def load(self, cols=None):
        """See datastore.load() for docs.

        Set `mask` attribute (a list of TimeFrames) before calling
        `load_chunks` to control which chunks are loaded.
        """
        return self.store.load(key=self.key, cols=cols, periods=self.mask)

    def load_metadata(self):
        return self.store.load_metadata(self.key)
