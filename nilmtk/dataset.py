from __future__ import print_function, division
import os
from collections import OrderedDict
import pandas as pd
from .building import Building
from .datastore.datastore import join_key
from .utils import get_datastore
from .timeframe import TimeFrame

class DataSet(object):
    """
    Attributes
    ----------
    buildings : OrderedDict
        Each key is an integer, starting from 1.
        Each value is a nilmtk.Building object.

    store : nilmtk.DataStore

    metadata : dict
        Metadata describing the dataset name, authors etc.
        (Metadata about specific buildings, meters, appliances etc.
        is stored elsewhere.)
        See http://nilm-metadata.readthedocs.org/en/latest/dataset_metadata.html#dataset
    """

    def __init__(self, filename=None, format='HDF'):
        """
        Parameters
        ----------
        filename : str
            path to data set
        
        format : str
            format of output. Either 'HDF' or 'CSV'. Defaults to 'HDF'
        """
        self.store = None
        self.buildings = OrderedDict()
        self.metadata = {}
        if filename is not None:
            self.import_metadata(get_datastore(filename, format))
        
    def import_metadata(self, store):
        """
        Parameters
        ----------
        store : nilmtk.DataStore
        """
        self.store = store
        self.metadata = store.load_metadata()
        self._init_buildings(store)
        return self
        
    def save(self, destination):
        for b_id, building in self.buildings.iteritems():
            building.save(destination, '/building' + str(b_id))

    def _init_buildings(self, store):
        buildings = store.elements_below_key('/')
        buildings.sort()

        for b_key in buildings:
            building = Building()
            building.import_metadata(store, '/'+b_key, self.metadata.get('name'))
            self.buildings[building.identifier.instance] = building

    def set_window(self, start=None, end=None):
        """Set the timeframe window on self.store. Used for setting the 
        'region of interest' non-destructively for all processing.
        
        Parameters
        ----------
        start, end : str or pd.Timestamp or datetime or None
        """
        if self.store is None:
            raise RuntimeError("You need to set self.store first!")

        tz = self.metadata.get('timezone')
        if tz is None:
            raise RuntimeError("'timezone' is not set in dataset metadata.")

        self.store.window = TimeFrame(start, end, tz)

    def describe(self, **kwargs):
        """Returns a DataFrame describing this dataset.  
        Each column is a building.  Each row is a feature."""
        keys = self.buildings.keys()
        keys.sort()
        results = pd.DataFrame(columns=keys)
        for i, building in self.buildings.iteritems():
            results[i] = building.describe(**kwargs)
        return results
