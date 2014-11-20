from __future__ import print_function, division
import os
from collections import OrderedDict
from .building import Building
from .datastore import join_key, get_datastore

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
        self.buildings = OrderedDict()
        self.metadata = {}
        self.load(get_datastore(filename, format))
        
    def load(self, store):
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

        for b_key in buildings:
            building = Building()
            building.load(store, '/'+b_key, self.metadata.get('name'))
            self.buildings[building.identifier.instance] = building
