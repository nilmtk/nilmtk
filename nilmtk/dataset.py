from __future__ import print_function, division
from collections import OrderedDict
from .building import Building
from .datastore import join_key, HDFDataStore

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

    def __init__(self, filename=None):
        """
        Parameters
        ----------
        filename : str
            HDF5 file
        """
        self.buildings = OrderedDict()
        self.metadata = {}
        if filename is not None:
            self.load(HDFDataStore(filename))
        
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
