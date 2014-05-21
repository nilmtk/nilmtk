from __future__ import print_function, division
from collections import OrderedDict
from nilmtk.building import Building
from nilmtk.datastore import join_key

class DataSet(object):
    """
    Attributes
    ----------
    buildings : OrderedDict
        Each key is an integer, starting from 1.
        Each value is a nilmtk.Building object.

    metadata : dict
        See http://nilm-metadata.readthedocs.org/en/latest/dataset_metadata.html#dataset
    """

    def __init__(self):
        self.buildings = OrderedDict()
        
    def load(self, store):
        """
        Parameters
        ----------
        store : nilmtk.DataStore
        """
        self._init_buildings(store)
        return self
        
    def save(self, destination):
        for b_id, building in self.buildings.iteritems():
            building.save(destination, '/building' + str(b_id))

    def _init_buildings(self, store):
        buildings = store.elements_below_key('/')

        for b_key in buildings:
            building = Building()
            building.load(store, '/'+b_key)
            self.buildings[building.identifier().instance] = building
