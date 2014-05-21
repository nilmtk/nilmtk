from __future__ import print_function, division
from collections import namedtuple
from .electricity import Electricity
from .datastore import join_key

BuildingID = namedtuple('BuildingID', ['instance', 'dataset'])

class Building(object):   
    """
    Attributes
    ----------
    metadata : dict
        instance : int, building instance, starts from 1
        dataset : string
        original_name : string
    """
    def __init__(self):
        self.electric = Electricity()
        self.metadata = {}
    
    def load(self, store, key):
        self.metadata = store.load_metadata(key)
        elec_meters = self.metadata.get('elec_meters', [])
        self.electric.load(store, elec_meters)
                
    def save(self, destination, key):
        destination.write_metadata(key, self.metadata)
        self.electric.save(destination, join_key(key, 'electric'))

    def identifier(self):
        md = self.metadata
        return BuildingID(instance=md.get('instance'), 
                          dataset=md.get('dataset'))
