from __future__ import print_function, division
from collections import namedtuple, OrderedDict
from .metergroup import MeterGroup
from .datastore import join_key
from .hashable import Hashable

BuildingID = namedtuple('BuildingID', ['instance', 'dataset'])

class Building(Hashable):
    """
    Attributes
    ----------
    elec : MeterGroup

    metadata : dict
        Metadata just about this building (e.g. geo location etc).
        See http://nilm-metadata.readthedocs.org/en/latest/dataset_metadata.html#building
        Has these additional keys: 
        dataset : string
    """
    def __init__(self):
        self.elec = MeterGroup()
        self.metadata = {}
    
    def load(self, store, key, dataset_name):
        self.metadata = store.load_metadata(key)
        if not self.metadata.has_key('dataset'):
            self.metadata['dataset'] = dataset_name
        elec_meters = self.metadata.pop('elec_meters', {})
        appliances = self.metadata.pop('appliances', [])
        self.elec.load(store, elec_meters, appliances, self.identifier)
                
    def save(self, destination, key):
        destination.write_metadata(key, self.metadata)
        self.elec.save(destination, join_key(key, 'elec'))

    @property
    def identifier(self):
        md = self.metadata
        return BuildingID(instance=md.get('instance'), 
                          dataset=md.get('dataset'))

    def describe(self):
        """Returns a dict describing this building."""
        md = self.metadata
        d = OrderedDict()
        def copy_from_metadata_to_dict(key):
            d[key] = md.get(key)

        for key in ['instance', 'building_type',
                    'construction_year', 'energy_improvements', 'heating', 
                    'ownership', 'n_occupants', 'description_of_occupants']:
            copy_from_metadata_to_dict(key)

        return d
