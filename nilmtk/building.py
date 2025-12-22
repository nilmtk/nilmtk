from typing import Optional
from collections import namedtuple, OrderedDict
import pandas as pd
from .metergroup import MeterGroup
from .datastore.datastore import join_key, DataStore
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
    
    def import_metadata(self, store:DataStore, key:str, dataset_name:Optional[str]=None):
        self.metadata = store.load_metadata(key)
        if 'dataset' not in self.metadata:
            assert dataset_name is not None, f"dataset not in metadata, please pass in the dataset_name"
            self.metadata['dataset'] = dataset_name
        elec_meters = self.metadata.pop('elec_meters', {})
        appliances = self.metadata.pop('appliances', [])
        self.elec.import_metadata(store, elec_meters, appliances, self.identifier)
                
    #  This method should be removed, elec(MeterGroup) has no group save, this would break
    # def save(self, destination, key):
    #     destination.write_metadata(key, self.metadata)
    #     self.elec.save(destination, join_key(key, 'elec'))

    @property
    def identifier(self) -> BuildingID:
        md = self.metadata
        return BuildingID(instance=md.get('instance'), 
                          dataset=md.get('dataset'))

    def describe(self, **kwargs) -> pd.Series:
        """Returns a Series describing this building."""
        md = self.metadata
        series = pd.Series(name=self.identifier.instance)

        for key in ['instance', 'building_type',
                    'construction_year', 'energy_improvements', 'heating', 
                    'ownership', 'n_occupants', 'description_of_occupants']:
            series[key] = md.get(key)

        series = pd.concat([series, self.elec.describe(**kwargs)])

        return series
