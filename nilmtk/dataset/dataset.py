import os
import json
import copy
import pandas as pd

"""Base class for all datasets."""


class DataSet(object):

    """Base class for all datasets.  This class can be used
    for loading nilmtk's REDD+ data format.

    Attributes
    ----------

    buildings : dict
        Each key is a string representing the name of the building and is 
        preserved from the original dataset.  Each value is a 
        nilmtk.building.Building object.

    metadata : dict
        Metadata regarding this DataSet.  Keys include:

        name : string
            Abbreviated name for the dataset, e.g. "REDD"

        full_name : string
            Full name of the dataset, eg. "Reference Energy Disaggregation Data Set"

        urls : list of strings, optional
            The URL(s) for more information about this dataset

        citations : list of strings, optional
            Academic citation(s) for this dataset

        nominal_voltage : float, optional

        timezone : string

        geographic_coordinates : pair (lat, long), optional
            The geo location of the research institution.  Used as a fall back
            if geo location isn't available for any individual building.
    
    """

    def __init__(self):
        self.buildings = {}
        self.metadata = {}

    def load(self, root_directory):
        """Load entire dataset into memory"""
        building_names = self.load_building_names(root_directory)
        for building in building_names:
            self.load_building(root_directory, building)

    def export(self, directory, format='HDF5', compact=False):
        """Export dataset to disk as HDF5.

        Arguments
        ---------
        directory : str
            Output directory

        format : str, optional
            `REDD+` or `HDF5`

        compact : boolean, optional
            Defaults to false.  If True then only save change points.
        """
        store = pd.HDFStore(
            os.path.join(directory, 'dataset.h5'), complevel=9)
        print store
        for building_name in self.buildings:
            building = self.buildings[building_name]
            utility = building.utility
            electric = utility.electric
            mains = electric.mains
            for main in mains:
                store.put('/%s/utility/electric/mains/%d/%d/' %
                          (building_name, main.split, main.meter),
                          mains[main], table=True)
            appliances = electric.appliances
            for appliance in appliances:
                store.put('%s/utility/electric/appliances/%s/%d/' %
                          (building_name, appliance.name, appliance.instance),
                          appliances[appliance], table=True)
        store.close()

    def print_summary_stats(self):
        raise NotImplementedError

    # This will be overridden by each subclass
    def load_building_names(self, root_directory):
        """return list of building names"""
        raise NotImplementedError

    # This will be overridden by each subclass
    def load_building(self, root_directory, building_name):
        # convert units
        # convert to standard appliance names
        raise NotImplementedError

    def to_json_temp(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

    def to_json(self):
        '''Returns the JSON representation of the dataset'''
        representation = copy.copy(self.metadata)
        representation["buildings"] = {}
        # Accessing list of buildings
        for building_name, building in self.buildings.iteritems():
            representation["buildings"][building_name] = building.to_json()

        return json.dumps(representation)
