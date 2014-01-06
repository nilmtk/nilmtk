import os
import json
import copy
import pandas as pd
from nilmtk.building import Building
from nilmtk.sensors.electricity import MainsName
from nilmtk.sensors.electricity import ApplianceName
from nilmtk.sensors.electricity import Measurement

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

    def load_hdf5(self, directory):
        """Imports dataset from HDF5 store into NILMTK object

        Arguments
        ----------

        directory : str
            Directory where the HDF5 store is located

        """
        store = pd.HDFStore(
            os.path.join(directory, 'dataset.h5'))
        self.buildings = {}

        # Finding all keys stored in the HDF5 store
        keys = store.keys()

        # Finding the buildings
        building_numbers = list(set([key.split("/")[1] for key in keys]))

        # Loading the structured information for each building
        for building_number in building_numbers:

            # Create a new building and add it to buildings
            b = Building()
            self.buildings[int(building_number)] = b

            # Find the keys which start with this particular building
            keys_building = [
                key for key in keys if key.split("/")[1] == building_number]

            # Loading utilites
            keys_utilities = [
                key for key in keys_building if "utility" in key]

            # Load electric if len(keys_utilities)>0

            if len(keys_utilities) > 0:
                # Load electric
                keys_electric = [
                    key for key in keys_utilities if "electric" in key]

                # Loading mains
                keys_mains = [
                    key for key in keys_electric if "mains" in key]

                if len(keys_mains) > 0:
                    b.utility.electric.mains = {}
                    for key in keys_mains:
                        mains_split = int(key.split("/")[-2])
                        mains_instance = int(key.split("/")[-1])
                        mains_name = MainsName(mains_split, mains_instance)
                        b.utility.electric.mains[mains_name] = store[key]

                # Loading appliances
                keys_appliances = [
                    key for key in keys_electric if "appliances" in key]

                if len(keys_appliances) > 0:
                    b.utility.electric.appliances = {}
                    for key in keys_appliances:
                        appliance_name = key.split("/")[-2]
                        appliance_instance = int(key.split("/")[-1])
                        appliance_name = ApplianceName(
                            appliance_name, appliance_instance)
                        b.utility.electric.appliances[
                            appliance_name] = store[key]

    def export_csv(self, directory):
        """For now just created this function separately

        NB: Needs to be written!! Don't expect this to run at the moment

        Parameters
        ----------
        directory : Complete path where to export the data 
        """
        for building_number in self.buildings:
            print("Writing data for %d" % (building_number))
            building = self.buildings[building_number]
            utility = building.utility
            electric = utility.electric
            mains = electric.mains
            for main in mains:
                dir_path = os.path.join(os.path.abspath(directory),
                                        "/%d/utility/electric/mains/%d_%d.csv"
                                        % (building_number, main.split,
                                           main.meter))
                print(dir_path)
                print("*"*80)
                os.makedirs(dir_path)
                temp = mains[main].copy()
                temp.index = (temp.index.astype(int) / 1e9).astype(int)
                temp.rename(columns=lambda x: "%s_%s" %
                            (x.physical_quantity, x.type), inplace=True)
                temp.to_csv(dir_path, index_label="timestamp")

            appliances = electric.appliances
            for appliance in appliances:
                dir_path = os.path.join(os.path.abspath(directory),
                                        "/%d/utility/electric/appliances/%d_%d.csv"
                                        % (building_number, appliance.name,
                                           appliance.instance))
                os.makedirs(dir_path)
                temp = mains[main].copy()
                temp.index = (temp.index.astype(int) / 1e9).astype(int)
                temp.rename(columns=lambda x: "%s_%s" %
                            (x.physical_quantity, x.type), inplace=True)
                temp.to_csv(dir_path, index_label="timestamp")

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
            os.path.join(directory, 'dataset.h5'), complevel=9, complib='zlib')
        for building_number in self.buildings:
            print("Writing data for %d" % (building_number))
            building = self.buildings[building_number]
            utility = building.utility
            electric = utility.electric
            mains = electric.mains
            for main in mains:

                store.put('/%d/utility/electric/mains/%d/%d/' %
                          (building_number, main.split, main.meter),
                          mains[main], table=True)
            appliances = electric.appliances
            for appliance in appliances:
                store.put('%d/utility/electric/appliances/%s/%d/' %
                          (building_number, appliance.name,
                           appliance.instance),
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
