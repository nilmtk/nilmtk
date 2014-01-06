'''
Pecan Dataset Loader

These routines load Pecan Dataset into NILMTK Dataset format

Authors :
License:

TODOS
-----

1. Handle LEG2V
2. Handle Grid
3. Handle Solar
4. Handle Gen
'''

import pandas as pd
from nilmtk.dataset import DataSet
from nilmtk.building import Building
from nilmtk.utils import get_immediate_subdirectories
from nilmtk.sensors.electricity import Measurement
from nilmtk.sensors.electricity import ApplianceName
from nilmtk.sensors.electricity import MainsName
import os
from collections import defaultdict

# Mapping between appliances actual name
appliance_name_mapping = {
    'ksac': 'misc',
    'tvroom': 'misc',
    'garage/refrigerator': 'refrigerator',
    'office': 'computer',
    'dryg': 'dryer_gas',
    'bath': 'bathroom',
    'genlight': 'lighting',
    'oven': 'oven',
    'bedroom': 'misc',
    'subl': 'subpanel',
    'masterbed': 'misc',
    'bathroom': 'bathroom',
    'livingroom': 'misc',
    'sprinkler': 'sprinkler',
    'disposal': 'disposal',
    'masterbath':
    'bathroom',
    'microwave': 'microwave',
    'drye': 'dryer_electric',
    'smallappliance': 'misc',
    'washer': 'washer',
    'furnace': 'furnace',
    'gri': 'grid',
    'lighting&plugs': 'plugs',
    'famroom': 'misc',
    'dryer': 'dryer',
    'diningroom': 'misc',
    'ove': 'oven',
    'backyard': 'misc',
    'cooktop': 'cooktop',
    'refrigerator': 'fridge',
    'kitchen': 'kitchen',
    'dishwasher': 'dishwasher',
    'theater': 'theater',
    'washingmachine': 'washingmachine',
    'car': 'car',
    'air': 'ac',
    'garage': 'misc',
    'range': 'range',
    'waterheater': 'waterheater',
    'security': 'security',
    'ai': 'ac'
}


class Pecan(DataSet):

    def __init__(self):
        super(Pecan, self).__init__()
        self.metadata = {
            'name': 'Pecan Street',
            'urls': ['http://www.pecanstreet.org/',
                     'http://www.pecanstreet.org/2013/04/with-free-sample-data'
                     '-set-pecan-street-research-institute-expands-access-to'
                     '-world-class-energy-use-data-to-global'
                     '-university-researchers/']
        }

    def load(self, root_directory):
        """Load entire dataset into memory"""
        building_names = self.load_building_names(root_directory)
        print (building_names)
        for building_name in building_names:
            self.load_building(root_directory, building_name)

    def add_mains(self, building, df):
        # Find columns containing mains in them
        mains_column_names = [x for x in df.columns if x]

        # Adding mains
        building.utility.electric.mains = {}
        building.utility.electric.mains[
            MainsName(1, 1)] = df[mains_column_names]
        return building

    def add_appliances(self, building, df):
        # Getting a list of appliance names
        appliance_names = list(set([a.split("_")[0] for a in df.columns
                                    if "mains" not in a]))

        # Adding appliances
        building.utility.electric.appliances = {}
        for appliance in appliance_names:
            # Finding headers corresponding to the appliance
            names = [x for x in df.columns if x.split("_")[0] == appliance]

            # TODO: Replace column names and remove the appliance name from
            # them
            building.utility.electric.appliances[appliance] = df[names]
        return building

    def standardize(self, df, building):

        # Converting power from kW to W
        # Note some homes contain Voltage as well, need to multiply that
        # back with 1e3
        df = df * 1e3

        # Convert to standard appliance names
        # 1. Mains is use [kW]; replace space with mains_0_active
        # 2. If voltage is present, rename the column and divide it by 1e3
        # 3. If 'gen' is present, delete the column; TODO think about where
        # to put this column
        # 4. Delete 'Grid' column; TODO same as #3
        # 5. Lower case all appliance names
        # 6 Replace " " with "_" in appliance name
        # 7. Appliance names should have separate active and apparent fields
        # (have a *)

        # 1
        df = df.rename(columns={'use [kW]': Measurement('power', 'active')})
        print df.columns

        # Adding Mains Power
        building.utility.electric.mains = {}
        building.utility.electric.mains[
            MainsName(1, 1)] = df[[Measurement('power', 'active')]]
        df = df.drop(Measurement('power', 'active'), 1)

        # 2
        if "LEG1V [V]" in df.columns:
            df = df.rename(columns={"LEG1V [V]": Measurement('voltage', '')})

            # Adding voltage if it exists
            building.utility.electric.mains[
                MainsName(1, 1)][Measurement('voltage', '')] = df[Measurement('voltage', '')] / 1e3
            df = df.drop(Measurement('voltage', ''), 1)
            '''For now delete leg2
            df = df.rename(columns=lambda x: x.replace("LEG2V [V]",
                                                       "mains_2_voltage"))            
            df['mains_2_voltage'] = df['mains_2_voltage'] / 1e3'''

            # TODO: See what to do with this bit of information
            df = df.drop('LEG2V [V]', 1)

        # 3
        if "gen [kW]" in df.columns:
            df = df.drop('gen [kW]', 1)

        # 4
        if 'Grid [kW]' in df.columns:
            df = df.drop('Grid [kW]', 1)
        if "Grid* [kVA]" in df.columns:
            df = df.drop('Grid* [kVA]', 1)

        # 5
        df = df.rename(columns=lambda x: x.lower())

        # 6
        df = df.rename(columns=lambda x: x.replace(" ", "_"))

        # 7
        df = df.rename(columns=lambda x: x.replace("[kw]", "active"))
        df = df.rename(columns=lambda x: x.replace("[kva]", "apparent"))
        df = df.rename(columns=lambda x: x.replace("*", ""))

        # List of appliance names
        appliance_names = list(set([a.split("_")[0] for a in df.columns
                                    if type(a) != type(Measurement('power', 'active'))]))

        # Adding appliances
        building.utility.electric.appliances = {}
        building_appliance_count = defaultdict(int)
        for appliance in appliance_names:
            # Finding headers corresponding to the appliance
            names = [x for x in df.columns if x.split("_")[0] == appliance]

            names_modified = [Measurement('power', x.split("_")[1])
                              for x in names]
            name_modification = {names[i]: names_modified[i] for i in range(len(names))}

            # TODO: Replace column names and remove the appliance name from
            # them
            if appliance[:-1] in appliance_name_mapping.keys():
                appliance_name = appliance_name_mapping[appliance[:-1]]
                building_appliance_count[appliance_name] += 1
                appliance_instance = building_appliance_count[appliance_name]
                building.utility.electric.appliances[
                    ApplianceName(appliance_name, appliance_instance)] = df[names]
                building.utility.electric.appliances[
                    ApplianceName(appliance_name, appliance_instance)] = building.utility.electric.appliances[
                    ApplianceName(appliance_name, appliance_instance)].rename(columns=name_modification)

                building.utility.electric.appliances[
                    ApplianceName(appliance_name, appliance_instance)] = building.utility.electric.appliances[
                    ApplianceName(appliance_name, appliance_instance)].astype('float32')
            building.utility.electric.mains[
                MainsName(1, 1)] = building.utility.electric.mains[
                MainsName(1, 1)].astype('float32')

        return building


class Pecan_15min(Pecan):

    def __init__(self):
        super(Pecan_15min, self).__init__()

    def load_building(self, root_directory, building_name):
        spreadsheet = pd.ExcelFile(os.path.join(root_directory,
                                                "15_min/Homes 01-10_15min_2012-0819-0825 .xlsx"))
        df = spreadsheet.parse(building_name, index_col=0,
                               date_parser=True).astype('float32')

        # Create a new building
        building = Building()

        building = self.standardize(df, building)

        # Adding this building to dict of buildings
        building_name = building_name.replace(" ", "_")
        building_number = int(building_name[-2:])
        self.buildings[building_number] = building

    def load_building_names(self, root_directory):
        spreadsheet = pd.ExcelFile(os.path.join(root_directory,
                                                "15_min/Homes 01-10_15min_2012-0819-0825 .xlsx"))
        return spreadsheet.sheet_names


class Pecan_1min(Pecan):

    def __init__(self):
        super(Pecan_1min, self).__init__()
        self.metadata = {
            'urls': ['http://www.pecanstreet.org/']
        }

    def load_building(self, root_directory, building_name):
        ''' Loads electrical data for specified building
        '''

        # Each building has a week worth data
        # Files are named as follows:
        # Home 01_1min_2012-0903.xlsx to Home 01_1min_2012-0909.xlsx
        # Pattern building_name + "_1min_2012-09" + ['03'-'09'].xlsx

        building_folder = os.path.join(root_directory, '1_min', building_name)
        df = pd.DataFrame()
        for day in ["03", "04", "05", "06", "07", "08", "09"]:
            spreadsheet = pd.ExcelFile(os.path.join(building_folder,
                                                    "%s_1min_2012-09%s.xlsx" % (building_name, day)))
            temp_df = spreadsheet.parse(
                'Sheet1', index_col=0, date_parser=True)
            df = df.append(temp_df)

        building = Building()

        building = self.standardize(df, building)

        # Adding this building to dict of buildings
        building_name = building_name.replace(" ", "_")
        building_number = int(building_name[-2:])
        self.buildings[building_number] = building

    def load_building_names(self, root_directory):
        dirs = get_immediate_subdirectories(os.path.join(root_directory,
                                                         "1_min"))
        return dirs
