'''
Pecan Dataset Loader

These routines load Pecan Dataset into NILMTK Dataset format

Authors :
License:
'''

import pandas as pd

from nilmtk.dataset import DataSet
from nilmtk.building import Building
from nilmtk.utils import get_immediate_subdirectories

import os


class Pecan(DataSet):

    def __init__(self):
        super(Pecan, self).__init__()
        self.urls = ['http://www.pecanstreet.org/']
        self.citations = None

    def load(self, root_directory):
        """Load entire dataset into memory"""
        building_names = self.load_building_names(root_directory)
        print (building_names)
        for building_name in building_names:
            self.load_building(root_directory, building_name)

    def add_mains(self, building, df):
        # Find columns containing mains in them
        mains_column_names = [x for x in df.columns if "mains" in x]

        # Adding mains
        building.utility.electric.mains = df[mains_column_names]
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

    def standardize(self, df):

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
        df = df.rename(columns={'use [kW]': 'mains_0_active'})

        # 2
        if "LEG1V [V]" in df.columns:
            df = df.rename(columns=lambda x: x.replace("LEG1V [V]",
                                                       "mains_1_voltage"))
            df = df.rename(columns=lambda x: x.replace("LEG2V [V]",
                                                       "mains_2_voltage"))
            df['mains_1_voltage'] = df['mains_1_voltage'] / 1e3
            df['mains_2_voltage'] = df['mains_2_voltage'] / 1e3

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

        return df


class Pecan_15min(Pecan):

    def __init__(self):
        super(Pecan_15min, self).__init__()

    def export(self, directory, format='REDD+', compact=False):
        """Export dataset to disk as REDD+.

        Arguments
        ---------
        directory : str
            Output directory

        format : str, optional
            `REDD+` or `HDF5`

        compact : boolean, optional
            Defaults to false.  If True then only save change points.
        """
        raise NotImplementedError

    def print_summary_stats(self):
        raise NotImplementedError

    def load_building(self, root_directory, building_name):
        spreadsheet = pd.ExcelFile(os.path.join(root_directory,
                                                "15_min/Homes 01-10_15min_2012-0819-0825 .xlsx"))
        df = spreadsheet.parse(building_name, index_col=0, date_parser=True).astype('float32')
        df = self.standardize(df)

        # Create a new building
        building = Building()

        # Add mains
        building = self.add_mains(building, df)

        # Add appliances
        building = self.add_appliances(building, df)

        # Adding this building to dict of buildings
        building_name = building_name.replace(" ", "_")
        self.buildings[building_name] = building

    def load_building_names(self, root_directory):
        spreadsheet = pd.ExcelFile(os.path.join(root_directory,
                                                "15_min/Homes 01-10_15min_2012-0819-0825 .xlsx"))
        return spreadsheet.sheet_names


class Pecan_1min(Pecan):

    def __init__(self):
        super(Pecan_1min, self).__init__()
        self.urls = ['http://www.pecanstreet.org/']
        self.citations = None

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
        df = self.standardize(df)

        # Create a new building
        building = Building()

        # Add mains
        building = self.add_mains(building, df)

        # Add appliances
        building = self.add_appliances(building, df)

        # Adding this building to dict of buildings
        building_name = building_name.replace(" ", "_")
        self.buildings[building_name] = building

    def load_building_names(self, root_directory):
        dirs = get_immediate_subdirectories(os.path.join(root_directory,
                                                         "1_min"))
        return dirs
