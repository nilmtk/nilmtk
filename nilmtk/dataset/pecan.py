'''
Pecan Dataset Loader

These routines load Pecan Dataset into NILMTK Dataset format

Authors :
License:
'''

from .dataset import DataSet
import pandas as pd
from nilmtk.building import Building

import os


class Pecan_15min(DataSet):

    def __init__(self):
        super(Pecan_15min, self).__init__()
        self.urls = ['http://www.pecanstreet.org/']
        self.citations = None

    def load(self, root_directory):
        """Load entire dataset into memory"""
        building_names = self.load_building_names(root_directory)
        print (building_names)
        for building_name in building_names:
            self.load_building(root_directory, building_name)

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
        df = spreadsheet.parse(building_name, index_col=0, date_parser=True)

        # Converting power from kW to W
        # Note some homes contain Voltage as well, need to multiply that
        # back with 1e3
        df = df * 1e3

        # Convert to standard appliance names
        # 1. Mains is use [kW]; replace space with mains_0_active
        # 2. If voltage is present, rename the column and multiply it by 1e3
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

        # 3
        df = df.drop('Grid [kW]', 1)

        # 4
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

        # Create a new building
        building = Building()

        # Find columns containing mains in them
        mains_column_names = [x for x in df.columns if "mains" in x]

        #Adding mains
        building.electric.mains = df[mains_column_names]

        # Getting a list of appliance names
        appliance_names = list(set([a.split("_")[0] for a in df.columns
                            if "mains" not in a]))

        # Adding appliances
        building.electric.appliances = {}
        for appliance in appliance_names:
            # Finding headers corresponding to the appliance
            names = [x for x in df.columns if x.split("_")[0] == appliance]
            building.electric.appliances[appliance] = df[names]

        # Adding this building to dict of buildings
        building_name = building_name.replace(" ", "_")
        self.buildings[building_name] = building

    def load_building_names(self, directory):
        spreadsheet = pd.ExcelFile(os.path.join(directory,
         "15_min/Homes 01-10_15min_2012-0819-0825 .xlsx"))
        return spreadsheet.sheet_names









