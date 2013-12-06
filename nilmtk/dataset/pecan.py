'''
Pecan Dataset Loader

These routines load Pecan Dataset into NILMTK Dataset format

Authors :
License:
'''

from .dataset import DataSet
import pandas as pd
from nilmtk.building import Building


class Pecan(DataSet):

    def __init__(self):
        self.buildings = {}


    def load(self, directory):
        """Load entire dataset into memory"""
        building_names = self.load_building_names(directory)
        for building in building_names:
            self.load_building(building, directory)

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

    def load_building(self, building, directory):
        spreadsheet = pd.ExcelFile(directory + \
         "15_min/Homes 01-10_15min_2012-0819-0825 .xlsx")
        df = spreadsheet.parse(building, index_col=0, date_parser=True)
        print building

        # Converting power from kW to W
        # Note some homes contain Voltage as well, need to multiply that
        # back with 1e3
        df = df * 1e3

        # Convert to standard appliance names
        # 1. Mains is use [kW]; replace space with mains_0_active
        # 2. If 'gen' is present, delete the column
        # 3. Delete 'Grid' column
        # 4. Lower case all appliance names
        # 5 Replace " " with "_" in appliance name
        # 6. Appliance names should have separate active and apparent fields
        # (have a *)


        # 1
        df = df.rename(columns={'use [kW]': 'mains_0_active'})

        if "LEG1V [V]" in df.columns:
            df = df.rename(columns=lambda x: x.replace("LEG1V [V]", "mains_0_voltage"))
            df = df.rename(columns=lambda x: x.replace("LEG2V [V]", "mains_1_voltage"))


        # 2
        if "gen [kW]" in df.columns:
            df = df.drop('gen [kW]', 1)

        # 3
        df = df.drop('Grid [kW]', 1)

        print df.columns

        # 4
        if "Grid* [kVA]" in df.columns:
            df = df.drop('Grid* [kVA]', 1)



        # 4
        df = df.rename(columns=lambda x: x.lower())

        # 5
        df = df.rename(columns=lambda x: x.replace(" ", "_"))

        # 6
        df = df.rename(columns=lambda x: x.replace("[kw]", "active"))
        df = df.rename(columns=lambda x: x.replace("[kva]", "apparent"))
        df = df.rename(columns=lambda x: x.replace("*", ""))

        # Find if voltage columns exist
        # 1. Multiply those back by 1e3
        # 2  Leg1 [V] to be replaced by mains_0_voltage
        # 3  Leg2 [V] to be replaces by mains_1_voltage






        # Create a new building
        b = Building()

        # Add mains DataFrame
        b.electric={}

        # Find columns containing mains in them
        mains_column_names = [x for x in df.columns if "mains" in x]
        b.electric["mains"] = df[mains_column_names]

        # Getting a list of appliance names
        appliance_names = list(set([a.split("_")[0] for a in df.columns \
        if "mains" not in a]))


        # Add appliances
        b.electric['appliances'] = {}
        for appliance in appliance_names:
            # Finding headers corresponding to the appliance
            names = [x for x in df.columns if x.split("_")[0]==appliance]
            b.electric['appliances'][appliance] = df[names]

        # Adding this building to dict of buildings
        building = building.replace(" ", "_")
        self.buildings[building] = b



    def load_building_names(self, directory):
        spreadsheet = pd.ExcelFile(directory + \
         "15_min/Homes 01-10_15min_2012-0819-0825 .xlsx")
        return spreadsheet.sheet_names









