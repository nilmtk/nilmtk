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
        spreadsheet = pd.ExcelFile(directory)
        df = spreadsheet.parse(building, index_col=0, date_parser=True)

        # Converting power from kW to W
        df = df / 1e3

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

        # 2
        if "gen" in df.columns:
            df.drop('gen [kW]', 1)

        # 3
        df.drop('Grid [kW]', 1)

        # 4
        df = df.rename(columns=lambda x: x.lower())

        # 5
        df = df.rename(columns=lambda x: x.replace(" ", "_"))

        # 6
        df = df.rename(columns=lambda x: x.replace("[kW]", "active"))
        df = df.rename(columns=lambda x: x.replace("[kVA]", "apparent"))

        # Create a new building
        building = Building()

        # Add mains DataFrame
        building.electric.mains = df.mains_0_active

        # Getting a list of appliance names
        appliance_names = list(set([a.split("_")[0] for a in df.columns \
        if "mains" not in a]))

        # Add appliances
        building.electric.appliances = {}
        for appliance in appliance_names:
            building.electric.appliances[appliance] = df[[appliance + "_active", appliance + "_apparent"]]

        # Adding this building to dict of buildings
        self.buildings[building] = building

        return self.buildings

    def load_building_names(self, directory):
        spreadsheet = pd.ExcelFile(directory + \
         "/15_Min/Homes 01-10_15min_2012-0819-0825 .xlsx")
        names = spreadsheet.sheet_names
        return [name.replace(" ", "_") for name in names]








