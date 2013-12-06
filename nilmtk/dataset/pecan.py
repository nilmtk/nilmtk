'''
Pecan Dataset Loader

These routines load Pecan Dataset into NILMTK Dataset format

Authors :
License:
'''

from .dataset import DataSet
import pandas as pd


class Pecan(DataSet):

    def __init__(self):
        super(Pecan, self).__init__()

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
        # 5. Appliance names should have separate active and apparent fields
        # (have a *)

        df.rename

        # self.buildings[building] = DataFrame storing building data
        raise NotImplementedError

    def load_building_names(self, directory):
        spreadsheet = pd.ExcelFile(directory + \
         "/15_Min/Homes 01-10_15min_2012-0819-0825 .xlsx")
        names = spreadsheet.sheet_names
        return [name.replace(" ", "_") for name in names]








