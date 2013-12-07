from __future__ import print_function, division
import os
import glob
import pandas as pd
from nilmtk.dataset import DataSet
from nilmtk.utils import get_immediate_subdirectories
from nilmtk.building import Building

 # Column name mapping
column_name_mapping = {
        'V': 'voltage',
        'I': 'current',
        'f': 'frequency',
        'DPF': 'dpf',
        'APF': 'apf',
        'P': 'power_active',
        'Pt': 'energy_acive',
        'Q': 'power_reactive',
        'Qt': 'energy_reactive',
        'S': 'power_apparent',
        'St': 'energy_apparent'
        }

class AMPDS(DataSet):
    """Load data from AMPDS."""



    # Mapping of appliance names to CSV files containing them
    electricity_mapping = {

        }

    def __init__(self):
        super(AMPDS, self).__init__()
        self.urls = ['http://ampds.org/']
        self.citations = ['Stephen Makonin, Fred Popowich, Lyn Bartram, '
                        'Bob Gill, and Ivan V. Bajic,'
                        'AMPds: A Public Dataset for Load Disaggregation and'
                        'Eco-Feedback Research, in Electrical Power and Energy'
                        'Conference (EPEC), 2013 IEEE, pp. 1-6, 2013.'
                        ]

    def read_electricity_csv_and_standardize(self, csv_path):
        # Loading appliance
        df = pd.read_csv(csv_path)

        # Convert index to DateTime
        df.index = pd.to_datetime((df.TS.values * 1e9).astype(int))

        # Delete the TS column
        df = df.drop('TS', 1)

        # Rename columns
        df = df.rename(columns=lambda x: column_name_mapping[x])

        return df

    def load_electricity(self, root_directory):
        # Getting list of all the CSVs in the directory
        # Each appliance (or mains) has got its own CSV
        # Mains is named WHE.csv

        # Path to electricity folder
        electricity_folder = os.path.join(root_directory, 'electricity')
        # Borrowed from http://stackoverflow.com/a/8990026/743775
        if os.path.isdir(electricity_folder):
            electricity_folder = os.path.join(electricity_folder, "")
        print(electricity_folder)

        list_of_files = glob.glob("/%s*.csv" %electricity_folder)
        print(list_of_files)

        # Create new building
        building = Building()

        # Add mains
        building.utility.electric.mains = self.read_electricity_csv_and_standardize(
            os.path.join(electricity_folder, 'WHE.csv'))

        print(list_of_files)
        print(os.path.join(electricity_folder, 'WHE.csv'))
        #Deleting mains from list_of_files
        list_of_files.remove(os.path.join(electricity_folder, 'WHE.csv'))

        for csv_file in list_of_files:
            appliance_name = csv_file.split("/")[-1]
            building.utility.electric.appliance[appliance_name] = self.read_electricity_csv_and_standardize(csv_file)

    def load_water(self, root_directory):
        return None

    def load_gas(self, root_directory):
        return None

    def load(self, root_directory):
        return None




