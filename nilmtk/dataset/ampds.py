from __future__ import print_function, division
import os
import glob
import pandas as pd
from nilmtk.dataset import DataSet
from nilmtk.utils import get_immediate_subdirectories
from nilmtk.building import Building


class AMPDS(DataSet):
    """Load data from AMPDS."""

    # Column name mapping
    column_name_mapping = {
        'V': 'voltage',
        'I': 'current',
        'f': 'frequency',
        'DPF': 'dpf',
        'APF': 'apf',
        'P': 'active_power',
        'Pt': 'active_energy',
        'Q': 'reactive_power',
        'Qt':
        }

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

    def load_electricity(self, root_directory):
        # Getting list of all the CSVs in the directory
        # Each appliance (or mains) has got its own CSV
        # Mains is named WHE.csv
        list_of_files = glob.glob("%s*.csv" % root_directory)

        # Path to electricity folder
        electricity_folder = os.path.join(root_directory, 'electricity')

        # Loading mains
        df = pd.read_csv(os.path.join(electricity_folder, 'WHE.csv'))

        # Convert index to DateTime
        df.index = pd.to_datetime((df.TS.values * 1e9).astype(int))

        # Delete the TS column
        df = df.drop('TS', 1)

        # Rename columns

        for csv_file in list_of_files:


        return None

    def load_water(self, root_directory):
        return None

    def load_gas(self, root_directory):
        return None

    def load(self, root_directory):
        return None




