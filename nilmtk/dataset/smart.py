from __future__ import print_function, division
import re
import pandas as pd
import glob
import os


from nilmtk.sensors.electricity import CircuitName, ApplianceName, Measurement, MainsName
from nilmtk.building import Building
from nilmtk.dataset import DataSet
from nilmtk.utils import get_immediate_subdirectories


# Column headers for circuits data (including mains)
names_circuits = ['CircuitName', 'CircuitNumber', 'TimestampUTC',
                  'RealPowerWatts', 'ApparentPowerVAs']

names_meter = ['MeterName', 'TimestampUTC', 'RealPowerWatts', 'CircuitNumber'
               ]

# Mapping from smart* to nilmtk
MAINS_NAME_MAP = {
    'grid': MainsName(1, 1)
}

APPLIANCE_NAME_MAP = {
    'ac': ApplianceName('air conditioner', 1),
    'toaster': ApplianceName('toaster', 1),
    'waterpump': ApplianceName('water pump', 1),
    'nitelight': ApplianceName('light', 4),
    'laptop': ApplianceName('computer laptop', 1),
    'telephone': ApplianceName('phone', 1),
    'wii': ApplianceName('entertainment', 1),
    'batteries': ApplianceName('misc', 1),
    'printer': ApplianceName('printer', 1),
    'video': ApplianceName('entertainment unit', 1),
    'lamp2': ApplianceName('light', 3),
    'lamp1': ApplianceName('light', 2),
    'tassimo': ApplianceName('misc', 2),
    'timemachine': ApplianceName('misc', 3),
    'microwave': ApplianceName('microwave', 1),
    'monitor': ApplianceName('monitor', 1),
    'clock': ApplianceName('misc', 4),
    'tv': ApplianceName('entertainment unit', 2),
    'hrv': ApplianceName('hrv', 1),
    'coffeepot': ApplianceName('coffee maker', 1),
    'freezer': ApplianceName('fridge', 1),
    'lamp': ApplianceName('light', 1),
    'dehumidifier': ApplianceName('dehumidifier', 1),
    'noise': ApplianceName('misc', 5),
    'ipod': ApplianceName('entertainment unit', 3),
    'subwoofer': ApplianceName('entertainment unit', 4),
    'tivo': ApplianceName('misc', 6),
    'clock1': ApplianceName('misc', 7),
    'macmini': ApplianceName('computer desktop', 1),
    'fan': ApplianceName('fan', 1),
    'cablebox': ApplianceName('entertainment unit', 4),
    'desklamp': ApplianceName('light', 5),
    'refrigerator': ApplianceName('refrigerator', 2),
    'cablemodem': ApplianceName('entertainment unit', 5),
    'nightstand2': ApplianceName('misc', 8),
    'washingmachine': ApplianceName('washing machine', 1),
    'dvd': ApplianceName('entertainment', 5),
    'iphone1': ApplianceName('misc', 9),
    'roku': ApplianceName('misc', 10),
    'panini': ApplianceName('misc', 11),
    'receiver': ApplianceName('misc', 12),
    'clockphone': ApplianceName('misc', 13)
}


def find_appliance_nilmtk_name(appliance_name):
    return APPLIANCE_NAME_MAP[appliance_name.split(":")[1]]


def circuit_name_mapping(circuit_name):
    return CircuitName(circuit_name, 1, 1)

# Building name mapping
building_name_mapping = {
    'A': 1,
    'B': 2,
    'C': 3
}


def load_labels(data_dir):
    """
        Uses unique entries in one of csv file to obtain labels
        Arguments
        ---------
        data_dir : str

        Returns
        -------
        labels : List
            of circuit names
    """
    # List of csvs in the directory
    list_csv = glob.glob(data_dir + '/*-circuit/*.csv')
    # Reaaing first CSV
    df = pd.read_csv(list_csv[0], header=None,
                     names=names_circuits, index_col=2)
    # Assuming first CSV has all appliances
    labels = list(df.CircuitName.unique())
    return labels


class Smart(DataSet):

    """Load data from Smart*"""

    def __init__(self):
        super(Smart, self).__init__()
        self.metadata = {
            'name': 'Smart*',
            'full_name': 'UMass Smart* Home Data Set',
            'urls': ['http://traces.cs.umass.edu/index.php/Smart/Smart'],
            #'citations':
            #'geographic_coordinates': (, )
            'timezone': 'US/Eastern'
        }

    def _pre_process_dataframe(self, df):
        df = self._drop_circuit_number(df)
        df = df.sort_index()  # raw data isn't always sorted
        df = df.tz_convert(self.metadata['timezone'])
        df = self._rename_columns(df)
        # COnvert to 32bits
        #df = df.astype('float32')
        return df

    def _rename_columns(self, df):
        """Rename columns from smart* to nilmtk"""
        return df.rename(
            columns={'RealPowerWatts': Measurement('power', 'active'),
                     'ApparentPowerVAs': Measurement('power', 'apparent')})

    def _drop_circuit_number(self, df):
        """Drops circuit number and name from columns"""
        if "CircuitName" in df.columns:
            df = df.drop("CircuitNumber", 1)
            df = df.drop("CircuitName", 1)
        if "MeterName" in df.columns:
            df = df.drop("MeterName", 1)
            df = df.drop("CircuitNumber", 1)
        return df

    def load_building(self, root_directory, building_name):
        building = Building()
        building_dir = os.path.join(root_directory, building_name)
        # Finds house number i.e. A, B or C; then maps it to 1,2,3
        building_name = re.search(
            'home(?P<building_name>\w+)', building_dir).group('building_name')
        labels = load_labels(building_dir)

        # --------Loading data from circuits- mains and circuit------#

        # Finding all CSVs
        list_csv = glob.glob(building_dir + '/*-circuit/*.csv')
        # List of dataframes
        df_list = []
        # Iterating over all CSVs and appending to list of dataframes
        for csv in list_csv:
            df = pd.read_csv(csv, header=None,
                             names=names_circuits, index_col=2)
            # Converting the index to DatetimeIndex
            df.index = pd.to_datetime(
                (df.index.values * 1E9).astype(int), utc=True)
            # Append to list of dfs
            df_list.append(df)

        # Merging all dfs
        combined_df = pd.concat(df_list)

        # Getting all the mains data
        mains_df = combined_df[combined_df["CircuitName"] == "Grid"]
        mains_df = self._pre_process_dataframe(mains_df)
        building.utility.electric.mains[MainsName(1, 1)] = mains_df

        # Dropping data from combined_df which contained information about
        # mains
        combined_df = combined_df[combined_df["CircuitName"] != "Grid"]

        # Grouping circuits data
        circuit_df_groups = combined_df.groupby('CircuitName')
        for name, df in circuit_df_groups:
            # Getting nilmtk name
            circuit_name = circuit_name_mapping(name)
            df = self._pre_process_dataframe(df)
            building.utility.electric.circuits[circuit_name] = df

        # --------Loading data from meter------#
        list_csv = glob.glob(building_dir + '/*-meter/*.csv')
        # List of dataframes
        df_list = []
        # Iterating over all CSVs and appending to list of dataframes
        for csv in list_csv:
            df = pd.read_csv(csv, header=None,
                             names=names_meter, index_col=1)
            # Converting the index to DatetimeIndex
            df.index = pd.to_datetime(
                (df.index.values * 1E9).astype(int), utc=True)
            # Append to list of dfs
            df_list.append(df)

        # Merging all dfs
        combined_df = pd.concat(df_list)

        # Grouping appliance data
        appliance_df_groups = combined_df.groupby('MeterName')
        for name, df in appliance_df_groups:
            print(name)
            # Getting nilmtk name
            appliance_name = find_appliance_nilmtk_name(name)
            df = self._pre_process_dataframe(df)
            building.utility.electric.appliances[appliance_name] = df

        # Storing building in buildings dictionary
        building_number = building_name_mapping[building_name]
        self.buildings[building_number] = building

    def load_building_names(self, root_directory):
        dirs = get_immediate_subdirectories(root_directory)
        pattern = re.compile('home[A-Z].*')
        dirs = [dir for dir in dirs if pattern.match(dir)]
        dirs.sort()
        return dirs
