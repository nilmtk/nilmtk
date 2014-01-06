"""
iAWE Dataset Loader
"""


import pandas as pd

from nilmtk.dataset import DataSet
from nilmtk.building import Building
from nilmtk.utils import get_immediate_subdirectories
from nilmtk.sensors.electricity import Measurement
from nilmtk.sensors.electricity import ApplianceName
from nilmtk.sensors.electricity import MainsName
import os
import MySQLdb
import pandas.io.sql as psql

mysql_conn = {}
mysql_conn['jplug'] = MySQLdb.connect(
    user='root', passwd='password', db='jplug')
mysql_conn['smart'] = MySQLdb.connect(
    user='root', passwd='password', db='smart_meter')

jplug_mapping = {

    '001EC00CC4A0': ApplianceName('fridge', 1),
    '001EC00CC49F': ApplianceName('ac', 1),
    '001EC00D7A1C': ApplianceName('ac', 2),
    '001EC00CC4AD': ApplianceName('washing_machine', 1),
    '001EC00D7A18': ApplianceName('laptop', 1),
    '001EC00CC49C': ApplianceName('iron', 1),
    '001EC00CC49D': ApplianceName('kitchen_misc', 1),
    '001EC00D7A1D': ApplianceName('television', 1),
    '001EC00E6BB6': ApplianceName('water_filter', 1),
    '001EC00E6BBD': ApplianceName('television', 1)
}

column_mapping = {
    'frequency': Measurement('frequency', ""),
    'voltage': Measurement('voltage', ""),
    'active_power': Measurement('power', 'active'),
    'energy': Measurement('energy', 'apparent'),
    'current': Measurement('current', ''),
    'reactive_power': Measurement('power', 'reactive'),
    'apparent_power': Measurement('power', 'apparent'),
    'power_factor': Measurement('pf', ''),
    'phase_angle': Measurement('phi', ''),
    'W1': Measurement('power', 'active'),
    'f': Measurement('frequency', ""),
    'VLN': Measurement('voltage', ""),
    'W2': Measurement('power', 'active')
}


def query_database_jplug(self, jplug):
    query = 'select active_power,voltage,timestamp from jplug_data where mac="%s";' % (
        jplug)
    data = psql.frame_query(query, mysql_conn['jplug'])
    data = data[data.timestamp < 1381069800]
    data.timestamp = data.timestamp.astype('int')
    data.index = pd.to_datetime(
        (data.timestamp.values * 1E9).astype(int), utc=True)
    data = data.drop('timestamp', 1)

    data = data.tz_convert(self.metadata['timezone'])
    # Get the data starting from June 7, 2013
    data = data[pd.Timestamp('2013-06-07'):pd.Timestamp('2014-01-01')]
    data = data.dropna()
    data = data.astype('float32')
    return data


class IAWE(DataSet):

    def __init__(self):
        super(IAWE, self).__init__()
        self.metadata = {
            'name': 'iAWE',
            'urls': ['http://www.energy.iiitd.edu.in/iawe'],
            'timezone': 'Asia/Kolkata'
        }
        self.building = Building()
        self.buildings[1] = self.building

    def load(self):
        """Load entire dataset into memory"""
        building = self.load_building_names(root_directory)
        for building_name in building_names:
            self.load_building(root_directory, building_name)

    def load_hdf5(self, directory):
        super(IAWE, self).load_hdf5(directory)

    def add_mains(self):
        query = 'select W1, W2, f, VLN, timestamp from smart_meter_data;'
        data = psql.frame_query(query, mysql_conn['smart'])
        data = data[data.timestamp < 1381069800]
        data.timestamp = data.timestamp.astype('int')
        data.index = pd.to_datetime(
            (data.timestamp.values * 1E9).astype(int), utc=True)
        data = data.drop('timestamp', 1)
        data = data.sort_index()
        data = data.tz_convert(self.metadata['timezone'])
        data = data[pd.Timestamp('2013-06-07'):pd.Timestamp('2014-01-01')]
        data = data.dropna()
        data = data.astype('float32')

        self.building.utility.electric.mains = {}
        self.building.utility.electric.mains[
            MainsName(1, 1)] = data[['W1', 'f', 'VLN']]
        self.building.utility.electric.mains[
            MainsName(1, 1)].rename(columns=lambda x: column_mapping[x], inplace=True)
        self.building.utility.electric.mains[
            MainsName(2, 1)] = data[['W2', 'f', 'VLN']]
        self.building.utility.electric.mains[
            MainsName(2, 1)].rename(columns=lambda x: column_mapping[x], inplace=True)

    def add_appliances(self):

        self.building.utility.electric.appliances = {}
        for jplug in jplug_mapping:
            print(jplug_mapping[jplug])
            if jplug_mapping[jplug] not in self.building.utility.electric.appliances.keys():
                self.building.utility.electric.appliances[
                    jplug_mapping[jplug]] = query_database_jplug(self, jplug)

            # Needed for appliances which are measured using multiple jPlugs at
            # different times
            else:
                self.building.utility.electric.appliances[jplug_mapping[jplug]] = pd.concat(
                    [self.building.utility.electric.appliances[jplug_mapping[jplug]], query_database_jplug(self, jplug)])

            # Sort values
            self.building.utility.electric.appliances[
                jplug_mapping[jplug]] = self.building.utility.electric.appliances[
                jplug_mapping[jplug]].sort_index()

        # Renaming measurements of columns
        for appliance in self.building.utility.electric.appliances.keys():
            self.building.utility.electric.appliances[appliance] = self.building.utility.electric.appliances[
                appliance].rename(columns=lambda x: column_mapping[x])

        # Adding motor data which was collected using Current Cost
        df = pd.read_csv('/home/nipun/Copy/motor_data_complete.csv',
                         names=['timestamp', 'power'])
        df.timestamp = df.timestamp.astype('int32')
        df.power = df.power.astype('int32')
        df.index = pd.to_datetime(
            (df.timestamp.values * 1E9).astype(int), utc=True)
        df = df.drop('timestamp', 1)
        df = df.sort_index()
        df = df.tz_convert(self.metadata['timezone'])

        # Filtering out insanely large values collected from some other
        # experiments
        df = df[df.power < 1000]
        df = df.dropna()

        # Renaming the column
        df.columns = [Measurement('power', 'active')]

        # Adding power to appliances
        self.building.utility.electric.appliances[
            ApplianceName('motor', 1)] = df
