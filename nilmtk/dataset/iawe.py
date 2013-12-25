'''
iAWE Dataset Loader

These routines load iAWE Dataset into NILMTK Dataset format

Authors :
License:


'''

'''
Geyser did not work!- So '001EC00CC4A1' is ignored
'''


'''
Iron June 6-
kitchen_misc June 6-
AC June 1
Fridge June 6
washing_machine June 1
laptop *
AC 2 June 6
TV from 001EEC00D7A1D June 13 to June 23; 
water_filter July 13
TV from 001EC00E6BBD July 10 to ..

'''


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
mysql_conn['jplug'] =MySQLdb.connect(
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


def query_database_jplug(jplug):
    query = 'select active_power,timestamp from jplug_data where mac="%s";' % (
        jplug)
    data = psql.frame_query(query, mysql_conn['jplug'])
    data = data.astype('float32')
    print 'Now converting index'
    data = data[data.timestamp < 1381069800]
    data.timestamp = data.timestamp.astype('int')
    data.index = pd.to_datetime((data.timestamp.values * 1E9).astype(int))
    data = data.drop('timestamp', 1)
    # Get the data starting from June 7, 2013
    data = data[pd.Timestamp('2013-06-07'):pd.Timestamp('2014-01-01')]
    print 'Now downsampling'
    print data.describe()
    print jplug_mapping[jplug]
    print "*" * 80
    # data.resample('1Min')
    return data


class IAWE(DataSet):

    def __init__(self):
        super(IAWE, self).__init__()
        self.metadata = {
            'name': 'iAWE',
            'urls': ['http://www.energy.iiitd.edu.in/iawe']}
        self.building = Building()
        self.buildings['Home_01'] = self.building

    def load(self):
        """Load entire dataset into memory"""
        building = self.load_building_names(root_directory)
        print (building_names)
        for building_name in building_names:
            self.load_building(root_directory, building_name)

    def add_mains(self):
        query = 'select W1, W2, f, VLN, timestamp from smart_meter_data ;'
        data = psql.frame_query(query, mysql_conn['smart'])
        data = data.astype('float32')
        print 'Now converting index'
        data = data[data.timestamp < 1381069800]
        data.timestamp = data.timestamp.astype('int')
        data.index = pd.to_datetime((data.timestamp.values * 1E9).astype(int))
        data = data.drop('timestamp', 1)
        data = data[pd.Timestamp('2013-06-07'):pd.Timestamp('2014-01-01')]

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
            print jplug
            if jplug_mapping[jplug] not in self.building.utility.electric.appliances.keys():
                self.building.utility.electric.appliances[
                    jplug_mapping[jplug]] = query_database_jplug(jplug)

                print self.building.utility.electric.appliances[
                    jplug_mapping[jplug]]
            # Needed for appliances which are measured using multiple jPlugs at
            # different times
            else:
                self.building.utility.electric.appliances[jplug_mapping[jplug]] = pd.concat(
                    [self.building.utility.electric.appliances[jplug_mapping[jplug]], query_database_jplug(jplug)])

        # Drop cost and mac columns from the data
        '''for appliance in self.building.utility.electric.appliances.keys():
            self.building.utility.electric.appliances[
                appliance].drop(['Cost', 'mac'], 1, inplace=True)
        '''
        # Renaming measurements of columns
        for appliance in self.building.utility.electric.appliances.keys():
            self.building.utility.electric.appliances[appliance] = self.building.utility.electric.appliances[
                appliance].rename(columns=lambda x: column_mapping[x])

        # Setting precision as 32 bits to save memory
        '''for appliance in self.building.utility.electric.appliances.keys():
            self.building.utility.electric.appliances[
                appliance] = self.building.utility.electric.appliances[appliance].astype('float32')
        '''
