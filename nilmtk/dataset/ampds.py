from __future__ import print_function, division
import os
import glob
import pandas as pd

from nilmtk.dataset import DataSet
from nilmtk.building import Building
from nilmtk.sensors.electricity import Measurement
from nilmtk.sensors.electricity import ApplianceName
from nilmtk.sensors.electricity import MainsName

from collections import namedtuple


# Column name mapping
column_name_mapping = {
    'V': Measurement('voltage', ''),
    'I': Measurement('current', ''),
    'f': Measurement('frequency', ''),
    'DPF': Measurement('pf', 'd'),
    'APF': Measurement('pf', 'a'),
    'P': Measurement('power', 'active'),
    'Pt': Measurement('energy', 'active'),
    'Q': Measurement('power', 'reactive'),
    'Qt': Measurement('energy', 'reactive'),
    'S': Measurement('power', 'apparent'),
    'St': Measurement('energy', 'apparent')
}

# Appliance name mapping
appliance_name_mapping = {
    'B1E': ApplianceName('misc', 1),
    'B2E': ApplianceName('misc', 2),
    'BME': ApplianceName('plugs', 1),
    'CDE': ApplianceName('dryer', 1),
    'CWE': ApplianceName('washer', 1),
    'DNE': ApplianceName('plugs', 2),
    'DWE': ApplianceName('dishwasher', 1),
    'EBE': ApplianceName('workbench', 1),
    'EQE': ApplianceName('network', 1),
    'FGE': ApplianceName('fridge', 1),
    'FRE': ApplianceName('thermostat', 1),
    'GRE': ApplianceName('misc', 3),
    'HPE': ApplianceName('heatpump', 1),
    'HTE': ApplianceName('water_heater', 1),
    'OFE': ApplianceName('misc', 4),
    'OUE': ApplianceName('plugs', 3),
    'TVE': ApplianceName('tv', 1),
    'UTE': ApplianceName('plugs', 4),
    'WOE': ApplianceName('oven', 1),
    'UNE': ApplianceName('unmetered', 1)
}


class AMPDS(DataSet):

    """Load data from AMPDS."""

    def __init__(self):
        super(AMPDS, self).__init__()
        self.urls = ['http://ampds.org/']
        self.citations = ['Stephen Makonin, Fred Popowich, Lyn Bartram, '
                          'Bob Gill, and Ivan V. Bajic,'
                          'AMPds: A Public Dataset for Load Disaggregation and'
                          'Eco-Feedback Research, in Electrical Power and Energy'
                          'Conference (EPEC), 2013 IEEE, pp. 1-6, 2013.'
                          ]
        self.building = Building()
        self.buildings['Building_1'] = self.building
        self.nominal_voltage = 230

    def read_electricity_csv_and_standardize(self, csv_path):
        # Loading appliance
        df = pd.read_csv(csv_path).astype('float32')

        # Convert index to DateTime
        df.index = pd.to_datetime((df.TS.values * 1e9).astype(int))

        # Delete the TS column
        df = df.drop('TS', 1)

        # Rename columns
        df = df.rename(columns=lambda x: column_name_mapping[x])

        return df

    def read_water_csv_and_standardize(self, csv_path):
        # Loading appliance
        df = pd.read_csv(csv_path).astype('float32')

        # Convert index to DateTime
        df.index = pd.to_datetime((df.ts.values * 1e9).astype(int))

        # Delete the TS column
        df = df.drop('ts', 1)

        # Rename columns
        #df = df.rename(columns=lambda x: column_name_mapping[x])

        return df

    def load_electricity(self, root_directory):

        # Path to electricity folder
        electricity_folder = os.path.join(root_directory, 'electricity')
        # Borrowed from http://stackoverflow.com/a/8990026/743775

        if os.path.isdir(electricity_folder):
            electricity_folder = os.path.join(electricity_folder, "")

        list_of_files = glob.glob("%s*.csv" % electricity_folder)

        # Add mains
        self.building.utility.electric.mains = {}
        self.building.utility.electric.mains[MainsName(1, 1)] = self.read_electricity_csv_and_standardize(
            os.path.join(electricity_folder, 'WHE.csv'))

        # Deleting mains from list_of_files
        list_of_files.remove(os.path.join(electricity_folder, 'WHE.csv'))

        self.building.utility.electric.appliances = {}
        for csv_file in list_of_files:
            appliance_name = appliance_name_mapping[
                csv_file.split("/")[-1][:3]]
            self.building.utility.electric.appliances[
                appliance_name] = self.read_electricity_csv_and_standardize(csv_file)

    def load_water(self, root_directory):
        # Path to water folder
        water_folder = os.path.join(root_directory, 'water')
        # Borrowed from http://stackoverflow.com/a/8990026/743775

        if os.path.isdir(water_folder):
            water_folder = os.path.join(water_folder, "")
        self.building.utility.water = {}
        self.building.utility.water["mains"] = self.read_water_csv_and_standardize(
            os.path.join(water_folder, 'WHW.csv'))
        self.building.utility.water["instant_heating"] = self.read_water_csv_and_standardize(
            os.path.join(water_folder, 'HTW.csv'))

    def load_gas(self, root_directory):
        # Path to natural gas folder
        gas_folder = os.path.join(root_directory, 'natural_gas')
        # Borrowed from http://stackoverflow.com/a/8990026/743775

        if os.path.isdir(gas_folder):
            gas_folder = os.path.join(gas_folder, "")
        self.building.utility.gas = {}
        self.building.utility.gas["mains"] = self.read_water_csv_and_standardize(
            os.path.join(gas_folder, 'WHG.csv'))
        self.building.utility.gas["hvac"] = self.read_water_csv_and_standardize(
            os.path.join(gas_folder, 'FRG.csv'))

    def load(self, root_directory):
        self.load_electricity(root_directory)
        self.load_water(root_directory)
        self.load_gas(root_directory)
