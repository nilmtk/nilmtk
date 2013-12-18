from __future__ import print_function, division
import re
import os
import datetime
import pandas as pd
import numpy as np
from collections import namedtuple
from nilmtk.dataset import DataSet
from nilmtk.utils import get_immediate_subdirectories
from nilmtk.building import Building
from nilmtk.sensors.electricity import MainsName, Measurement, ApplianceName, MultiSupply

# Maps from REDD name to:
#   tuple : ('<nilmtk name>', <metadata dict>)
ApplianceMetadata = namedtuple('ApplianceMetadata', ['name', 'metadata'])
APPLIANCE_NAME_MAP = {
    'oven': ApplianceMetadata('oven', {'fuel':'electricity', 'multisupply': True}),
    'refrigerator': ApplianceMetadata('fridge', {}), 
    'dishwaser': ApplianceMetadata('dishwasher', {}),
    'kitchen_outlets': ApplianceMetadata('kitchen outlets', {}),
    'washer_dryer': ApplianceMetadata('washer dryer', {'multisupply': True}),
    'bathroom_gfi': ApplianceMetadata('bathroom misc', {}),
    'electric_heat': ApplianceMetadata('space heater', {'fuel':'electricity'}),
    'stove': ApplianceMetadata('hob', {'fuel':'electricity'})
}

# TODO: 
# Check that these multisupply==True appliances really are multisupply!


def load_chan(building_dir, chan, colname=Measurement('power','active')):
    """Returns DataFrame containing data for this channel"""
    filename = os.path.join(building_dir, 'channel_{:d}.dat'.format(chan))
    print('Loading', filename)
    # Don't use date_parser with pd.read_csv.  Instead load it all
    # and then convert to datetime.  Thanks to Nipun for linking to
    # this discussion where jreback gives this tip:
    # https://github.com/pydata/pandas/issues/3757
    df = pd.read_csv(filename, sep=' ', header=None, index_col=0,
                     parse_dates=False, names=[colname], 
                     dtype={colname:np.float32})
    df.index = pd.to_datetime((df.index.values*1E9).astype(int), utc=True)
    return df


def load_labels(data_dir):
    """Loads data from labels.dat file.

    Arguments
    ---------
    data_dir : str

    Returns
    -------
    labels : dict
        mapping channel numbers (ints) to appliance names (str)
    """
    filename = os.path.join(data_dir, 'labels.dat')
    with open(filename) as labels_file:
        lines = labels_file.readlines()

    labels = {}
    for line in lines:
        line = line.split(' ')
        # TODO add error handling if line[0] not an int
        labels[int(line[0])] = line[1].strip()

    return labels


class REDD(DataSet):
    """Load data from REDD."""

    def __init__(self):
        super(REDD, self).__init__()
        self.metadata = {
            'name': 'REDD',
            'full_name': 'Reference Energy Disaggregation Data Set',
            'urls': ['http://redd.csail.mit.edu'],
            'citations': ['J. Zico Kolter and Matthew J. Johnson.'
                          ' REDD: A public data set for energy disaggregation'
                          ' research. In proceedings of the SustKDD workshop on'
                          ' Data Mining Applications in Sustainability, 2011.'],
            'geographic_coordinates': (42.360091, -71.09416), # MIT's coorindates
            'timezone': 'US/Eastern' # MIT is on the east coast
        }

    def _pre_process_dataframe(self, df):
        df = df.sort_index() # raw REDD data isn't always sorted
        df = df.tz_convert(self.metadata['timezone'])
        return df

    def load_building(self, root_directory, building_name):
        # Construct new Building and set known attributes
        building = Building()

        # Load labels
        building_dir = os.path.join(root_directory, building_name)
        labels = load_labels(building_dir)

        # Convert appliance names from REDD to nilmtk standard names
        appliance_metadata = {}
        for chan, label in labels.iteritems():
            nilmtk_appliance = APPLIANCE_NAME_MAP.get(label)
            if nilmtk_appliance is not None:
                labels[chan] = nilmtk_appliance.name
                if nilmtk_appliance.metadata:
                    appliance_metadata[nilmtk_appliance.name] = nilmtk_appliance.metadata

        # Split channels into mains and appliances
        mains_chans = []
        appliance_chans = []
        for chan, label in labels.iteritems():            
            if label == 'mains':
                mains_chans.append(chan)
            else:
                appliance_chans.append(chan)

        # Load mains chans
        for mains_chan in mains_chans:
            mainsname = MainsName(split=mains_chan, meter=1)
            df = load_chan(building_dir, mains_chan)
            df = self._pre_process_dataframe(df)
            building.utility.electric.mains[mainsname] = df

        # Find MultiSupply appliances
        # These are identified as appliances where metadata['multisupply']==True
        # and where the channels are next to each other in the REDD labels.dat
        multisupply_chans = {}
        for i, chan in enumerate(appliance_chans[:-1]):
            label = labels[chan]
            next_chan = appliance_chans[i+1]
            next_label = labels[next_chan]
            metadata = appliance_metadata.get(label)
            if (label == next_label and metadata and metadata.get('multisupply')):
                multisupply_chans[chan] = 1
                multisupply_chans[next_chan] = 2

        # Load sub metered channels
        instances = {} # keep track of instances!
        measurement = Measurement('power', 'active')
        for appliance_chan in appliance_chans:
            # Get appliance label and instance
            label = labels[appliance_chan]
            instance = instances.get(label, 1)
            appliancename = ApplianceName(name=label, instance=instance)

            # Increment instance. Handle multisupply appliances
            multisupply = multisupply_chans.get(appliance_chan)
            if multisupply is None:
                # This is not a MultiSupply appliance
                instances[label] = instance + 1
                colname = measurement
                df = load_chan(building_dir, appliance_chan, colname)
                df = self._pre_process_dataframe(df)
                df[colname].name = appliancename
                building.utility.electric.appliances[appliancename] = df
            else:
                colname = MultiSupply(measurement, multisupply)
                df = load_chan(building_dir, appliance_chan, colname)
                df = self._pre_process_dataframe(df)
                df[colname].name = appliancename
                if multisupply == 1:
                    building.utility.electric.appliances[appliancename] = df
                else:
                    building.utility.electric.appliances[appliancename] = \
                     building.utility.electric.appliances[appliancename].join(df)
                    instances[label] = instance + 1            

            # TODO: put some channels into `circuits` if that's what we want to do

        # TODO
        # Store appliance_metadata for each appliance instance in electric.metadata['appliances']
        # Set up wiring

        self.buildings[building_name] = building

    def load_building_names(self, root_directory):
        dirs = get_immediate_subdirectories(root_directory)
        pattern = re.compile('house_[0-9]*')
        dirs = [dir for dir in dirs if pattern.match(dir)]
        dirs.sort()
        return dirs
