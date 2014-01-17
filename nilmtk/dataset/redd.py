from __future__ import print_function, division
import re
import os
import datetime
import sys
import pandas as pd
import numpy as np
from collections import namedtuple
from nilmtk.dataset import DataSet
from nilmtk.utils import get_immediate_subdirectories
from nilmtk.building import Building
from nilmtk.sensors.electricity import MainsName, Measurement, ApplianceName, DualSupply
from nilmtk.sensors.electricity import get_dual_supply_columns

# Maps from REDD name to:
#   tuple : ('<nilmtk name>', <metadata dict>)
ApplianceMetadata = namedtuple('ApplianceMetadata', ['name', 'metadata'])
APPLIANCE_NAME_MAP = {
    'oven': ApplianceMetadata('oven', {'fuel':'electricity', 'dualsupply': True}),
    'refrigerator': ApplianceMetadata('fridge', {}), 
    'dishwaser': ApplianceMetadata('dishwasher', {}),
    'kitchen_outlets': ApplianceMetadata('kitchen outlets', {}),
    'washer_dryer': ApplianceMetadata('washer dryer', {'dualsupply': True}),
    'bathroom_gfi': ApplianceMetadata('bathroom misc', {}),
    'electric_heat': ApplianceMetadata('space heater', {'fuel':'electricity'}),
    'stove': ApplianceMetadata('hob', {'fuel':'electricity'})
}

# TODO: 
# Check that these dualsupply==True appliances really are dualsupply!

# maps from house number to a list of dud REDD channel numbers
DUD_CHANNELS = {1: [19]}

def load_chan(building_dir, chan=None, filename=None, colnames=None, 
              usecols=None, sep=' '):
    """Loads CSV files where the first column is a UNIX timestamp, 
    like REDD or UKPD CSV files.

    Parameters
    ----------
    building_dir : string
        The base path
    chan : int, optional
        filename will be formed from 'building_dir/channel_<chan>.dat'
    filename : string, optinal
        if you want to load a filename not of the form `channel_<chan>.dat` then
        leave `chan` as None and provide just the `filename`, no path.
    colnames : list, optional
        The names to give to each column
    usecols : list of ints, optional
        A list of column indicies to load.  If colnames is provided then
        len(colnames) == len(usecols)
    sep : character, optional
        Defaults to ' '

    Returns 
    -------
    DataFrame.  Index is DatetimeIndex in UTC.  Data values are float32.  
        Column names are `colnames` if provided.
    """
    if colnames is None:
        colnames = [Measurement('power','active')]
    
    if filename is None:
        filename = os.path.join(building_dir, 'channel_{:d}.dat'.format(chan))
    else:
        filename = os.path.join(building_dir, filename)

    print('Attempting to load', filename, '...', end='')
    if usecols:
        usecols.insert(0,0)
        if colnames:
            colnames.insert(0, 'index')
        print("Only using columns", usecols, '...', end='')
    sys.stdout.flush()
    # Don't use date_parser with pd.read_csv.  Instead load it all
    # and then convert to datetime.  Thanks to Nipun for linking to
    # this discussion where jreback gives this tip:
    # https://github.com/pydata/pandas/issues/3757
    try:
        df = pd.read_csv(filename, sep=sep, header=None, index_col=0,
                         parse_dates=False, names=colnames, usecols=usecols,
                         dtype={colname:np.float32 for colname in colnames
                                if colname != 'index'})
    except Exception as e:
        print('failed:', str(e))
        raise
    else:
        print('done.')
        df.index = pd.to_datetime((df.index.values*1E9).astype(int), utc=True)
    return df


def load_labels(data_dir):
    """Loads data from labels.dat file.

    Parameters
    ----------
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
        building.metadata['original_name'] = building_name

        # Load labels
        building_number = int(building_name[-1])
        building_dir = os.path.join(root_directory, building_name)
        labels = load_labels(building_dir)

        # Remove dud channels
        try:
            dud_channels_for_building = DUD_CHANNELS[building_number]
        except KeyError:
            # DUD_CHANNELS doesn't specify dud channels for all buildings
            pass
        else:
            for dud_chan in dud_channels_for_building:
                labels.pop(dud_chan)

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
            df = load_chan(building_dir, mains_chan, colnames=[Measurement('power', 'apparent')])
            df = self._pre_process_dataframe(df)
            building.utility.electric.mains[mainsname] = df

        # Load sub metered channels
        instances = {} 
        # instances is a dict which maps:
        # {<'appliance name'>: 
        #  (<index of next appliance instance>, <i of next supply>)}
        measurement = Measurement('power', 'active')
        for appliance_chan in appliance_chans:
            # Get appliance label and instance
            label = labels[appliance_chan]
            instance, supply = instances.get(label, (1,1))
            appliancename = ApplianceName(name=label, instance=instance)
            metadata = appliance_metadata.get(label)
            is_dualsupply = metadata and metadata.get('dualsupply')
            if is_dualsupply:
                colname = DualSupply(measurement, supply)
                df = load_chan(building_dir, appliance_chan, colname)
                df = self._pre_process_dataframe(df)
                df[colname].name = appliancename
                if supply == 1:
                    building.utility.electric.appliances[appliancename] = df
                    instances[label] = (instance, supply + 1)
                else:
                    building.utility.electric.appliances[appliancename] = \
                     building.utility.electric.appliances[appliancename].join(df)
                    instances[label] = (instance + 1, 1)
            else:
                # This is not a DualSupply appliance
                instances[label] = (instance + 1, 1)
                colname = measurement
                df = load_chan(building_dir, appliance_chan, colnames=[colname])
                df = self._pre_process_dataframe(df)
                df[colname].name = appliancename
                building.utility.electric.appliances[appliancename] = df


        # Now go through all DualSupply appliances to make sure there are two chans
        appliances = building.utility.electric.appliances
        for appliance_name, appliance_df in appliances.iteritems():
            dual_supply_columns = get_dual_supply_columns(appliance_df)
            n_dual_supply_columns = len(dual_supply_columns)
            if n_dual_supply_columns == 1:
                col = dual_supply_columns[0]
                print("converting", appliance_name, "in building", building_number)
                appliances[appliance_name].rename(columns={col:col.measurement},
                                                  inplace=True)

        # TODO
        # Store appliance_metadata for each appliance instance in electric.metadata['appliances']
        # Set up wiring

        self.buildings[building_number] = building

    def load_building_names(self, root_directory):
        dirs = get_immediate_subdirectories(root_directory)
        pattern = re.compile('house_[0-9]*')
        dirs = [dir for dir in dirs if pattern.match(dir)]
        dirs.sort()
        return dirs
