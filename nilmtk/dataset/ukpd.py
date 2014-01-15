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
from nilmtk.sensors.electricity import MainsName, Measurement, ApplianceName, DualSupply
from nilmtk.sensors.electricity import get_dual_supply_columns
from nilmtk.dataset.redd import load_chan, load_labels, ApplianceMetadata

# Maps from UKPD name to:
#   tuple : ('<nilmtk name>', <metadata dict>)

# TODO: fill in this map
APPLIANCE_NAME_MAP = {
    #    'oven': ApplianceMetadata('oven', {'fuel':'electricity', 'dualsupply': True}),
}

# maps from house number to a list of dud channel numbers
DUD_CHANNELS = {}

# load metadata


START = pd.Timestamp("2013-03-01")
END = pd.Timestamp("2013-07-01")


def _load_sometimes_unplugged(data_dir):
    """Loads data_dir/sometimes_unplugged.dat file and returns a
    list of strings.  Returns an empty list if file doesn't exist.
    """
    su_filename = os.path.join(data_dir, 'sometimes_unplugged.dat')
    try:
        file = open(su_filename)
    except IOError:
        return []
    else:
        lines = file.readlines()
        return [line.strip() for line in lines if line.strip()]


class UKPD(DataSet):

    """Load data from UKPD."""

    def __init__(self):
        super(UKPD, self).__init__()
        self.metadata = {
            'name': 'UKPD',
            'full_name': 'UK Power Dataset',
            'urls': ['http://www.doc.ic.ac.uk/~dk3810/data/'],
            # Imperial's coorindates
            'geographic_coordinates': (51.464462, -0.076544),
            'timezone': 'Europe/London'
            # TODO: citations
        }

    def _pre_process_dataframe(self, df):
        df = df.tz_convert(self.metadata['timezone'])
        
        return df

    def load_building(self, root_directory, building_name,
                      load_1_sec_mains_if_available=False):
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
                    appliance_metadata[
                        nilmtk_appliance.name] = nilmtk_appliance.metadata

        # Split channels into mains and appliances
        mains_chans = []
        appliance_chans = []
        for chan, label in labels.iteritems():
            if label == 'aggregate':
                mains_chans.append(chan)
            else:
                appliance_chans.append(chan)

        # Load mains chans
        for mains_chan in mains_chans:
            mainsname = MainsName(split=1, meter=1)
            df = load_chan(building_dir, mains_chan,
                           colnames=[Measurement('power', 'apparent')])
            df = self._pre_process_dataframe(df)
            building.utility.electric.mains[mainsname] = df[START:END]

        if load_1_sec_mains_if_available:
            # Load 1-second mains, if available
            try:
                df = load_chan(building_dir, filename='mains.dat',
                               colnames=[Measurement('power', 'active'),
                                         Measurement('power', 'apparent'),
                                         Measurement('voltage', '')])
            except IOError:
                # some houses don't have 1-second mains
                pass
            else:
                building.utility.electric.mains[
                    MainsName(split=1, meter=2)] = df

        # Load sub metered channels
        instances = {}
        # instances is a dict which maps:
        # {<'appliance name'>:
        #  (<index of next appliance instance>, <i of next supply>)}
        measurement = Measurement('power', 'active')
        for appliance_chan in appliance_chans:
            # Get appliance label and instance
            label = labels[appliance_chan]
            instance, supply = instances.get(label, (1, 1))
            appliancename = ApplianceName(name=label, instance=instance)
            instances[label] = (instance + 1, 1)
            colname = measurement
            df = load_chan(building_dir, appliance_chan, colnames=[colname])
            df = self._pre_process_dataframe(df)
            df[colname].name = appliancename
            building.utility.electric.appliances[appliancename] = df[START:END]

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
