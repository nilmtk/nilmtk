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

"""
MANUAL:

UKPD is a large dataset so this class provides some options to allow
you to load only a subset of the data.

For example, to only load house 1, and for the 1 sec mains to:
* only load active power
* downsample the 1sec mains

ukpd.load('/data/mine/vadeec/merged', 
           buildings_to_load=['house_1'], 
           one_sec_mains_params_to_load=['active'], 
           periods_to_load={1: ('2013-05-01', '2013-06-01')},
           downsample_one_sec_mains_rule='6S')
"""

"""
TODO:
* re-use more code from REDD
* put lighting_circuit into circuits
* set up wiring
* use correct measurements (some are apparent; some are active)
* convert to UKPD standard appliance names
* import metadata from house 1
* add citations to metadata
"""

# Maps from UKPD name to:
#   tuple : ('<nilmtk name>', <metadata dict>)

# TODO: fill in this map
APPLIANCE_NAME_MAP = {
    #    'oven': ApplianceMetadata('oven', {'fuel':'electricity', 'dualsupply': True}),
}

# maps from house number to a list of dud channel numbers
DUD_CHANNELS = {}

# load metadata

# Start and end times per building
DEFAULT_PERIODS_TO_LOAD = {1: ("2013-04-01", None)}

MIN_SAMPLES_TO_LOAD = 100

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

    def load_building(self, root_directory, building_name, 
                      periods_to_load=None, 
                      one_sec_mains_params_to_load=None, 
                      downsample_one_sec_mains_rule=None):
        """
        Parameters
        ----------
        periods_to_load : dict of tuples, optional
           Key of dict is the building number (int).
           Values are (<start date>, <end date>)
           e.g. ("2013-04-01", None) or ("2013-04-01", "2013-08-01")
           defaults to {1: ("2013-04-01", None)}
        one_sec_mains_params_to_load : list of strings, optional
            some combination of {'active', 'apparent', 'voltage'}
            Defaults to ['active', 'voltage']
        downsample_one_sec_mains_rule : string, optional
            How to download the 1-second mains data, if available.
            e.g. '6S'
            if None then no downsampling will be done on the 1-sec mains data.
        """

        if one_sec_mains_params_to_load is None:
            one_sec_mains_params_to_load = ['active', 'voltage']

        # Construct new Building and set known attributes
        building = Building()
        building.metadata['original_name'] = building_name
        electric = building.utility.electric

        # Load labels
        building_number = int(building_name[-1])
        building_dir = os.path.join(root_directory, building_name)
        labels = load_labels(building_dir)

        print("Loading building {:d}, orig name={}, path={}"
              .format(building_number, building_name, building_dir))

        # Process periods to load
        if periods_to_load is None:
            periods_to_load = DEFAULT_PERIODS_TO_LOAD

        start, end = periods_to_load.get(building_number, (None,None))
        if start or end:
            print("Will crop all channels for this building to start={}, end={}"
                  .format(start, end))

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

        def _pre_process_dataframe(df):
            df = df.tz_convert(self.metadata['timezone'])
            return df[start:end]

        # Load 1-second mains, if available
        usecols = []
        # columns in mains.dat are: index, active, apparent, voltage
        # usecols counts the index column as col 0
        if 'active' in one_sec_mains_params_to_load:
            usecols.append(1)
        if 'apparent' in one_sec_mains_params_to_load:
            usecols.append(2)
        if 'voltage' in one_sec_mains_params_to_load:
            usecols.append(3)
        try:
            df = load_chan(building_dir, filename='mains.dat', usecols=usecols,
                           colnames=[Measurement('power', 'active'), 
                                     Measurement('power', 'apparent'),
                                     Measurement('voltage', '')])
        except IOError:
            # some houses don't have 1-second mains
            pass
        else:
            df = _pre_process_dataframe(df)
            if downsample_one_sec_mains_rule:
                df = df.resample(rule=downsample_one_sec_mains_rule, how='mean')
            if len(df) > MIN_SAMPLES_TO_LOAD:
                electric.mains[MainsName(split=1, meter=1)] = df

        # Split channels into mains and appliances
        mains_chan = None
        appliance_chans = []
        for chan, label in labels.iteritems():
            if label == 'aggregate':
                mains_chan = chan
            else:
                appliance_chans.append(chan)


        # Load Current Cost mains chans (only if we haven't loaded 1sec mains)
        if mains_chan and electric.mains.get(MainsName(1,1)) is None:
            mainsname = MainsName(split=1, meter=1)
            df = load_chan(building_dir, mains_chan,
                           colnames=[Measurement('power', 'apparent')])
            df = _pre_process_dataframe(df)
            electric.mains[mainsname] = df

        # Load sub metered channels
        instances = {}
        # instances is a dict which maps:
        # {<'appliance name'>: <index of next appliance instance>}
        measurement = Measurement('power', 'active')
        for appliance_chan in appliance_chans:
            # Get appliance label and instance
            label = labels[appliance_chan]
            instance = instances.get(label, 1)
            appliancename = ApplianceName(name=label, instance=instance)
            instances[label] = instance + 1
            df = load_chan(building_dir, appliance_chan, colnames=[measurement])
            df = _pre_process_dataframe(df)
            df[measurement].name = appliancename
            if len(df) > MIN_SAMPLES_TO_LOAD:
                electric.appliances[appliancename] = df

        self.buildings[building_number] = building

    def load_building_names(self, root_directory):
        dirs = get_immediate_subdirectories(root_directory)
        pattern = re.compile('house_[0-9]*')
        dirs = [dir for dir in dirs if pattern.match(dir)]
        dirs.sort()
        return dirs
