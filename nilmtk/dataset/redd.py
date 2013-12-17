from __future__ import print_function, division
import re
import  os
import datetime
import pandas as pd
from nilmtk.dataset import DataSet, load_labels
from nilmtk.utils import get_immediate_subdirectories
from nilmtk.building import Building
from nilmtk.sensors.electricity import MainsName

def load_chan(building_dir, chan):
    """Returns DataFrame containing data for this channel"""
    filename = os.path.join(building_dir, 'channel_{:d}.dat'.format(chan))
    print('Loading', filename)
    date_parser = lambda x: datetime.datetime.utcfromtimestamp(x)
    df = pd.read_csv(filename, sep=' ', header=None, index_col=0,
                     parse_dates=True, date_parser=date_parser,
                     names=['active'], squeeze=True).astype('float32')


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
                          ' Data Mining Applications in Sustainability, 2011.']
        }

    def load_building(self, root_directory, building_name):
        # Construct new Building and set known attributes
        building = Building()
        # MIT's coorindates
        building.geographic_coordinates = (42.360091, -71.09416)

        # Load labels
        building_dir = os.path.join(root_directory, building_name)
        labels = load_labels(building_dir)

        # Load mains
        mains_chans = [chan for chan, label in labels.iteritems()
                       if label == 'mains']
        for mains_chan in mains_chans:
            col_name = MainsName(mains_chan, 1)
            df = load_chan(building_dir, mains_chan)
            df = df.tz_localize('UTC')
            df = df.tz_convert('US/Eastern')  # MIT is on the east coast
            building.utility.electric.mains[col_name] = df

        # Load sub metered channels
        # TODO
        # Convert from REDD channel names to standardised names
        # Set up wiring

        self.buildings[building_name] = building

    def load_building_names(self, root_directory):
        dirs = get_immediate_subdirectories(root_directory)
        pattern = re.compile('house_[0-9]*')
        dirs = [dir for dir in dirs if pattern.match(dir)]
        dirs.sort()
        return dirs
