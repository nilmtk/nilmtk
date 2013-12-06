from __future__ import print_function, division
import re, os, datetime
import pandas as pd
from nilmtk.dataset import DataSet, load_labels
from nilmtk.utils import get_immediate_subdirectories
from nilmtk.building import Building

def load_chan(building_dir, chan):
    """Returns DataFrame containing data for this channel"""
    filename = os.path.join(building_dir, 'channel_{:d}.dat'.format(chan))
    print('Loading', filename)
    date_parser = lambda x: datetime.datetime.utcfromtimestamp(x)
    return pd.read_csv(filename, sep=' ', header=None, index_col=0,
                       parse_dates=True, date_parser=date_parser,
                       names=['active'], squeeze=True)

class REDD(DataSet):
    """Load data from REDD."""

    def __init__(self):
        super(REDD, self).__init__()
        self.urls = ['http://redd.csail.mit.edu']
        self.citations = ['J. Zico Kolter and Matthew J. Johnson.'
                          ' REDD: A public data set for energy disaggregation'
                          ' research. In proceedings of the SustKDD workshop on'
                          ' Data Mining Applications in Sustainability, 2011.']

    def load_building(self, root_directory, building_name):
        # Construct new Building and set known attributes
        building = Building()
        building.geographic_coordinates = (42.360091, -71.09416) # MIT's coorindates

        # Load labels
        building_dir = os.path.join(root_directory, building_name)
        labels = load_labels(building_dir)

        # Load mains
        mains_chans = [chan for chan, label in labels.iteritems()
                       if label == 'mains']
        mains_chan_dict = {}
        for mains_chan in mains_chans:
            col_name = 'mains_{:d}_meter_1_active'.format(mains_chan)
            mains_chan_dict[col_name] = load_chan(building_dir, mains_chan)

        # Make a DataFrame containing all mains channels
        df = pd.DataFrame(mains_chan_dict)
        df = df.tz_localize('UTC')
        df = df.tz_convert('US/Eastern') # MIT is on the east coast!
        building.electric.mains = df

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
