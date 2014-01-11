from __future__ import print_function, division
import pandas as pd
from os.path import join
from datetime import datetime
from nilmtk.dataset import DataSet
from nilmtk.building import Building

"""
HES notes

* 14 homes recorded mains but only 5 were kept after cleaning
* circuit-level data from the consumer unit was recorded as 'sockets' for 216 houses
* 'total_profiles.csv' records pairs of <house>,<appliance> which are the 
  channels which need to be added to produce the whole-home total, which
  I think consists of all the circuit-level meters plus all appliances
  which are not also monitored at circuit level.
* appliance 2000 represents the calculated aggregate
* appliance 159 represents the difference between
  this and the sum of the known appliances
* appliance_codes.csv maps from <appliance code> to <appliance name>
* seasonal_adjustments.csv stores the trends in energy usage per appliance 
  activation over a year.
"""

CHUNKSIZE = 1E5 # number of rows
COL_NAMES = ['interval id', 'house id', 'appliance code', 'date',
             'energy', 'time']
LAST_PWR_COLUMN = 250
NANOSECONDS_PER_TENTH_OF_AN_HOUR = 1E9 * 60 * 6


def datetime_converter(datetime_str):
    return datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')


def load_list_of_houses(data_dir):
    """Returns a list of house IDs in HES (ints)."""
    filename = join(data_dir, 'ipsos-anonymised-corrected 310713.csv')
    series = pd.read_csv(filename, usecols=[0], index_col=False, squeeze=True)
    return series.tolist()


class HES(DataSet):
    """Load data from UK Government's Household Electricity Survey 
    (the cleaned version of the dataset released in summer 2013).
    """

    # TODO: re-use code from 
    # https://github.com/JackKelly/pda/blob/master/scripts/hes/load_hes.py

    """
    Broad approach:
    * load list of houses
    * create dataset.buildings dict with empty Buildings
    * the keys of `dataset.buildings` are the HES house IDs, which
      will be converted to the nilmtk standard after loading.

    * load CHUNK_SIZE of data from CSV into a DataFrame
    * convert datetime
    * get list of houses in the DF
    * for each house:
        * Load previously converted data from the HDFStore
        * append new data
        * save back to HDFStore

    * When all houses are complete, post-process:
        * convert energy to nilmtk standard energy unit (kWh?)
        * convert deci-kWh to watts (retain energy) 
          (see 'convert_hes_to_watts.py' from pda)
        * convert keys of `dataset.buildings`
        * convert keys of `electric.appliances`
    """

    def __init__(self):
        super(HES, self).__init__()

    def load(self, data_dir):
        # load list of houses
        house_ids = load_list_of_houses(data_dir)
        for house_id in house_ids:
            building = Building()
            building.metadata['original_name'] = house_id
            self.buildings[house_id] = building

        # Load appliance energy data chunk-by-chunk
        filename = join(data_dir, 'appliance_group_data-1a.csv')
        reader = pd.read_csv(filename, names=COL_NAMES, 
                             index_col=False, chunksize=CHUNKSIZE)
        for chunk in reader:
            # Convert date and time columns to np.datetime64 objects
            dt = chunk['date'] + ' ' + chunk['time']
            del chunk['date']
            del chunk['time']
            chunk['datetime'] = dt.apply(datetime_converter)
            
            houses_in_chunk = chunk['house id'].unique()
            for house_id in houses_in_chunk:
                building = self.buildings[house_id]
                electric = building.utility.electric
                for appliance_code in appliances_in_house_chunk:
                    pass
                # TODO

    def load_building(self, filename, building_name):
        raise NotImplementedError

    def load_building_names(self, filename):
        raise NotImplementedError
