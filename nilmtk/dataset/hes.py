from __future__ import print_function, division
import pandas as pd
import numpy as np
from os.path import join
from datetime import datetime
from pytz import UTC
from sys import stderr
from nilmtk.dataset import DataSet
from nilmtk.building import Building
from nilmtk.sensors.electricity import Measurement, MainsName, CircuitName

"""
TODO
----

* convert HES appliance names to NILMTK standard
* what exactly is measured? Real power? Apparent?
* houses which have multiple mains: are they multiple 'splits' or phases or meters?
* dataset metadata
* some houses have both 2- and 10-minute data.  Might need a function to ignore 10 minute data.
* set up wiring to take into consideration the information in 
  'total_profiles.csv'  Sockets 1-11 are circuits monitored
  at the consumer unit which feed fall sockets around the dwelling.
* import the enormous amount of appliance metadata in 'appliance_data.csv', 
  especially channels which recorded multiple appliances
* use the metadata in 'ipsos.csv' and 'rdsap_data.csv' and 'rdsap_*.csv' for each Building
* Maybe email CAR to let them know that nilmtk can now import HES.

HES notes
---------

* 14 homes recorded mains but only 5 were kept after cleaning
* circuit-level data from the consumer unit was recorded as 'sockets' for 216 houses
* 'total_profiles.csv' records pairs of <house>,<appliance> which are the 
  channels which need to be added to produce the whole-home total, which
  I think consists of all the circuit-level meters plus all appliances
  which are not also monitored at circuit level.
* appliance 2000 represents the calculated aggregate ???
* appliance 159 represents the difference between ???
  this and the sum of the known appliances
* appliance_codes.csv maps from <appliance code> to <appliance name>
* seasonal_adjustments.csv stores the trends in energy usage per appliance 
  activation over a year.

"""

FILENAMES = ['appliance_group_data-{}.csv'.format(s) for s in
             ['1a','1b','1c','1d','2','3']]
CHUNKSIZE = 1E5 # number of rows
COL_NAMES = ['interval id', 'house id', 'appliance code', 'date',
             'data', 'time']
LAST_PWR_COLUMN = 250
NANOSECONDS_PER_TENTH_OF_AN_HOUR = 1E9 * 60 * 6
MAINS_CODES = [240, 241]
TEMPERATURE_CODES = range(251,256)
CIRCUIT_CODES = range(208, 218) + [222]
E_MEASUREMENT = Measurement('energy', 'active')


def datetime_converter(s):
    """
    Parameters
    ----------
    s : int
        of the form 2011-02-02 14:48:00

    Returns
    -------
    datetime
    """
    # 0123456789012345678
    # 2011-02-02 14:48:00
    # the code below is ~8 times faster 
    # than datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    return datetime(year=int(s[0:4]), month=int(s[5:7]), day=int(s[8:10]),
                    hour=int(s[11:13]), minute=int(s[14:16]), 
                    second=int(s[17:19]), tzinfo=UTC)

def load_list_of_house_ids(data_dir):
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
        * sort all indicies
        * set timezone
        * convert energy to nilmtk standard energy unit (kWh?)
        * convert Wh to watts (retain energy) 
          (see 'convert_hes_to_watts.py' from pda)
        * convert keys of `dataset.buildings`
    """

    def __init__(self):
        super(HES, self).__init__()
        self.metadata = {
            'name': 'HES',
            'geographic_coordinates': (51.464462,-0.076544), # London
            'timezone': 'Europe/London'
        }

    def load(self, data_dir, max_chunks=None):
        # load list of houses
        house_ids = load_list_of_house_ids(data_dir)
        for house_id in house_ids:
            building = Building()
            building.metadata['original_name'] = house_id
            self.buildings[house_id] = building

        houses_loaded = set()

        for filename in FILENAMES:
            # Load appliance energy data chunk-by-chunk
            full_filename = join(data_dir, filename)
            print('loading', full_filename)
            try:
                reader = pd.read_csv(full_filename, names=COL_NAMES, 
                                     index_col=False, chunksize=CHUNKSIZE)
            except IOError as e:
                print(e, file=stderr)
                continue

            # Process each chunks
            chunk_i = 0
            for chunk in reader:
                if max_chunks is not None and chunk_i >= max_chunks:
                    break

                print('processing chunk', chunk_i, 'of', filename)
                # Convert date and time columns to np.datetime64 objects
                dt = chunk['date'] + ' ' + chunk['time']
                del chunk['date']
                del chunk['time']
                chunk['datetime'] = dt.apply(datetime_converter)

                # Data is either tenths of a Wh or tenths of a degree
                chunk['data'] *= 10
                chunk['data'] = chunk['data'].astype(np.float32)

                # Process each house in chunk
                houses_in_chunk = chunk['house id'].unique() #TODO: use groupby?!?
                houses_loaded = houses_loaded.union(set(houses_in_chunk))
                for house_id in houses_in_chunk:
                    self._process_house_in_chunk(house_id, chunk)

                chunk_i += 1
        print('houses with some data loaded:', houses_loaded)

    def _process_house_in_chunk(self, house_id, chunk):
        building = self.buildings[house_id]
        electric = building.utility.electric
        house_data = chunk[chunk['house id'] == house_id]
        appliances_in_house_chunk = house_data['appliance code'].unique()
        for appliance_code in appliances_in_house_chunk:
            # TODO: do this using Pandas groupby???
            appliance_data = house_data[house_data['appliance code'] == 
                                        appliance_code]
            data = appliance_data['data'].values
            index = appliance_data['datetime']
            df = pd.DataFrame(data=data, index=index,
                              columns=[E_MEASUREMENT])

            is_temperature = False
            if appliance_code in MAINS_CODES:
                dict_ = electric.mains
                split = MAINS_CODES.index(appliance_code) + 1
                key = MainsName(split=split, meter=1)
            elif appliance_code in CIRCUIT_CODES:
                dict_ = electric.circuits
                split = CIRCUIT_CODES.index(appliance_code) + 1
                key = CircuitName(name='sockets', split=split, meter=1)
            elif appliance_code in TEMPERATURE_CODES:
                is_temperature = True
                # TODO
            else:
                dict_ = electric.appliances
                key = appliance_code # TODO use nilmtk ApplianceNames

            if not is_temperature:
                try:
                    dict_[key].append(df)
                except KeyError:
                    dict_[key] = df

    def load_building(self, filename, building_name):
        raise NotImplementedError

    def load_building_names(self, filename):
        raise NotImplementedError
