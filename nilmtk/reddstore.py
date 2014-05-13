from __future__ import print_function, division
import pandas as pd
import numpy as np
from datastore import DataStore
from measurement import Power
from timeframe import TimeFrame
import os


map_redd_labels_to_nilmtk = {
    'air_conditioning': 
        {'type': 'air conditioner'},
    'bathroom_gfi': # ground fault interrupter? (A type of RCD.)
        {'type': 'unspecified', 'room': {'name': 'bathroom'}},
    'dishwaser': 
        {'type': 'dish washer'},
    'disposal': 
        {'type': 'waste disposal unit'},
    'electric_heat': 
        {'type': 'electric space heater'},
    'electronics':  
        {'type': 'ICT appliance'},
    'furance':  
        {'type': 'electric boiler'},
    'kitchen_outlets':  
        {'type': 'sockets', 'room': {'name': 'kitchen'},
    'lighting':  
        {'type': 'light'},
    'microwave':  
        {'type': 'microwave'},
    'miscellaeneous':  
        {'type': 'unspecified'},
    'outdoor_outlets':  
         {'type': 'sockets', 'room' {'name': 'outdoors'}},
    'outlets_unknown':  # TODO
        {'type': ''},
    'oven':  
        {'type': ''},
    'refrigerator':  
        {'type': ''},
    'smoke_alarms':  
        {'type': ''},
    'stove':  
        {'type': ''},
    'subpanel':  
        {'type': ''},
    'washer_dryer':  
        {'type': ''}   
}


def _split_key(key):
    return key.strip('/').split('/')


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


class REDDStore(DataStore):
    def __init__(self, path):
        """
        Parameters
        ----------
        path : string
        """
        if not os.path.isdir(path):
            raise ValueError("'{}' is not a valid path".format(path))

        self.path = path
        super(REDDStore, self).__init__()

    def load(self, key):
        """
        Parameters
        ----------
        key : string, the location of a table within the DataStore.

        Returns
        ------- 
        Returns a generator of DataFrame objects.  
        Each DataFrame has extra attributes:
                - timeframe : TimeFrame of period intersected with self.window
                - look_ahead : pd.DataFrame:
                    with `n_look_ahead_rows` rows.  The first row will be for
                    `period.end`.  `look_ahead` stores data which appears on 
                    disk immediately after `period.end`; i.e. it ignores
                    the next `period.start`.
        """
        path = self._path_for_house(key)

        # Get filename
        split_key = _split_key(key)
        filename = split_key[-1].replace('meter', 'channel_') + '.dat'
        filename = os.path.join(path, filename)

        # load data
        df = pd.read_csv(filename, sep=' ', index_col=0,
                         names=[Power('active')], 
                         tupleize_cols=True, # required to use Power('active')
                         dtype={Power('active'): np.float32})

        # Basic post-processing
        df = df.sort_index() # raw REDD data isn't always sorted
        df.index = pd.to_datetime((df.index.values*1E9).astype(int), utc=True)
        df = df.tz_convert('US/Eastern')
        df.timeframe = TimeFrame(df.index[0], df.index[-1])
        df.timeframe.include_end = True

        yield df

    def _path_for_house(self, key):
        split_key = _split_key(key)
        assert split_key[0].startswith('building')
        assert split_key[-1].startswith('meter')

        # Get path of house
        house_dir = split_key[0].replace('building', 'house_')
        path = os.path.join(self.path, house_dir)
        assert os.path.isdir(path)
        return path

    def load_metadata(self, key='/'):
        """
        Parameters
        ----------
        key : string, optional
            if None then load metadata for the whole dataset.

        Returns
        -------
        metadata : dict
        """

        # whole-dataset metadata
        if key == '/':
            return {
                'meter_devices': {
                    'eMonitor': {
                        'model': 'eMonitor',
                        'manufacturer': 'Powerhouse Dynamics',
                        'manufacturer_url': 'http://powerhousedynamics.com',
                        'sample_period': 3,
                        'max_sample_period': 30,
                        'measurements': [Power('active')],
                        'measurement_limits': {
                            Power('active'): {'lower': 0, 'upper': 5000}}
                        },
                    'REDD_whole_house': {
                        'sample_period': 1,
                        'max_sample_period': 10,
                        'measurements': [Power('active')],
                        'measurement_limits': {
                            Power('active'): {'lower': 0, 'upper': 50000}}
                    }
                }
            }
        
        split_key = key.strip('/').split('/')
        building_instance = int(split_key[0].replace('building', ''))
        assert 1 <= building_instance <= 6

        # building metadata
        if len(split_key) == 1:
            return {
                'building_instance': building_instance,
                'dataset': 'REDD',
                'original_name': 'house_{:d}'.format(building_instance)
            }

        # meter-level metadata
        meter_instance = int(split_key[-1].replace('meter', ''))

        meter_metadata = {
            'device_model': 'REDD_whole_house' if meter_instance in [1,2] else 'eMonitor',
            'instance': meter_instance,
            'building': building_instance,
            'dataset': 'REDD'            
        }
        if meter_instance in [1,2]:
            meter_metadata.update({'site_meter': True})
        else:
            meter_metadata.update({'submeter_of': 0})

        building_path = self._path_for_house(key)
        labels = load_labels(building_path)
        redd_label = labels[meter_instance]
        nilmtk_label = map_redd_labels_to_nilmtk[redd_label]
        
        #TODO: submeter and site_meter and dominant_appliance and appliance data...

        

        raise NotImplementedError()

    def elements_below_key(self, key='/'):
        """
        Returns
        -------
        list of strings
        """
        raise NotImplementedError()
