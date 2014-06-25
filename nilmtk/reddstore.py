from __future__ import print_function, division
import pandas as pd
import numpy as np
from copy import deepcopy
import os
from .datastore import DataStore, Key
from .measurement import Power
from .timeframe import TimeFrame


MAP_REDD_LABELS_TO_NILMTK = {
    'air_conditioning':
        {'appliances': [
            {'type': 'air conditioner'}]},
    'bathroom_gfi': # GFI = ground fault interrupter (a type of RCD)?
        {'room': {'name': 'bathroom'},
         'category': 'misc'},
    'dishwaser':
        {'appliances': [
            {'type': 'dish washer'}]},
    'disposal':
        {'appliances': [
            {'type': 'waste disposal unit'}]},
    'electric_heat':
        {'appliances': [
            {'type': 'electric space heater'}]},
    'electronics':
        {'category': 'consumer electronics'},
    'furance':
        {'appliances': [
            {'type': 'electric boiler'}]},
    'kitchen_outlets':
        {'room': {'name': 'kitchen'},
         'category': 'sockets'},
    'lighting':
        {'category': 'lighting'},
    'microwave':
        {'appliances': [
            {'type': 'microwave'}]},
    'miscellaeneous':
        {'category': 'misc'},
    'outdoor_outlets':
        {'room': {'name': 'outdoors'},
         'category': 'sockets'},
    'outlets_unknown':
        {'category': 'sockets'},
    'oven':
        {'appliances': [
            {'type': 'electric oven'}]},
    'refrigerator':
        {'appliances': [
            {'type': 'fridge'}]},
    'smoke_alarms':
        {'appliances': [
            {'type': 'smoke alarm', 'multiple': True}]},
    'stove':
        {'appliances': [
            {'type': 'electric stove'}]},
    'subpanel': # not an appliance
        {}, 
    'washer_dryer':
        {'appliances': [
            {'type': 'washer dryer'}]}
}


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

    def load(self, key, periods=None):
        """
        Parameters
        ----------
        key : string, the location of a table within the DataStore.
        periods : list of TimeFrame objects.

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
        key_obj = Key(key)
        path = self._path_for_house(key_obj)

        # Get filename
        filename = 'channel_{:d}.dat'.format(key_obj.meter)
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

        if periods:
            for period in periods:
                yield period.slice(df)
        else:
            yield df

    def _path_for_house(self, key_obj):
        assert isinstance(key_obj, Key)
        house_dir = 'house_{:d}'.format(key_obj.building)
        path = os.path.join(self.path, house_dir)
        assert os.path.isdir(path)
        return path

    def load_metadata(self, key='/'):
        """
        Parameters
        ----------
        key : string, optional
            if '/' then load metadata for the whole dataset.

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
                        'max_sample_period': 50,
                        'measurements': [Power('active')], # this page says eMonitor records active power: http://www.energycircle.com/node/14103
                        'measurement_limits': {
                            Power('active'): {'lower_limit': 0, 'upper_limit': 5000}}
                        },
                    'REDD_whole_house': {
                        'sample_period': 1,
                        'max_sample_period': 30,
                        'measurements': [Power('active')],
                        'measurement_limits': {
                            Power('active'): {'lower_limit': 0, 'upper_limit': 50000}}
                    }
                }
            }
        
        # building metadata
        key_obj = Key(key)
        assert key_obj.meter is None
        if not 1 <= key_obj.building <= 6:
            raise ValueError("Building {} is not a valid building instance."
                             .format(key_obj.building))

        # Build dict of elec meter metadata dicts
        elec_meters = {}

        building_path = self._path_for_house(key_obj)
        labels = load_labels(building_path)
        appliance_instances = {} # map NILMTK appliance name to int
        for meter_i, label in labels.iteritems():
            if meter_i == 1:
                meter_metadata = {
                    'site_meter': True,
                    'device_model': 'REDD_whole_house',
                    'sensors': [
                        {'data_location':
                         str(Key(building=key_obj.building, meter=1))},
                        {'data_location':
                         str(Key(building=key_obj.building, meter=2))}]
                    }
            elif meter_i == 2:
                continue # split-phase mains is handled by meter 1
            else:
                meter_metadata = {
                    'submeter_of': 1,
                    'device_model': 'eMonitor',
                    'sensors': [
                        {'data_location':
                         str(Key(building=key_obj.building, meter=meter_i))}]
                }
                meter_metadata.update(deepcopy(MAP_REDD_LABELS_TO_NILMTK[label]))
                appliances = meter_metadata.get('appliances', [])
                for appliance_i, appliance_dict in enumerate(appliances):
                    appliance_type = appliance_dict['type']
                    instance = appliance_instances.setdefault(appliance_type, 1)
                    appliances[appliance_i]['instance'] = instance
                    appliance_instances[appliance_type] += 1

            elec_meters[meter_i] = meter_metadata

        return { # building metadata
            'instance': key_obj.building,
            'dataset': 'REDD',
            'original_name': 'house_{:d}'.format(key_obj.building),
            'elec_meters': elec_meters
        }

    def elements_below_key(self, key='/'):
        """
        Returns
        -------
        list of strings
        """
        # List buildings
        if key is None or key == '/':
            return ['building{:d}'.format(b) for b in range(1,7)]

        # List meters
        key_obj = Key(key)
        assert key_obj.building is not None
        assert key_obj.meter is None
        if not 1 <= key_obj.building <= 6:
            raise ValueError("Building {} is not a valid building instance."
                             .format(key_obj.building))

        if key_obj.utility is None:
            return ['elec']

        house_path = self._path_for_house(key_obj)
        labels = load_labels(house_path)
        return ["meter{:d}".format(meter) for meter in labels.keys()
                if meter not in [2]] # second mains channel is loaded as meter1


def test():
    from nilmtk import DataSet
    redd = REDDStore('/data/REDD/low_freq')
    dataset = DataSet()
    dataset.load(redd)
    print(dataset.buildings[1].elec.meters)
    return dataset
