from __future__ import print_function, division
import pandas as pd
from datetime import timedelta
from nilmtk.tests.testingtools import data_dir
from os.path import join
import itertools
import numpy as np
from nilmtk.measurement import Voltage, Energy, Power, AC_TYPES
from nilmtk.consts import JOULES_PER_KWH

MAX_SAMPLE_PERIOD = 15

def power_data(simple=True):
    """
    Returns
    -------
    DataFrame
    """

    if simple:
        STEP = 10
        data = [0,  0,  0, 100, 100, 100, 150, 150, 200,   0,   0, 100, 5000,    0]
        secs = np.arange(start=0, stop=len(data)*STEP, step=STEP)
    else:
        data = [0,  0,  0, 100, 100, 100, 150, 150, 200,   0,   0, 100, 5000,    0]
        secs = [0, 10, 20,  30, 200, 210, 220, 230, 240, 249, 260, 270,  290, 1000]

    data = np.array(data, dtype=np.float32) 
    active = data
    reactive = data * 0.9
    apparent = data * 1.1
    
    index = [pd.Timestamp('2010-01-01') + timedelta(seconds=sec) for sec in secs]

    df = pd.DataFrame(np.array([active, reactive, apparent]).transpose(),
                      index=index, dtype=np.float32, 
                      columns=[Power(ac_type) for ac_type in 
                               ['active', 'reactive', 'apparent']])

    # calculate energy
    # this is not cumulative energy
    timedelta_secs = np.diff(secs).clip(0, MAX_SAMPLE_PERIOD).astype(np.float32)

    for ac_type in AC_TYPES:
        joules = timedelta_secs * df[Power(ac_type)].values[:-1]
        joules = np.concatenate([joules, [0]])
        kwh = joules / JOULES_PER_KWH
        if ac_type == 'reactive':
            df[Energy(ac_type)] = kwh
        elif ac_type == 'apparent':
            df[Energy(ac_type, cumulative=True)] = kwh.cumsum()

    return df


def create_random_df_hierarchical_column_index():
    N_PERIODS = 1E4
    N_METERS = 5
    N_MEASUREMENTS_PER_METER = 3

    meters = ['meter{:d}'.format(i) for i in range(1,N_METERS+1)]
    meters = [[m]*N_MEASUREMENTS_PER_METER for m in meters]
    flatten_2d_list = lambda lst: list(itertools.chain(*lst))
    meters = flatten_2d_list(meters)
    level2 = ['power', 'power', 'voltage'][:N_MEASUREMENTS_PER_METER] * N_METERS
    level3 = ['active', 'reactive', ''][:N_MEASUREMENTS_PER_METER] * N_METERS


    columns = [meters, level2, level3]
    columns = pd.MultiIndex.from_arrays(columns)
    rng = pd.date_range('2012-01-01', freq='S', periods=N_PERIODS)
    data = np.random.randint(low=0, high=1000, 
                             size=(N_PERIODS, 
                                   N_METERS*N_MEASUREMENTS_PER_METER))
    return pd.DataFrame(data=data, index=rng, columns=columns, dtype=np.float32)

MEASUREMENTS = [Power('active'), Energy('reactive'), Voltage()]


def create_random_df():
    N_PERIODS = 1E4
    columns = MEASUREMENTS
    rng = pd.date_range('2012-01-01', freq='S', periods=N_PERIODS)
    data = np.random.randint(low=0, high=1000, size=(N_PERIODS, len(columns)))
    return pd.DataFrame(data=data, index=rng, columns=columns, dtype=np.float32)


TEST_METER = {'manufacturer': 'Test Manufacturer', 
              'model': 'Random Meter', 
              'sample_period': 10,
              'max_sample_period': MAX_SAMPLE_PERIOD,
              'measurements': [Power('apparent')],
              'measurement_limits': {}
          }
for col in MEASUREMENTS:
    TEST_METER['measurement_limits'][col] = {'lower_limit': 0, 'upper_limit': 6000}


def add_building_metadata(store, elec_meters, key='building1', appliances=[]):
    node = store.get_node(key)
    md = {
        'instance': 1,
        'elec_meters': elec_meters,
        'appliances': appliances
    }
    node._f_setattr('metadata', md)


def create_random_hdf5():
    FILENAME = join(data_dir(), 'random.h5')
    N_METERS = 5

    store = pd.HDFStore(FILENAME, 'w', complevel=9, complib='bzip2')
    elec_meter_metadata = {}
    for meter in range(1, N_METERS+1):
        key = 'building1/elec/meter{:d}'.format(meter)
        print("Saving", key)
        store.put(key, create_random_df(), format='table')
        elec_meter_metadata[meter] = {
            'device_model': TEST_METER['model'], 
            'submeter_of': 1,
            'data_location': key
        }

    # Save dataset-wide metadata
    store.root._v_attrs.metadata = {'meter_devices': {TEST_METER['model']: TEST_METER}}
    print(store.root._v_attrs.metadata)

    # Building metadata
    add_building_metadata(store, elec_meter_metadata)

    store.flush()
    store.close()


def create_energy_hdf5(simple=True):
    fname = 'energy.h5' if simple else 'energy_complex.h5'
    FILENAME = join(data_dir(), fname)

    df = power_data(simple=simple)

    meter_device = {'manufacturer': 'Test Manufacturer', 
             'model': 'Energy Meter', 
             'sample_period': 10,
             'max_sample_period': MAX_SAMPLE_PERIOD,
             'measurements': df.columns,
             'measurement_limits': {}
         }
    for col in df.columns:
        meter_device['measurement_limits'][col] = {'lower_limit': 0, 'upper_limit': 6000}

    store = pd.HDFStore(FILENAME, 'w', complevel=9, complib='bzip2')

    elec_meter_metadata = {}

    # Save sensor data
    for meter_i in [1,2,3]:
        key = 'building1/elec/meter{:d}'.format(meter_i)
        print("Saving", key)
        store.put(key, df, format='table')
        meta = {
            'device_model': meter_device['model'],
            'data_location': key
        }
        additional_meta = {
            1: {'site_meter': True},
            2: {'submeter_of': 1},
            3: {'submeter_of': 2}
        }
        meta.update(additional_meta[meter_i])
        elec_meter_metadata[meter_i] = meta

    # Save dataset-wide metadata
    store.root._v_attrs.metadata = {'meter_devices': {meter_device['model']: meter_device}}

    # Add building metadata 
    add_building_metadata(store, elec_meter_metadata)

    store.flush()
    store.close()


def create_all():
    create_energy_hdf5()
    create_energy_hdf5(simple=False)
    create_random_hdf5() 
