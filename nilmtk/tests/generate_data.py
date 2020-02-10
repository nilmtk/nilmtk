import pandas as pd
from datetime import timedelta
from nilmtk.tests.testingtools import data_dir
from os.path import join
import itertools
from collections import OrderedDict
import numpy as np
from nilmtk.consts import JOULES_PER_KWH
from nilmtk.measurement import measurement_columns, AC_TYPES
from nilmtk.utils import flatten_2d_list

MAX_SAMPLE_PERIOD = 15


def power_data(simple=True):
    """
    Returns
    -------
    DataFrame
    """

    if simple:
        STEP = 10
        data = [0,  0,  0, 100, 100, 100, 150,
                150, 200,   0,   0, 100, 5000,    0]
        secs = np.arange(start=0, stop=len(data) * STEP, step=STEP)
    else:
        data = [0,  0,  0, 100, 100, 100, 150,
                150, 200,   0,   0, 100, 5000,    0]
        secs = [0, 10, 20,  30, 200, 210, 220,
                230, 240, 249, 260, 270,  290, 1000]

    data = np.array(data, dtype=np.float32)
    active = data
    reactive = data * 0.9
    apparent = data * 1.1

    index = [pd.Timestamp('2010-01-01') + timedelta(seconds=sec)
             for sec in secs]
    column_tuples = [('power', ac_type)
                     for ac_type in ['active', 'reactive', 'apparent']]
    df = pd.DataFrame(np.array([active, reactive, apparent]).transpose(),
                      index=index, dtype=np.float32,
                      columns=measurement_columns(column_tuples))

    # calculate energy
    # this is not cumulative energy
    timedelta_secs = np.diff(secs).clip(
        0, MAX_SAMPLE_PERIOD).astype(np.float32)

    for ac_type in AC_TYPES:
        joules = timedelta_secs * df['power', ac_type].values[:-1]
        joules = np.concatenate([joules, [0]])
        kwh = joules / JOULES_PER_KWH
        if ac_type == 'reactive':
            df['energy', ac_type] = kwh
        elif ac_type == 'apparent':
            df['cumulative energy', ac_type] = kwh.cumsum()

    return df


def create_random_df_hierarchical_column_index():
    N_PERIODS = 1E4
    N_METERS = 5
    N_MEASUREMENTS_PER_METER = 3

    meters = ['meter{:d}'.format(i) for i in range(1, N_METERS + 1)]
    meters = [[m] * N_MEASUREMENTS_PER_METER for m in meters]
    meters = flatten_2d_list(meters)
    level2 = ['power', 'power', 'voltage'][
        :N_MEASUREMENTS_PER_METER] * N_METERS
    level3 = ['active', 'reactive', ''][:N_MEASUREMENTS_PER_METER] * N_METERS

    columns = [meters, level2, level3]
    columns = pd.MultiIndex.from_arrays(columns)
    rng = pd.date_range('2012-01-01', freq='S', periods=N_PERIODS)
    data = np.random.randint(low=0, high=1000,
                             size=(N_PERIODS,
                                   N_METERS * N_MEASUREMENTS_PER_METER))
    return pd.DataFrame(data=data, index=rng, columns=columns, dtype=np.float32)

MEASUREMENTS = [('power', 'active'), ('energy', 'reactive'), ('voltage', '')]


def create_random_df():
    N_PERIODS = 1E4
    rng = pd.date_range('2012-01-01', freq='S', periods=N_PERIODS)
    data = np.random.randint(
        low=0, high=1000, size=(N_PERIODS, len(MEASUREMENTS)))
    return pd.DataFrame(data=data, index=rng, dtype=np.float32,
                        columns=measurement_columns(MEASUREMENTS))


TEST_METER = {'manufacturer': 'Test Manufacturer',
              'model': 'Random Meter',
              'sample_period': 10,
              'max_sample_period': MAX_SAMPLE_PERIOD,
              'measurements': []}

for col in MEASUREMENTS:
    TEST_METER['measurements'].append({
        'physical_quantity': col[0], 'type': col[1],
        'lower_limit': 0, 'upper_limit': 6000})


def add_building_metadata(store, elec_meters, key='building1', appliances=[]):
    node = store.get_node(key)
    md = {
        'instance': 1,
        'elec_meters': elec_meters,
        'appliances': appliances
    }
    node._f_setattr('metadata', md)


def create_co_test_hdf5():
    FILENAME = join(data_dir(), 'co_test.h5')
    N_METERS = 3
    chunk = 1000
    N_PERIODS = 4 * chunk
    rng = pd.date_range('2012-01-01', freq='S', periods=N_PERIODS)

    dfs = OrderedDict()
    data = OrderedDict()

    # mains meter data
    data[1] = np.array([0, 200, 1000, 1200] * chunk)

    # appliance 1 data
    data[2] = np.array([0, 200, 0, 200] * chunk)

    # appliance 2 data
    data[3] = np.array([0, 0, 1000, 1000] * chunk)

    for i in range(1, 4):
        dfs[i] = pd.DataFrame(data=data[i], index=rng, dtype=np.float32,
                              columns=measurement_columns([('power', 'active')]))

    store = pd.HDFStore(FILENAME, 'w', complevel=9, complib='zlib')
    elec_meter_metadata = {}
    for meter in range(1, N_METERS + 1):
        key = 'building1/elec/meter{:d}'.format(meter)
        print("Saving", key)
        store.put(key, dfs[meter], format='table')
        elec_meter_metadata[meter] = {
            'device_model': TEST_METER['model'],
            'submeter_of': 1,
            'data_location': key
        }

    # For mains meter, we need to specify that it is a site meter
    del elec_meter_metadata[1]['submeter_of']
    elec_meter_metadata[1]['site_meter'] = True

    # Save dataset-wide metadata
    store.root._v_attrs.metadata = {
        'meter_devices': {TEST_METER['model']: TEST_METER}}
    print(store.root._v_attrs.metadata)

    # Building metadata
    add_building_metadata(store, elec_meter_metadata)
    for key in store.keys():
        print(store[key])

    store.flush()
    store.close()


def create_random_hdf5():
    FILENAME = join(data_dir(), 'random.h5')
    N_METERS = 5

    store = pd.HDFStore(FILENAME, 'w', complevel=9, complib='zlib')
    elec_meter_metadata = {}
    for meter in range(1, N_METERS + 1):
        key = 'building1/elec/meter{:d}'.format(meter)
        print("Saving", key)
        store.put(key, create_random_df(), format='table')
        elec_meter_metadata[meter] = {
            'device_model': TEST_METER['model'],
            'submeter_of': 1,
            'data_location': key
        }

    # Save dataset-wide metadata
    store.root._v_attrs.metadata = {
        'meter_devices': {TEST_METER['model']: TEST_METER}}
    print(store.root._v_attrs.metadata)

    # Building metadata
    add_building_metadata(store, elec_meter_metadata)

    store.flush()
    store.close()


def create_energy_hdf5(simple=True):
    fname = 'energy.h5' if simple else 'energy_complex.h5'
    FILENAME = join(data_dir(), fname)

    df = power_data(simple=simple)

    meter_device = {
        'manufacturer': 'Test Manufacturer',
        'model': 'Energy Meter',
        'sample_period': 10,
        'max_sample_period': MAX_SAMPLE_PERIOD,
        'measurements': []
    }

    for physical_quantity, ac_type in df.columns.tolist():
        meter_device['measurements'].append({
            'physical_quantity': physical_quantity, 'type': ac_type,
            'lower_limit': 0, 'upper_limit': 6000})

    store = pd.HDFStore(FILENAME, 'w', complevel=9, complib='zlib')

    elec_meter_metadata = {}

    # Save sensor data
    for meter_i in [1, 2, 3]:
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
    store.root._v_attrs.metadata = {
        'meter_devices': {meter_device['model']: meter_device}}

    # Add building metadata
    add_building_metadata(store, elec_meter_metadata)

    store.flush()
    store.close()


def create_all():
    create_energy_hdf5()
    create_energy_hdf5(simple=False)
    create_random_hdf5()
    create_co_test_hdf5()
