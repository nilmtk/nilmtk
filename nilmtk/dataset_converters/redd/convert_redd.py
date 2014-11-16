from __future__ import print_function, division
import pandas as pd
import numpy as np
from copy import deepcopy
from os.path import join, isdir, isfile
from os import listdir
import re
from sys import stdout
from nilmtk.datastore import Key
from nilmtk.timeframe import TimeFrame
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import get_module_directory, check_directory_exists
from nilm_metadata import convert_yaml_to_hdf5

"""
TODO:
* The bottleneck appears to be CPU.  So could be sped up by using 
  multiprocessing module to use multiple CPU cores to load REDD channels in 
  parallel.
"""


def convert_redd(redd_path, hdf_filename):
    """
    Parameters
    ----------
    redd_path : str
        The root path of the REDD low_freq dataset.
    hdf_filename : str
        The destination HDF5 filename (including path and suffix).
    """

    def _redd_measurement_mapping_func(house_id, chan_id):
        ac_type = 'apparent' if chan_id <= 2 else 'active'
        return [('power', ac_type)]

    _convert(redd_path, hdf_filename, _redd_measurement_mapping_func, 'US/Eastern')

    # Add metadata
    convert_yaml_to_hdf5(join(get_module_directory(), 
                              'dataset_converters', 
                              'redd', 
                              'metadata'),
                         hdf_filename)

    print("Done converting REDD to HDF5!")


def _convert(input_path, hdf_filename, measurement_mapping_func, tz):
    """
    Parameters
    ----------
    input_path : str
        The root path of the REDD low_freq dataset.
    hdf_filename : str
        The destination HDF5 filename (including path and suffix).
    measurement_mapping_func : function
        Must take these parameters:
            - house_id
            - chan_id
        Function should return a list of tuples e.g. [('power', 'active')]
    tz : str 
        Timezone e.g. 'US/Eastern'
    """

    check_directory_exists(input_path)

    # Open HDF5 file
    store = pd.HDFStore(hdf_filename, 'w', complevel=9, complib='zlib')

    # Iterate though all houses and channels
    houses = _find_all_houses(input_path)
    for house_id in houses:
        print("Loading house", house_id, end="... ")
        stdout.flush()
        chans = _find_all_chans(input_path, house_id)
        for chan_id in chans:
            print(chan_id, end=" ")
            stdout.flush()
            key = Key(building=house_id, meter=chan_id)
            measurements = measurement_mapping_func(house_id, chan_id)
            df = _load_chan(input_path, key, measurements, tz)
            store.put(str(key), df, format='table')
            store.flush()
        print()

    store.close()
    

def _find_all_houses(input_path):
    """
    Returns
    -------
    list of integers (house instances)
    """
    dir_names = [p for p in listdir(input_path) if isdir(join(input_path, p))]
    return _matching_ints(dir_names, '^house_(\d)$')


def _find_all_chans(input_path, house_id):
    """
    Returns
    -------
    list of integers (channels)
    """
    house_path = join(input_path, 'house_{:d}'.format(house_id))
    filenames = [p for p in listdir(house_path) if isfile(join(house_path, p))]
    return _matching_ints(filenames, '^channel_(\d\d?).dat$')


def _matching_ints(strings, regex):
    """Uses regular expression to select and then extract an integer from
    strings.

    Parameters
    ----------
    strings : list of strings
    regex : string
        Regular Expression.  Including one group.  This group is used to
        extract the integer from each string.

    Returns
    -------
    list of ints
    """
    ints = []
    p = re.compile(regex)
    for string in strings:
        m = p.match(string)
        if m:
            integer = int(m.group(1))
            ints.append(integer)
    ints.sort()
    return ints


def _load_chan(input_path, key_obj, columns, tz):
    """
    Parameters
    ----------
    input_path : (str) the root path of the REDD low_freq dataset
    key_obj : (nilmtk.Key) the house and channel to load
    columns : list of tuples (for hierarchical column index)
    tz : str e.g. 'US/Eastern'

    Returns
    ------- 
    DataFrame of data.
    """
    assert isinstance(input_path, str)
    assert isinstance(key_obj, Key)

    # Get path
    house_path = 'house_{:d}'.format(key_obj.building)
    path = join(input_path, house_path)
    assert isdir(path)

    # Get filename
    filename = 'channel_{:d}.dat'.format(key_obj.meter)
    filename = join(path, filename)
    assert isfile(filename)

    # Load data
    df = pd.read_csv(filename, sep=' ', names=columns,
                     dtype={m:np.float32 for m in columns})
    
    # Modify the column labels to reflect the power measurements recorded.
    df.columns.set_names(LEVEL_NAMES, inplace=True)

    # raw REDD data isn't always sorted
    df = df.sort_index()
    
    # Convert the integer index column to timezone-aware datetime 
    df.index = pd.to_datetime(df.index.values, unit='s', utc=True)
    df = df.tz_convert(tz)

    return df
