import pandas as pd
import numpy as np
from copy import deepcopy
from os.path import join, isdir, isfile
from os import listdir
import re
from sys import stdout
from nilmtk.utils import get_datastore
from nilmtk.datastore import Key
from nilmtk.timeframe import TimeFrame
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import get_module_directory, check_directory_exists
from nilm_metadata import convert_yaml_to_hdf5, save_yaml_to_datastore

"""
TODO:
* The bottleneck appears to be CPU.  So could be sped up by using 
  multiprocessing module to use multiple CPU cores to load REDD channels in 
  parallel.
"""


def convert_redd(redd_path, output_filename, format='HDF'):
    """
    Parameters
    ----------
    redd_path : str
        The root path of the REDD low_freq dataset.
    output_filename : str
        The destination filename (including path and suffix).
    format : str
        format of output. Either 'HDF' or 'CSV'. Defaults to 'HDF'
    """

    def _redd_measurement_mapping_func(house_id, chan_id):
        ac_type = 'apparent' if chan_id <= 2 else 'active'
        return [('power', ac_type)]
        
    # Open DataStore
    store = get_datastore(output_filename, format, mode='w')

    # Convert raw data to DataStore
    _convert(redd_path, store, _redd_measurement_mapping_func, 'US/Eastern')

    s=join(get_module_directory(),
                              'dataset_converters',
                              'redd',
                              'metadata')


    # Add metadata
    save_yaml_to_datastore(join(get_module_directory(), 
                              'dataset_converters', 
                              'redd', 
                              'metadata'),
                         store)
    store.close()

    print("Done converting REDD to HDF5!")

def _convert(input_path, store, measurement_mapping_func, tz, sort_index=True, drop_duplicates=False):
    """
    Parameters
    ----------
    input_path : str
        The root path of the REDD low_freq dataset.
    store : DataStore
        The NILMTK DataStore object.
    measurement_mapping_func : function
        Must take these parameters:
            - house_id
            - chan_id
        Function should return a list of tuples e.g. [('power', 'active')]
    tz : str 
        Timezone e.g. 'US/Eastern'
    sort_index : bool
        Defaults to True
    drop_duplicates : bool
        Remove entries with duplicated timestamp (keeps the first value)
        Defaults to False for backwards compatibility.
    """

    check_directory_exists(input_path)

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
            csv_filename = _get_csv_filename(input_path, key)
            df = _load_csv(csv_filename, measurements, tz, 
                sort_index=sort_index, 
                drop_duplicates=drop_duplicates
            )

            store.put(str(key), df)
        print()


def _find_all_houses(input_path):
    """
    Returns
    -------
    list of integers (house instances)
    """
    dir_names = [p for p in listdir(input_path) if isdir(join(input_path, p))]
    return _matching_ints(dir_names, r'^house_(\d)$')


def _find_all_chans(input_path, house_id):
    """
    Returns
    -------
    list of integers (channels)
    """
    house_path = join(input_path, 'house_{:d}'.format(house_id))
    filenames = [p for p in listdir(house_path) if isfile(join(house_path, p))]
    return _matching_ints(filenames, r'^channel_(\d\d?).dat$')


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


def _get_csv_filename(input_path, key_obj):
    """
    Parameters
    ----------
    input_path : (str) the root path of the REDD low_freq dataset
    key_obj : (nilmtk.Key) the house and channel to load

    Returns
    ------- 
    filename : str
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

    return filename


def _load_csv(filename, columns, tz, drop_duplicates=False, sort_index=False):
    """
    Parameters
    ----------
    filename : str
    columns : list of tuples (for hierarchical column index)
    tz : str 
        e.g. 'US/Eastern'
    sort_index : bool
        Defaults to True
    drop_duplicates : bool
        Remove entries with duplicated timestamp (keeps the first value)
        Defaults to False for backwards compatibility.

    Returns
    -------
    pandas.DataFrame
    """
    # Load data
    df = pd.read_csv(filename, sep=' ', names=columns,
                     dtype={m:np.float32 for m in columns})
    
    # Modify the column labels to reflect the power measurements recorded.
    df.columns.set_names(LEVEL_NAMES, inplace=True)

    # Convert the integer index column to timezone-aware datetime 
    df.index = pd.to_datetime(df.index.values, unit='s', utc=True)
    df = df.tz_convert(tz)

    if sort_index:
        df = df.sort_index() # raw REDD data isn't always sorted
        
    if drop_duplicates:
        dups_in_index = df.index.duplicated(keep='first')
        if dups_in_index.any():
            df = df[~dups_in_index]

    return df
