from __future__ import print_function, division
import pandas as pd
import numpy as np
from copy import deepcopy
from os.path import join, isdir, isfile
from os import listdir, remove
from sys import stdout
from nilmtk.datastore import Key
from nilmtk.measurement import Power
from nilmtk.timeframe import TimeFrame

"""
TODO:
* Load just the data into HDF5.
* Use a hand-written set of .YAML files for metadata.
* Write general function (in NILM Metadata???) for converting YAML to HDF5
* remove any imports we don't need.
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

    assert isdir(redd_path)

    # Open HDF5 file
    if isfile(hdf_filename):
        remove(hdf_filename)
    store = pd.HDFStore(hdf_filename, 'w', complevel=9, complib='bzip2')

    # Iterate though all houses and channels
    houses = find_all_houses(redd_path)
    for house_id in [1]:
        print("Loading house", house_id, end="... ")
        stdout.flush()
        chans = find_all_chans(redd_path, house_id)
        for chan_id in chans:
            print(chan_id, end=" ")
            stdout.flush()
            key = Key(building=house_id, meter=chan_id)
            df = load_chan(redd_path, key)
            df.columns = [Power('active')] # modify as per metadata
            store.put(str(key), df, format='table')
            store.flush()
        print()

    store.close()
    print("done!")


def find_all_houses(redd_path):
    dir_names = listdir(redd_path)
    house_ids = [int(directory.replace('house_', '')) for directory in dir_names
                 if directory.startswith('house_')]
    house_ids.sort()
    return house_ids


def find_all_chans(redd_path, house_id):
    house_path = join(redd_path, 'house_{:d}'.format(house_id))
    filenames = listdir(house_path)
    chans = [int(fname.replace('channel_', '').replace('.dat', '')) 
             for fname in filenames
             if fname.startswith('channel_') and fname.endswith('.dat')]
    chans.sort()
    return chans


def load_chan(redd_path, key_obj):
    """
    Parameters
    ----------
    redd_path : (str) the root path of the REDD low_freq dataset
    key_obj : (nilmtk.Key) the house and channel to load

    Returns
    ------- 
    DataFrame of data. Has extra attributes:
        - timeframe : TimeFrame of period intersected with self.window
    """
    assert isinstance(redd_path, str)
    assert isinstance(key_obj, Key)

    # Get path
    house_path = 'house_{:d}'.format(key_obj.building)
    path = join(redd_path, house_path)
    assert isdir(path)

    # Get filename
    filename = 'channel_{:d}.dat'.format(key_obj.meter)
    filename = join(path, filename)
    assert isfile(filename)

    # Load data
    df = pd.read_csv(filename, sep=' ', index_col=0, header=None,
                    dtype={1: np.float32})

    # Basic post-processing
    df = df.sort_index() # raw REDD data isn't always sorted
    df.index = pd.to_datetime((df.index.values*1E9).astype(int), utc=True)
    df = df.tz_convert('US/Eastern')
    df.timeframe = TimeFrame(df.index[0], df.index[-1])
    df.timeframe.include_end = True
    return df
