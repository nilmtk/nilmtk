from __future__ import print_function, division
import pandas as pd
import numpy as np
from copy import deepcopy
from os.path import join, isdir, isfile, dirname
from os import listdir, getcwd
from sys import stdout, getfilesystemencoding
from nilmtk.datastore import Key
from nilmtk.timeframe import TimeFrame
from nilm_metadata import convert_yaml_to_hdf5
from inspect import currentframe, getfile, getsourcefile


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

    assert isdir(redd_path)

    # Open HDF5 file
    store = pd.HDFStore(hdf_filename, 'w', complevel=9, complib='bzip2')

    # Iterate though all houses and channels
    houses = _find_all_houses(redd_path)
    for house_id in houses:
        print("Loading house", house_id, end="... ")
        stdout.flush()
        chans = _find_all_chans(redd_path, house_id)
        for chan_id in chans:
            print(chan_id, end=" ")
            stdout.flush()
            key = Key(building=house_id, meter=chan_id)
            ac_type = 'apparent' if chan_id <= 2 else 'active'
            df = _load_chan(redd_path, key, [('power', ac_type)])
            store.put(str(key), df, format='table')
            store.flush()
        print()

    store.close()
    
    # Add metadata
    convert_yaml_to_hdf5(join(_get_module_directory(), 'metadata'),
                         hdf_filename)

    print("done!")


def _find_all_houses(redd_path):
    dir_names = listdir(redd_path)
    house_ids = [int(directory.replace('house_', '')) for directory in dir_names
                 if directory.startswith('house_')]
    house_ids.sort()
    return house_ids


def _find_all_chans(redd_path, house_id):
    house_path = join(redd_path, 'house_{:d}'.format(house_id))
    filenames = listdir(house_path)
    chans = [int(fname.replace('channel_', '').replace('.dat', '')) 
             for fname in filenames
             if fname.startswith('channel_') and fname.endswith('.dat')]
    chans.sort()
    return chans


def _load_chan(redd_path, key_obj, columns):
    """
    Parameters
    ----------
    redd_path : (str) the root path of the REDD low_freq dataset
    key_obj : (nilmtk.Key) the house and channel to load
    columns : list of tuples (for hierarchical column index)

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
    df = pd.read_csv(filename, sep=' ', names=columns,
                     dtype={m:np.float32 for m in columns})

    # Basic post-processing
    df = df.sort_index() # raw REDD data isn't always sorted
    df.index = pd.to_datetime((df.index.values*1E9).astype(int), utc=True)
    df = df.tz_convert('US/Eastern')
    df.timeframe = TimeFrame(df.index[0], df.index[-1])
    df.timeframe.include_end = True
    return df


def _get_module_directory():
    # Taken from http://stackoverflow.com/a/6098238/732596
    path_to_this_file = dirname(getfile(currentframe()))
    if not isdir(path_to_this_file):
        encoding = getfilesystemencoding()
        path_to_this_file = dirname(unicode(__file__, encoding))
    if not isdir(path_to_this_file):
        abspath(getsourcefile(lambda _: None))
    if not isdir(path_to_this_file):
        path_to_this_file = getcwd()
    assert isdir(path_to_this_file), path_to_this_file + ' is not a directory'
    return path_to_this_file
