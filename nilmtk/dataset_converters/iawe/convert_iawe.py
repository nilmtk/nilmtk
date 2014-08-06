from __future__ import print_function, division
import pandas as pd
import numpy as np
from copy import deepcopy
from os.path import join, isdir, isfile, dirname, abspath
from os import listdir, getcwd
import re
from sys import stdout, getfilesystemencoding
from nilmtk.datastore import Key
from nilmtk.timeframe import TimeFrame
from nilmtk.measurement import LEVEL_NAMES
from nilm_metadata import convert_yaml_to_hdf5
from inspect import currentframe, getfile, getsourcefile


column_mapping = {
    'frequency': ('frequency', ""),
    'voltage': ('voltage', ""),
    'W': ('power', 'active'),
    'energy': ('energy', 'apparent'),
    'A': ('current', ''),
    'reactive_power': ('power', 'reactive'),
    'apparent_power': ('power', 'apparent'),
    'power_factor': ('pf', ''),
    'PF': ('pf', ''),
    'phase_angle': ('phi', ''),
    'VA': ('power', 'apparent'),
    'VAR': ('power', 'reactive'),
    'VLN': ('voltage', ""),
    'V': ('voltage', ""),
    'f': ('frequency', "")
}


def convert_iawe(iawe_path, hdf_filename):
    """
    Parameters
    ----------
    iawe_path : str
        The root path of the iawe dataset.
    hdf_filename : str
        The destination HDF5 filename (including path and suffix).
    """

    assert isdir(iawe_path)

    # Open HDF5 file
    store = pd.HDFStore(hdf_filename, 'w', complevel=9, complib='zlib')

    electricity_path = join(iawe_path, "electricity")

    # Mains data
    for chan in range(1, 13):
        key = Key(building=1, meter=chan)
        filename = join(electricity_path, "%d.csv" % chan)
        print('Loading ', chan)
        df = pd.read_csv(filename)
        df.index = pd.to_datetime(
            (df.timestamp.values * 1E9).astype(int), utc=True)
        df = df.tz_convert('Asia/Kolkata')
        df = df.drop('timestamp', 1)
        df.rename(columns=lambda x: column_mapping[x], inplace=True)
        df.columns.set_names(LEVEL_NAMES, inplace=True)
        df = df.convert_objects(convert_numeric=True)
        df = df.dropna()
        df = df.astype(np.float32)
        df = df.sort_index()
        store.put(str(key), df, format='table')
        store.flush()
    store.close()
    convert_yaml_to_hdf5(join(_get_module_directory(), 'metadata'),
                         hdf_filename)

    print("Done converting iAWE to HDF5!")


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
