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
from collections import OrderedDict

PARAMS_TO_USE = ['Current', 'Energy', 'Power']

SUBMETER_PATHS = OrderedDict({
    'Building Total Mains':[0],
    'Lifts':[0],
    'Floor Total':[1, 2, 5],
    'AHU': [0, 1, 2, 5]})


column_mapping = OrderedDict({
    'Power': ('power', 'active'),
    'Energy': ('energy', 'active'),
    'Current': ('current', '')
    })


def convert_combed(combed_path, hdf_filename):
    """
    Parameters
    ----------
    combed_path : str
        The root path of the combed dataset.
    hdf_filename : str
        The destination HDF5 filename (including path and suffix).
    """

    assert isdir(combed_path)

    # Open HDF5 file
    store = pd.HDFStore(hdf_filename, 'w', complevel=9, complib='zlib')
    chan = 1
    for building, meter_array in SUBMETER_PATHS.iteritems():
        for meter in meter_array:
            key = Key(building=1, meter=chan)
            dfs = []
            total = pd.DataFrame()
            for attribute in column_mapping.keys():
                filename_attribute = join(combed_path, building, str(meter), "%s.csv" %attribute )
                print(filename_attribute)
                dfs.append(pd.read_csv(filename_attribute, parse_dates = True, index_col = 0, header = True, names=[attribute]))
            total = pd.concat(dfs, axis = 1)
                   
            total.rename(columns=lambda x: column_mapping[x], inplace=True)
            total.columns.set_names(LEVEL_NAMES, inplace=True)
            store.put(str(key), total, format='table')
            store.flush()
            chan = chan+ 1
    convert_yaml_to_hdf5(join(_get_module_directory(), 'metadata'),
                         hdf_filename)

    print("Done converting COMBED to HDF5!")


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
