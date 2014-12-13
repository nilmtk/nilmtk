from __future__ import print_function, division
import pandas as pd
import numpy as np
from os.path import join, isdir, isfile, dirname, abspath
from os import getcwd
from sys import getfilesystemencoding
from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import check_directory_exists, get_datastore
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

TIMESTAMP_COLUMN_NAME = "timestamp"
TIMEZONE = "Asia/Kolkata"

def convert_iawe(iawe_path, output_filename, format="HDF"):
    """
    Parameters
    ----------
    iawe_path : str
        The root path of the iawe dataset.
    output_filename : str
        The destination filename (including path and suffix).
    """

    check_directory_exists(iawe_path)

    # Open data store
    store = get_datastore(output_filename, format, mode='w')
    electricity_path = join(iawe_path, "electricity")

    # Mains data
    for chan in range(1, 13):
        key = Key(building=1, meter=chan)
        filename = join(electricity_path, "%d.csv" % chan)
        print('Loading ', chan)
        df = pd.read_csv(filename)
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.index = pd.to_datetime(df.timestamp.values, unit='s', utc=True)
        df = df.tz_convert(TIMEZONE)
        df = df.drop(TIMESTAMP_COLUMN_NAME, 1)
        df.rename(columns=lambda x: column_mapping[x], inplace=True)
        df.columns.set_names(LEVEL_NAMES, inplace=True)
        df = df.convert_objects(convert_numeric=True)
        df = df.dropna()
        df = df.astype(np.float32)
        df = df.sort_index()
        store.put(str(key), df)
    store.close()
    convert_yaml_to_hdf5(join(_get_module_directory(), 'metadata'),
                         output_filename)

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
