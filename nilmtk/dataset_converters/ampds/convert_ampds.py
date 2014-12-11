from __future__ import print_function, division
import numpy as np
import pandas as pd
from os.path import *
from os import getcwd
from os import listdir
from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import check_directory_exists, get_datastore
from nilm_metadata import *
from inspect import currentframe, getfile, getsourcefile
from sys import getfilesystemencoding

# Column name mapping
columnNameMapping = {'V': ('voltage', ''),
                     'I': ('current', ''),
                     'f': ('frequency', ''),
                     'DPF': ('pf', 'd'),
                     'APF': ('power factor', 'apparent'),
                     'P': ('power', 'active'),
                     'Pt': ('energy', 'active'),
                     'Q': ('power', 'reactive'),
                     'Qt': ('energy', 'reactive'),
                     'S': ('power', 'apparent'),
                     'St': ('energy', 'apparent')}

TIMESTAMP_COLUMN_NAME = "TIMESTAMP"
TIMEZONE = "America/Vancouver"


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


def convert_ampds(input_path, output_filename, format='HDF'):
    """
    Parameters: 
    -----------
    inputPath: str
            The path of the directory where all the csv 
            files are supposed to be stored
    hdfFilename: str
            The path of the h5 file where all the 
            standardized data is supposed to go. The path 
            should refer to a particular file and not just a
             random directory in order for this to work.
    Example usage:
    --------------
    convert('/AMPds/electricity', 'store.h5')    

    """
    check_directory_exists(input_path)
    files = [f for f in listdir(input_path) if isfile(join(input_path, f)) and
             '.csv' in f and '.swp' not in f]
    # Sorting Lexicographically
    files.sort()

    # Remove Whole Home and put it at top
    files.remove("WHE.csv")
    files.insert(0, "WHE.csv")
    assert isdir(input_path)
    store = get_datastore(output_filename, format, mode='w')
    for i, csv_file in enumerate(files):
        key = Key(building=1, meter=(i + 1))
        print('Loading file #', (i + 1), ' : ', csv_file, '. Please wait...')
        df = pd.read_csv(join(input_path, csv_file))
        # Due to fixed width, column names have spaces :(
        df.columns = [x.replace(" ", "") for x in df.columns]
        df.index = pd.to_datetime(df[TIMESTAMP_COLUMN_NAME], unit='s', utc=True)
        df = df.drop(TIMESTAMP_COLUMN_NAME, 1)
        df = df.tz_localize('GMT').tz_convert(TIMEZONE)
        df.rename(columns=lambda x: columnNameMapping[x], inplace=True)
        df.columns.set_names(LEVEL_NAMES, inplace=True)
        df = df.convert_objects(convert_numeric=True)
        df = df.dropna()
        df = df.astype(np.float32)
        store.put(str(key), df)
        print("Done with file #", (i + 1))
    store.close()
    metadata_path = join(_get_module_directory(), 'metadata')
    print('Processing metadata...')
    convert_yaml_to_hdf5(metadata_path, output_filename)