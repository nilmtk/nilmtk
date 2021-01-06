from __future__ import print_function, division

import numpy as np
import pandas as pd
import csv
from os.path import join, isfile, isdir

from os import listdir, getcwd
from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import check_directory_exists, get_datastore, get_module_directory
from nilm_metadata import convert_yaml_to_hdf5
from datetime import datetime

def convert_rae(input_path, output_filename='RAE.h5', format='HDF'):

    """
    Parameters
    ----------
    input_path : str
        The path of the RAE active power dataset.
    output_filename : str
        The destination filename (including path and suffix).
    format : str
        format of output. Either 'HDF' or 'CSV'. Defaults to 'HDF'
    """

    """
        Convert the data
    """
    input_path = join(getcwd(), input_path)

    check_directory_exists(input_path)
    files = [f for f in listdir(input_path) if isfile(join(input_path, f)) and
             '.csv' in f and '.swp' not in f]

    files.sort()
    assert isdir(input_path)

    store = get_datastore(output_filename, format, mode='w')
    print('\n\n\n')
    for i, csv_file in enumerate(files):
        metadata_path = join(get_module_directory(), 'dataset_converters', 'rae', 'metadata')
        if 'power' in csv_file:
            print('Loading file #', (i + 1), ': ', csv_file, '. Please wait...')
            df_rae = pd.read_csv(join(input_path, csv_file))
            if 'house1' in csv_file:
                active_power_loader(df_rae, store, building=1)
            elif 'house2' in csv_file:
                active_power_loader(df_rae, store, building=2)
            print("Done with file #", (i + 1), '\n\n')
        else:
            print('Skipping File', (i+1), 'because it doesn`t contain power readings.')
            print('Please do not change the original file names.')

    print('Processing metadata...')
    convert_yaml_to_hdf5(metadata_path, output_filename)

"""
    Function for converting active power
"""
def active_power_loader(df_rae, store, building=1):
    """
    Parameters
    ----------
    df_rae : pd.DataFrame
        The pandas dataframe containing all the submeters.
    store : Datastore
        The NILMTK DataStore object.
    format : int
        The building number, i.e., building=1
    """
    TIMESTAMP_COLUMN_NAME = "unix_ts"
    TIMEZONE = "America/Vancouver"
    df_rae.index = pd.to_datetime(df_rae[TIMESTAMP_COLUMN_NAME], unit='s', utc=True)
    df_rae = df_rae.drop(['unix_ts', 'ihd'], 1)
    if building == 1:
        columnsTitles = ['mains', 'sub1+sub2', 'sub3+sub4', 'sub5+sub6', 'sub7', 'sub8', 'sub9', 'sub10', 'sub11',
                         'sub12', 'sub13+sub14', 'sub15+sub16', 'sub17+sub18', 'sub19', 'sub20', 'sub21+sub22',
                         'sub23', 'sub24' ]

        df_rae['sub1+sub2'] = df_rae['sub1'] + df_rae['sub2']
        df_rae['sub3+sub4'] = df_rae['sub3'] + df_rae['sub4']
        df_rae['sub5+sub6'] = df_rae['sub5'] + df_rae['sub6']
        df_rae['sub13+sub14'] = df_rae['sub13'] + df_rae['sub14']
        df_rae['sub15+sub16'] = df_rae['sub15'] + df_rae['sub16']
        df_rae['sub17+sub18'] = df_rae['sub17'] + df_rae['sub18']
        df_rae['sub21+sub22'] = df_rae['sub21'] + df_rae['sub22']
        df_rae = df_rae.drop(['sub1', 'sub2', 'sub3', 'sub4', 'sub5', 'sub6', 'sub13', 'sub14', 'sub15',
                              'sub16', 'sub17', 'sub18', 'sub21', 'sub22'], 1)
        df_rae = df_rae.reindex(columns=columnsTitles)
    elif building == 2:
        columnsTitles = ['mains', 'sub3', 'sub4+sub5', 'sub6', 'sub7', 'sub8', 'sub9', 'sub10', 'sub11', 'sub12',
                        'sub13', 'sub14', 'sub15', 'sub16', 'sub17', 'sub18', 'sub19', 'sub20', 'sub21']

        df_rae = df_rae.drop(['sub1', 'sub2'], 1)
        df_rae['sub4+sub5'] = df_rae['sub4'] + df_rae['sub5']
        df_rae.drop(['sub4', 'sub5'], 1)
        df_rae = df_rae.reindex(columns=columnsTitles)
    for i, column in enumerate(df_rae.columns):
        key = Key(building=building, meter=(i+1)) # generate the key
        df = pd.DataFrame(data = df_rae[column], index=df_rae.index)
        df = df.tz_convert(TIMEZONE)
        df.columns = pd.MultiIndex.from_tuples([('power', 'active')], names=LEVEL_NAMES)
        df = df.apply(pd.to_numeric, errors='ignore')
        df = df.dropna()
        df = df.astype(np.float32)
        store.put(str(key), df)
    return 0
