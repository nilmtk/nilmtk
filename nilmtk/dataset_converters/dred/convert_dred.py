'''
DRED Dataset converter.
The .h5 format is hosted in DRED official website. But the file is not fully compatible with NILMTK.
Download the .h5 file directly or download .csv and convert to .h5 using this converter.

https://drive.google.com/open?id=1NDiRGVb33SQaKL_W7Y6WXEs9_xR1xTij

CSV file name :- csvmerged.csv
.h5 file name :- book1DRED.h5

'''

from __future__ import print_function, division
import pandas as pd
import numpy as np
from copy import deepcopy
from os.path import join, isdir, isfile
from os import listdir
import fnmatch
import re
from sys import stdout
from nilmtk.utils import get_datastore
from nilmtk.datastore import Key
from nilmtk.timeframe import TimeFrame
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import get_module_directory, check_directory_exists
from nilm_metadata import convert_yaml_to_hdf5, save_yaml_to_datastore


def convert_dred(input_path, output_filename, format='HDF'):
    """
    Parameters
    ----------
    input_path : str
        The root path of the CSV files, e.g. House1.csv
    output_filename : str
        The destination filename (including path and suffix).
    format : str
        format of output. Either 'HDF' or 'CSV'. Defaults to 'HDF'
    """
        
    # Open DataStore
    store = get_datastore(output_filename, format, mode='w')

    # Convert raw data to DataStore
    _convert(input_path, store, 'Europe/Amsterdam')

    # Add metadata
    save_yaml_to_datastore(join(get_module_directory(), 
                              'dataset_converters', 
                              'dred', 
                              'metadata'),
                         store)
    store.close()

    print("Done converting DRED to HDF5!")

def _convert(input_path, store, tz, sort_index=True):
    """
    Parameters
    ----------
    input_path : str
        The root path of the DRED dataset.
    store : DataStore
        The NILMTK DataStore object.
    measurement_mapping_func : function
        Must take these parameters:
            - house_id
            - chan_id
        Function should return a list of tuples e.g. [('power', 'apparent')]
    tz : str 
        Timezone e.g. 'Europe/Amsterdam'
    sort_index : bool
    """

    check_directory_exists(input_path)

    # Iterate though all houses and channels
    # house 14 is missing!
    houses = [1]
    nilmtk_house_id = 0
    for house_id in houses:
        nilmtk_house_id += 1
        print("Loading house", house_id, end="... ")
        stdout.flush()
        csv_filename = join(input_path, 'csvmerged' + '.csv')
        # The clean version already includes header, so we
        # just skip the text version of the timestamp
        usecols = ['unix',
                   'mains','television',
                   'fan','fridge',
                   'laptop computer','electric heating element',
                   'oven','unknown',
                   'washing machine','microwave',
                   'toaster','sockets','cooker'
                  ]
        
        df = _load_csv(csv_filename, usecols, tz)
        if sort_index:
            df = df.sort_index() # might not be sorted...
        chan_id = 0
        for col in df.columns:
            chan_id += 1
            print(chan_id, end=" ")
            stdout.flush()
            key = Key(building=nilmtk_house_id, meter=chan_id)
            
            chan_df = pd.DataFrame(df[col])
            chan_df.columns = pd.MultiIndex.from_tuples([('power', 'apparent')])
            
            # Modify the column labels to reflect the power measurements recorded.
            chan_df.columns.set_names(LEVEL_NAMES, inplace=True)
            
            store.put(str(key), chan_df)
        print('')

def _load_csv(filename, usecols, tz):
    """
    Parameters
    ----------
    filename : str
    usecols : list of columns to keep
    tz : str e.g. 'US/Eastern'

    Returns
    -------
    dataframe
    """
    # Load data
    df = pd.read_csv(filename, usecols=usecols)
    
    # Convert the integer index column to timezone-aware datetime 
    df['unix'] = pd.to_datetime(df['unix'], unit='s', utc=True)
    df.set_index('unix', inplace=True)
    df = df.tz_convert(tz)

    return df