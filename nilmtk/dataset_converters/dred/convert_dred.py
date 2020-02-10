'''
DRED Dataset converter.
The .h5 format is hosted in DRED official website. But the file is not fully compatible with NILMTK.

Download All_data.csv from the official website and use this converter

Official Website :- http://www.st.ewi.tudelft.nl/~akshay/dred/

'''

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
import pickle
import glob
import numpy as np
import time
from datetime import datetime


def convert_dred(input_path, output_filename, format='HDF'):
    """
    Parameters
    ----------
    input_path : str
        The root path of the CSV files, e.g. All_data.csv
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


def _convert(csv_filename, store, tz, sort_index=True):
    """
    Parameters
    ----------
    csv_filename : str
        The csv_filename that will be loaded. Must end with .csv
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

    # Iterate though all houses and channels
    houses = [1]
    nilmtk_house_id = 0
    for house_id in houses:
        nilmtk_house_id += 1
        print("Loading house", house_id, end="... ")
        stdout.flush()
        
        usecols=['Timestamp','mains',
                 'television','fan','fridge',
                 'laptop computer','electric heating element',
                 'oven','unknown','washing machine',
                 'microwave','toaster',
                 'sockets','cooker'
                ]
        df = _load_csv(csv_filename, usecols, 3, tz)

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


def _load_csv(filename, usecols, skip, tz):
    """
    Parameters
    ----------
    filename : str
    usecols : list of columns to keep
    skip : number of columns to skip from beginning. 3 rows are irrelevant in .csv file
    tz : str e.g. 'Europe/Amsterdam'

    Returns
    -------
    dataframe
    """
    # Load data
    df = pd.read_csv(filename, skiprows=skip, header=None)
    df.columns = usecols
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
    df.set_index('Timestamp', inplace=True)
    df = df.tz_convert(tz)
    
    return df
