'''
REFIT dataset converter for the clean version avaiable at the URLs below:

"REFIT: Electrical Load Measurements (Cleaned)"
https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned
https://pureportal.strath.ac.uk/files/52873459/Processed_Data_CSV.7z
https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z

The original version of the dataset include duplicated timestamps. 
Check the dataset website for more information.

For citation of the dataset, use:
http://dx.doi.org/10.1038/sdata.2016.122
https://doi.org/10.15129/9ab14b0e-19ac-4279-938f-27f643078cec

'''

import pandas as pd
import numpy as np
from copy import deepcopy
from os.path import join, isdir, isfile, exists
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


def convert_refit(input_path, output_filename, format='HDF'):
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
    _convert(input_path, store, 'Europe/London')

    # Add metadata
    save_yaml_to_datastore(join(get_module_directory(), 
                              'dataset_converters', 
                              'refit', 
                              'metadata'),
                         store)
    store.close()

    print("Done converting REFIT to HDF5!")

def _convert(input_path, store, tz, sort_index=True):
    """
    Parameters
    ----------
    input_path : str
        The root path of the REFIT dataset.
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
    """

    check_directory_exists(input_path)

    # Iterate though all houses and channels
    # house 14 is missing!
    houses = [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21]
    nilmtk_house_id = 0
    prefix = ''
    suffix = '_'
    version_checked = False
    
    for house_id in houses:
        nilmtk_house_id += 1
        print("Loading house", house_id, end="... ")
        stdout.flush()
        csv_filename = join(input_path, prefix + 'House' + suffix + str(house_id) + '.csv')
        if not version_checked:
            version_checked = True
            
            if exists(csv_filename):
                print('Using original filenames (House_XX.csv)')
            else:
                prefix = 'CLEAN_'
                suffix = ''
                csv_filename = join(input_path, prefix + 'House' + suffix + str(house_id) + '.csv')
                print('Using CLEAN filenames (CLEAN_HouseXX.csv)')
                
        if not exists(csv_filename):
            raise RuntimeError('Could not find REFIT files. Please check the provided folder.')
        
        # The clean version already includes header, so we
        # just skip the text version of the timestamp
        usecols = ['Unix','Aggregate','Appliance1','Appliance2','Appliance3','Appliance4','Appliance5','Appliance6','Appliance7','Appliance8','Appliance9']
        
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
            chan_df.columns = pd.MultiIndex.from_tuples([('power', 'active')])
            
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
    df['Unix'] = pd.to_datetime(df['Unix'], unit='s', utc=True)
    df.set_index('Unix', inplace=True)
    df = df.tz_convert(tz)

    return df
