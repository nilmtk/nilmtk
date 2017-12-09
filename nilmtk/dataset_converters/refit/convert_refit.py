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


def convert_refit(input_path, output_filename, format='HDF',verbose=True):
    """
    Parameters
    ----------
    input_path : str
        The root path of the CSV files, e.g. House1.csv
    output_filename : str
        The destination filename (including path and suffix).
    format : str
        format of output. Either 'HDF' or 'CSV'. Defaults to 'HDF'
    verbose: boolean
        Ture for more detailed diagnostic output and help
    """
        
    # Open DataStore
    store = get_datastore(output_filename, format, mode='w')

    # Convert raw data to DataStore
    _convert(input_path, store, 'Europe/London',verbose)
# MEN
    if verbose:
        print ('Note this converter uses a localisation of Europe/London which must be consistent with the timezone variable in the dataset.yaml file' )
    # Add metadata
    save_yaml_to_datastore(join(get_module_directory(), 
                              'dataset_converters', 
                              'refit', 
                              'metadata'),
                         store)
    store.close()

    print("Done converting REFIT to HDF5!")

def _convert(input_path, store, tz, verbose,sort_index=True):
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
    if verbose:
        print ('Files names begin with: CLEAN_House')  
        print
    check_directory_exists(input_path)
    columns = ['Timestamp_datetime','Timestamp','Aggregate','Appliance1','Appliance2','Appliance3','Appliance4','Appliance5','Appliance6','Appliance7','Appliance8','Appliance9','Issues']
    if verbose:
        print ('\n Expected column names: \n' + str(columns) + '\n')
    # Iterate though all houses and channels
    # house 14 is missing!
    houses = [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21]
    nilmtk_house_id = 0
    for house_id in houses:
        nilmtk_house_id += 1
        print("Loading house", house_id, end="... ")
        csv_filename = input_path + 'CLEAN_House' + str(house_id) + '.csv'
        df = _load_csv(csv_filename, columns, tz,verbose)
        print ('Meter:',end=' ')
        stdout.flush()
        if sort_index:
            df = df.sort_index() # might not be sorted...
        chan_id = 0
        for col in df.columns:
            chan_id += 1
            print(chan_id, end=" ")
            stdout.flush()
            key = Key(building=nilmtk_house_id, meter=chan_id)
            
            chan_df = pd.DataFrame(df[col])
            df_nulls = ((len(chan_df) - chan_df.count()[0])*1.0)/(len(chan_df) *1.0)
            if verbose and df_nulls>0:
                print ('%% null values in column ' + col + ' %0.3f' % (df_nulls*100))
            chan_df.columns = pd.MultiIndex.from_tuples([('power', 'active')])
            
            # Modify the column labels to reflect the power measurements recorded.
            chan_df.columns.set_names(LEVEL_NAMES, inplace=True)
            
            store.put(str(key), chan_df)
        print('')

def _load_csv(filename, columns, tz,verbose):
    """
    Parameters
    ----------
    filename : str
    columns : list of tuples (for hierarchical column index)
    tz : str e.g. 'US/Eastern'

    Returns
    -------
    dataframe
    """
    # Load data
    df = pd.read_csv(filename, names=columns,skiprows=1)
    df_issues = (df.Issues.sum()*1.0)/(len(df)*1.0)
    if verbose and df_issues>0:
        print ('Records with issues dropped: %0.3f%%' % (df_issues*100))
    df = df[df.Issues ==0]
    df = df.drop(['Timestamp_datetime','Issues'],axis=1)
    # Convert the integer index column to timezone-aware datetime 
    df['Timestamp'] = pd.to_datetime(df.Timestamp, unit='s', utc=True)
    df.set_index('Timestamp', inplace=True)
    df = df.tz_localize('GMT').tz_convert(tz)

    return df
