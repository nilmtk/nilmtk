from __future__ import print_function, division
import pandas as pd
import numpy as np
from pandas import *
from os.path import *
from os import listdir
from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilm_metadata import *
from nilmtk.utils import get_module_directory

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


def convert_ampds(inputPath, hdfFilename):
    '''
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

    '''
    files = [f for f in listdir(inputPath) if isfile(join(inputPath, f)) and '.csv' in f and '.swp' not in f]
    assert isdir(inputPath)
    store = HDFStore(hdfFilename)
    for i, csv_file in enumerate(files):  
        key = Key(building=1, meter=(i + 2))
        print('Loading file #', (i + 1), ' : ', csv_file, '. Please wait...')
        df = pd.read_csv(join(inputPath, csv_file))
        df.index = pd.to_datetime(df["TIMESTAMP"], unit='s')
        df = df.drop('TIMESTAMP', 1)
        df.rename(columns=lambda x: columnNameMapping[x], inplace=True)
        df.columns.set_names(LEVEL_NAMES, inplace=True)
        df = df.convert_objects(convert_numeric=True)
        df = df.dropna()
        df = df.astype(np.float32)
        store.put(str(key), df, format='Table')
        store.flush()
        print("Done with file #", (i + 1))
    store.close()
    metadataPath = join(get_module_directory(), 'metadata')
    print('Processing metadata...')
    convert_yaml_to_hdf5(metadataPath, hdfFilename)