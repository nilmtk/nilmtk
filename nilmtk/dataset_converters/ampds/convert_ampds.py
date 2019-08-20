import numpy as np
import pandas as pd
from os.path import *
from os import getcwd
from os import listdir
from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import check_directory_exists, get_datastore, get_module_directory
from nilm_metadata import convert_yaml_to_hdf5
from sys import getfilesystemencoding

# Column name mapping
columnNameMapping = {'V': ('voltage', ''),
                     'I': ('current', ''),
                     'f': ('frequency', ''),
                     'DPF': ('power factor', 'real'),
                     'APF': ('power factor', 'apparent'),
                     'P': ('power', 'active'),
                     'Pt': ('energy', 'active'),
                     'Q': ('power', 'reactive'),
                     'Qt': ('energy', 'reactive'),
                     'S': ('power', 'apparent'),
                     'St': ('energy', 'apparent')}

TIMESTAMP_COLUMN_NAME = "TS"
TIMEZONE = "America/Vancouver"

def convert_ampds(input_path, output_filename, format='HDF'):
    """
    Convert AMPds R2013 as seen on Dataverse. Download the files
    as CSVs and put them in the `input_path` folder for conversion.
    
    Download URL: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MXB7VO
    
    Parameters: 
    -----------
    input_path: str
            The path of the directory where all the csv 
            files are supposed to be stored
    output_filename: str
            The path of the h5 file where all the 
            standardized data is supposed to go. The path 
            should refer to a particular file and not just a
             random directory in order for this to work.
    format: str
        Defaults to HDF5
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
        df = df.tz_convert(TIMEZONE)
        df.columns = pd.MultiIndex.from_tuples(
            [columnNameMapping[x] for x in df.columns],
            names=LEVEL_NAMES
        )
        df = df.apply(pd.to_numeric, errors='ignore')
        df = df.dropna()
        df = df.astype(np.float32)
        store.put(str(key), df)
        print("Done with file #", (i + 1))
        
    store.close()
    metadata_path = join(get_module_directory(), 'dataset_converters', 'ampds', 'metadata')
    print('Processing metadata...')
    convert_yaml_to_hdf5(metadata_path, output_filename)
