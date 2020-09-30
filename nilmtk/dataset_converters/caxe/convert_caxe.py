import pandas as pd
import numpy as np
from nilmtk.datastore import Key
from nilmtk.utils import check_directory_exists, get_datastore, get_module_directory
from nilm_metadata import convert_yaml_to_hdf5

TIMESTAMP_COLUMN_NAME = "timestamp"
TIMEZONE = "Asia/Kolkata"
FREQ = "1T"
LEVEL_NAMES = ['physical_quantity', 'type']

def convert_caxe(file_path):
    '''
    Parameters
    ------------
    Takes input csv_file name to be tested as string.
    Data columns of the csv should contain following the following values in columns:
    timestamp,reactive_power,apparent_power,current,frequency,voltage,active_power) 
    Converts it into hdf5 Format and save as test.h5.
    '''
    df = pd.read_csv(f'{file_path}',names =['timestamp','R','A','C','F','V','T'])
    column_mapping = {
        'F': ('frequency', ""),
        'V': ('voltage', ""),
        'T': ('power', 'active'),
        'C': ('current', ''),
        'R': ('power', 'reactive'),
        'A': ('power', 'apparent'),
    }


    output_filename = 'test.h5'

    # Open data store
    store = get_datastore(output_filename, format='HDF', mode='w')
    key = Key(building=1, meter=1)
    print('Loading ', 1)
    df.index = pd.to_datetime(df.timestamp.values)
    df = df.tz_convert(TIMEZONE) #  if error occurs use tz_localize for tz naive timestamps
    df = df.drop(TIMESTAMP_COLUMN_NAME, 1)
    df.index = pd.to_datetime(df.index.values)
    df.columns = pd.MultiIndex.from_tuples(
                [column_mapping[x] for x in df.columns],
                names=LEVEL_NAMES
            )
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.dropna()
    df = df.astype(np.float32)
    df = df.sort_index()
    df = df.resample("1T").mean()
    assert df.isnull().sum().sum() == 0
    store.put(str(key), df)
    store.close()
    convert_yaml_to_hdf5('./metadata', output_filename)

    print("Done converting test data to HDF5!")





