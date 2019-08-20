import pandas as pd
import numpy as np
from os.path import join
from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import check_directory_exists, get_datastore, get_module_directory
from nilm_metadata import convert_yaml_to_hdf5
from copy import deepcopy

def reindex_fill_na(df, idx):
    df_copy = deepcopy(df)
    df_copy = df_copy.reindex(idx)

    power_columns = [
        x for x in df.columns if x[0] in ['power']]
    non_power_columns = [x for x in df.columns if x not in power_columns]

    for power in power_columns:
        df_copy[power].fillna(0, inplace=True)
    for measurement in non_power_columns:
        df_copy[measurement].fillna(df[measurement].median(), inplace=True)

    return df_copy


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
START_DATETIME, END_DATETIME = '2013-07-13', '2013-08-04'
FREQ = "1T"


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
    idx = pd.date_range(start=START_DATETIME, end=END_DATETIME, freq=FREQ)
    idx = idx.tz_localize('GMT').tz_convert(TIMEZONE)

    # Open data store
    store = get_datastore(output_filename, format, mode='w')
    electricity_path = join(iawe_path, "electricity")

    # Mains data
    for chan in range(1, 12):
        key = Key(building=1, meter=chan)
        filename = join(electricity_path, "%d.csv" % chan)
        print('Loading ', chan)
        df = pd.read_csv(filename, dtype=np.float64, na_values='\\N')
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.index = pd.to_datetime(df.timestamp.values, unit='s', utc=True)
        df = df.tz_convert(TIMEZONE)
        df = df.drop(TIMESTAMP_COLUMN_NAME, 1)
        df.columns = pd.MultiIndex.from_tuples(
            [column_mapping[x] for x in df.columns],
            names=LEVEL_NAMES
        )
        df = df.apply(pd.to_numeric, errors='ignore')
        df = df.dropna()
        df = df.astype(np.float32)
        df = df.sort_index()
        df = df.resample("1T").mean()
        df = reindex_fill_na(df, idx)
        assert df.isnull().sum().sum() == 0
        store.put(str(key), df)
    store.close()
    
    metadata_dir = join(get_module_directory(), 'dataset_converters', 'iawe', 'metadata')
    convert_yaml_to_hdf5(metadata_dir, output_filename)

    print("Done converting iAWE to HDF5!")

