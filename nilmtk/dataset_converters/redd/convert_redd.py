from __future__ import print_function, division
import pandas as pd
import numpy as np
from copy import deepcopy
from os.path import join, isdir, isfile
from nilmtk.datastore import Key
from nilmtk.measurement import Power
from nilmtk.timeframe import TimeFrame

"""
TODO:
* Load just the data into HDF5.
* Use a hand-written set of .YAML files for metadata.
* Write general function (in NILM Metadata???) for converting YAML to HDF5
* remove any imports we don't need.
"""


def find_all_houses(redd_path):
    pass


def load_chan(redd_path, key_obj, measurement=Power('active')):
    """
    Parameters
    ----------
    redd_path : (str) the root path of the REDD low_freq dataset
    key_obj : (nilmtk.Key) the house and channel to load
    measurement : (nilmtk.measurement) (optional)

    Returns
    ------- 
    DataFrame of data. Has extra attributes:
        - timeframe : TimeFrame of period intersected with self.window
    """
    assert isinstance(key_obj, Key)

    # Get path
    house_path = 'house_{:d}'.format(key_obj.building)
    path = join(redd_path, house_path)
    assert isdir(path)

    # Get filename
    filename = 'channel_{:d}.dat'.format(key_obj.meter)
    filename = join(path, filename)
    assert isfile(filename)

    # Load data
    df = pd.read_csv(filename, sep=' ', index_col=0,
                     names=[measurement], 
                     tupleize_cols=True, # required to use Power('active')
                     dtype={measurement: np.float32})

    # Basic post-processing
    df = df.sort_index() # raw REDD data isn't always sorted
    df.index = pd.to_datetime((df.index.values*1E9).astype(int), utc=True)
    df = df.tz_convert('US/Eastern')
    df.timeframe = TimeFrame(df.index[0], df.index[-1])
    df.timeframe.include_end = True
    return df
