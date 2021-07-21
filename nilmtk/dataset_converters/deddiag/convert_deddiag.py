from os.path import join
from sys import stdout
from nilmtk.utils import get_datastore
from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import get_module_directory
from nilm_metadata import convert_yaml_to_hdf5, save_yaml_to_datastore
import pandas as pd

from pandas import Timestamp

# check whether deddiag-loader package has already installed
try:
    from deddiag_loader import Connection, Items, MeasurementsExpanded
except ImportError:
    raise ImportError('Please install package deddiag-loader with "pip install deddiag-loader"')

# DEDDIAG measurements [[(physical_quantity, ac_type)]]
measurements_conf = [['power'], ['active']]

DEFAULT_TZ = 'Europe/Berlin'

# Analyzed time period
DEFAULT_START_DATE = Timestamp('2017-10-21 00:00:00', freq='MS')
DEFAULT_END_DATE = Timestamp('2018-01-18 23:59:59', freq='MS')

# channels
channels = [24, 26, 27, 28, 35, 51, 52, 53, 59]

# house
house_nr = 8


def convert_deddiag(connection,
                    output_filename,
                    format='HDF',
                    start_date=DEFAULT_START_DATE,
                    end_date=DEFAULT_END_DATE,
                    tz=DEFAULT_TZ):
    """
    Parameters
    ----------
    connection: Connection
        Connection to the DEDDIAG database
        Example: connection = Connection(host="localhost", port="5432", db_name="postgres", user="postgres", password="password")
    output_filename : str
        The destination filename including path and suffix
        Example: ./data/deddiag.h5
    format : str
        format of output. Either 'HDF' or 'CSV'. Defaults to 'HDF'
    """

    # Open DataStore
    # todo try catch

    dest_file = get_datastore(output_filename, format, mode='w')

    # Convert raw data to DataStore
    _convert(connection, dest_file, start_date, end_date, tz)

    path_to_metadata = join(get_module_directory(), 'dataset_converters', 'deddiag', 'metadata')

    # Add metadata
    save_yaml_to_datastore(path_to_metadata, dest_file)

    print("Done converting DEDDIAG to HDF5!")


def _convert(connection, dest_file, start_date, end_date, tz, sort_index=True):
    """
    Parameters
    ----------
    connection: Connection
        Connection to the DEDDIAG database
    dest_file : DataStore
        The NILMTK DataStore object
    tz : str
        Timezone e.g. 'Europe/Berlin'
    sort_index : bool
        Defaults to True
    """

    print(f"Loading house {house_nr}", end="... ")
    stdout.flush()

    # Find all houses and channels
    for channel in channels:
        print(f"{channel}", end=" ")
        stdout.flush()

        measurements = MeasurementsExpanded(channel, start_date, end_date).request(connection)
        measurements.drop(columns='item_id', inplace=True)
        measurements['time'] = pd.to_datetime(measurements['time'], utc=True, unit='s')
        measurements.set_index('time', inplace=True)
        # set index und columns as LEVEL_NAMES
        measurements = measurements.tz_convert(tz)
        measurements.columns = pd.MultiIndex.from_arrays(measurements_conf,
                                                         names=LEVEL_NAMES)  # measurements_conf = [['power'], ['active']]

        if sort_index:
            measurements.sort_index(inplace=True)

        key = Key(building=house_nr, meter=channel)
        # write data
        dest_file.put(str(key), measurements)

