import os
import re
import datetime
import shutil
import sys
from os.path import join, isdir, isfile, dirname, abspath
import pandas as pd
import yaml
import psycopg2 as db
from nilmtk.measurement import measurement_columns
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.datastore import Key
from nilm_metadata import convert_yaml_to_hdf5
from nilmtk.utils import check_directory_exists, get_datastore, get_module_directory
import shutil
import tempfile

# This script is based upon download_dataport.py
# We import from it all the common functions and variables
from nilmtk.dataset_converters.dataport.download_dataport import feed_mapping, feed_ignore
from nilmtk.dataset_converters.dataport.download_dataport import database_assert, _dataport_dataframe_to_hdf    
    

def convert_dataport(input_path, hdf_filename,
                     user_selected_table='eg_realpower_1s',
                     time_column="localminute"):
    """Converts the Pecan Dataport sample dataset to NILMTK HDF5 format.

    For more information about the Pecan Dataport dataset, and to download
    it, please see https://www.pecanstreet.org/dataport/

    Parameters
    ----------
    input_path : str
        The root path of the Pecan Dataport dataset, where all
        the csv of 1Hz (1 second) frequency should be contained.
    hdf_filename : str
        The destination filename (including path and suffix).
    user_selected_table: str
    time_column: str
    """
    # Check if input directory exist
    check_directory_exists(input_path)
    # List csv files in directory
    # We will only use the ones of 1Hz frequency
    files = [f for f in listdir(input_path) if isfile(join(input_path, f)) and
             '.csv' in f and '.gz' not in f and '1s' in f]
    # Sorting Lexicographically
    files.sort()
    
    # Assert that the type of selected table exist
    database_assert(user_selected_table)
    
    # Set up a new HDF5 datastore (overwrites existing store)
    store = pd.HDFStore(hdf_filename, 'w', complevel=9, complib='zlib')

    # Create a temporary metadata dir, remove existing building
    # yaml files in module dir (if any)
    original_metadata_dir = join(get_module_directory(),
                                 'dataset_converters',
                                 'dataport',
                                 'metadata')
    tmp_dir = tempfile.mkdtemp()
    metadata_dir = join(tmp_dir, 'metadata')
    shutil.copytree(original_metadata_dir, metadata_dir)
    print("Using temporary dir for metadata:", metadata_dir)

    for f in os.listdir(metadata_dir):
        if re.search('^building', f):
            os.remove(join(metadata_dir, f))
    
    # Create a subdirectory called building in the temp directory
    # We will use it to store individual building csv
    build_tmp_dir = tmp_dir + "/building"
    try:
        os.mkdir(build_tmp_dir)
    except:
        shutil.rmtree(build_tmp_dir)
        os.mkdir(build_tmp_dir)
        
    # Iterate through all the csv files
    # Export a csv for each individual building
    for file in files:
        # Load the csv as an iterator over a dataframe
        file_path = input_path + "/" + file
        data_iter = pd.read_csv(file_path, header=0, chunksize=1)
        # Iterate through each row of the dataframe
        for row in data_iter:
            # Get dataid - it is different for each household
            dataid = row["dataid"].values[0]
            # Locate csv path
            path_csv = build_tmp_dir + "/" + str(dataid) + ".csv"
            try:
                # If csv exist for that dataid, append the new row
                row.to_csv(path_csv, mode='a', header=True)
            except:
                # If it doesn't exist, create new csv
                row.to_csv(path_csv, index=False)
    
    # Initialize nilmtk building id (it requires to start at 1)
    nilmtk_building_id = 0
    
    # Iterate through each building
    buildings = os.listdir(build_tmp_dir)
    # Sorting Lexicographically
    buildings.sort()
    for file in buildings:
        building_id = int(file.rsplit(".", 1)[0])
        print("Loading building {:d} @ {}".format(building_id, datetime.datetime.now()))
        sys.stdout.flush()
        dataframe = pd.read_csv(build_tmp_dir + "/" + file, index_col=0)
        # Convert date column from string to date format
        dataframe[time_column] = pd.to_datetime(dataframe[time_column])
        # Update nilmtk id
        nilmtk_building_id += 1
        # convert to nilmtk-df and save to disk
        nilmtk_dataframe = _dataport_dataframe_to_hdf(
            dataframe, store,
            nilmtk_building_id,
            building_id,
            time_column,
            metadata_dir,
            user_selected_table                           
        )

    store.close()

    # write yaml to hdf5
    # dataset.yaml and meter_devices.yaml are static, building<x>.yaml are dynamic
    convert_yaml_to_hdf5(metadata_dir, hdf_filename)

    # remote the temporary dir when finished
    shutil.rmtree(tmp_dir)
