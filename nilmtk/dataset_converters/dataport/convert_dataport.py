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

# Load the appliances list from the download script
from nilmtk.dataset_converters.dataport.download_dataport import feed_mapping, feed_ignore, _dataport_dataframe_to_hdf


def convert_dataport(input_path, hdf_filename):
    """Converts the Pecan Dataport dataset to NILMTK HDF5 format.

    For more information about the Pecan Dataport dataset, and to download
    it, please see https://www.pecanstreet.org/dataport/

    Parameters
    ----------
    input_path : str
        The root path of the Pecan Dataport dataset, where all
        the csv should be contained.
    hdf_filename : str
        The destination filename (including path and suffix).
    """
    # Check if directory exist
    check_directory_exists(input_path)
    # List files in directory
    files = [f for f in listdir(input_path) if isfile(join(input_path, f)) and
             '.csv' in f and '.gz' not in f]
    
    # Sorting Lexicographically
    files.sort()
    
    # set up a new HDF5 datastore (overwrites existing store)
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
    
    # Initialize nilmtk building id (it requires to start at 1)
    nilmtk_building_id = 0
    last_building_id = -1
    # Iterate through all the csv files
    for i, file in enumerate(files):
        # Load the dataframe
        dataframe = pd.read_csv(file)
        # List buildings ids
        unique_dataid = dataframe["dataid"].unique()
        # Iterate through each building
        for building_id in unique_dataid:
            print("Loading building {:d} @ {}".format(building_id, datetime.datetime.now()))
            sys.stdout.flush()
            chunck_dataframe = dataframe[dataframe["dataid"] == building_id].copy()
            # Update nilmtk id
            if building_id != last_building_id:
                nilmtk_building_id += 1
            # convert to nilmtk-df and save to disk
            nilmtk_dataframe = _dataport_dataframe_to_hdf(
                chunk_dataframe, store,
                nilmtk_building_id,
                building_id,
                timestamp_map[user_selected_table], # TODO: Define this
                metadata_dir,
                user_selected_table  # TODO: Define this                            
            )

    store.close()

    # write yaml to hdf5
    # dataset.yaml and meter_devices.yaml are static, building<x>.yaml are dynamic
    convert_yaml_to_hdf5(metadata_dir, hdf_filename)

    # remote the temporary dir when finished
    shutil.rmtree(tmp_dir)