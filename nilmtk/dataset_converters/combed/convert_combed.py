from os.path import join, isdir, dirname, abspath
from os import getcwd
import os
from sys import getfilesystemencoding
from collections import OrderedDict
import pandas as pd
from nilm_metadata import convert_yaml_to_hdf5
from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import check_directory_exists, get_datastore, get_module_directory

#{"load_type": {"floor/wing":meter_number_in_nilmtk}
acad_block_meter_mapping = {'Building Total Mains': {'0': 1},
                            'Lifts': {'0': 2},
                            'Floor Total': {'1': 3, '2': 4, '3': 5, '4': 6, '5': 7},
                            'AHU': {'0': 8, '1': 9, '2': 10, '5': 11},
                            'Light': {'3': 12},
                            'Power Sockets': {'3': 13},
                            'UPS Sockets': {'3': 14}}

lecture_block_meter_mapping = {'Building Total Mains': {'0': 1},
                               'Floor Total': {'0': 2, '1': 3, '2': 4},
                               'AHU': {'1': 5, '2': 6, '3': 7}}

overall_dataset_mapping = OrderedDict({'Academic Block': acad_block_meter_mapping,
                                       'Lecture Block': lecture_block_meter_mapping})

building_number_mapping = {'Academic Block': 1, 'Lecture Block': 2}


column_mapping = OrderedDict({
    'Power': ('power', 'active'),
    'Energy': ('energy', 'active'),
    'Current': ('current', '')})


def convert_combed(combed_path, output_filename, format='HDF'):
    """
    Parameters
    ----------
    combed_path : str
        The root path of the combed dataset.
    output_filename : str
        The destination HDF5 filename (including path and suffix).
    """

    check_directory_exists(combed_path)

    # Open store
    store = get_datastore(output_filename, format, mode='w')

    any_file_converted = False
    
    for building_name, building_mapping in overall_dataset_mapping.items():
        for load_name, load_mapping in building_mapping.items():
            for load_mapping_path, meter_number in load_mapping.items():
                building_number = building_number_mapping[building_name]
                key = Key(building=building_number, meter=meter_number)
                dfs = []
                for attribute in column_mapping.keys():
                    filename_attribute = join(combed_path, building_name, load_name, load_mapping_path, "%s.csv" %attribute)
                    if not os.path.isfile(filename_attribute):
                        # File not found directly in the combed_path provided
                        # Try adding 'iiitd' to it
                        filename_attribute = join(combed_path, 'iiitd', building_name, load_name, load_mapping_path, "%s.csv" %attribute)
                    
                    if os.path.isfile(filename_attribute):
                        exists = True
                        print(filename_attribute)
                        df = pd.read_csv(filename_attribute, names=["timestamp", attribute])
                        df.index = pd.to_datetime(df["timestamp"], unit='ms')
                        df = df.drop("timestamp", 1)
                        dfs.append(df)
                    else:
                        exists = False
                        
                if exists:
                    total = pd.concat(dfs, axis=1)
                    total = total.tz_localize('UTC').tz_convert('Asia/Kolkata')
                    total.columns = pd.MultiIndex.from_tuples([column_mapping[x] for x in total.columns])
                    total.columns.set_names(LEVEL_NAMES, inplace=True)
                    assert total.index.is_unique
                    store.put(str(key), total)
                    any_file_converted = True
                    
    if not any_file_converted:
        raise RuntimeError('No files converted, did you specify the correct path?')
                    
    convert_yaml_to_hdf5(
        join(get_module_directory(), 'dataset_converters', 'combed', 'metadata'),
        output_filename
    )

    print("Done converting COMBED to HDF5!")
