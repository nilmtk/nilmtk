from __future__ import print_function, division
from os.path import join, isdir, dirname, abspath
from os import getcwd
from sys import getfilesystemencoding
from inspect import currentframe, getfile, getsourcefile
from collections import OrderedDict

import pandas as pd
from nilm_metadata import convert_yaml_to_hdf5

from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import check_directory_exists, get_datastore

#{"load_type": {"floor/wing":meter_number_in_nilmtk}
acad_block_meter_mapping = {'Building Total Mains': {'0': 1},
                            'Lifts': {'0': 2},
                            'Floor Total': {'1': 3, '2': 4, '3': 5, '4': 6, '5': 7},
                            'AHU': {'0': 8, '1': 9, '2': 10, '5': 11},
                            'Lights': {'3': 12},
                            'Power Sockets': {'3A': 13, '3B': 14},
                            'UPS Sockets': {'3': 15}}

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

    for building_name, building_mapping in overall_dataset_mapping.iteritems():
        for load_name, load_mapping in building_mapping.iteritems():
            for load_mapping_path, meter_number in load_mapping.iteritems():
                building_number = building_number_mapping[building_name]
                key = Key(building=building_number, meter=meter_number)
                dfs = []
                for attribute in column_mapping.keys():
                    filename_attribute = join(combed_path, building_name, load_name, load_mapping_path, "%s.csv" %attribute)
                    print(filename_attribute)
                    dfs.append(pd.read_csv(filename_attribute, parse_dates=True, index_col=0, header=True, names=[attribute]))
                total = pd.concat(dfs, axis=1)
                total = total.tz_localize('UTC').tz_convert('Asia/Kolkata')
                total.rename(columns=lambda x: column_mapping[x], inplace=True)
                total.columns.set_names(LEVEL_NAMES, inplace=True)
                assert total.index.is_unique
                store.put(str(key), total)
    convert_yaml_to_hdf5(join(_get_module_directory(), 'metadata'),
                         output_filename)

    print("Done converting COMBED to HDF5!")


def _get_module_directory():
    # Taken from http://stackoverflow.com/a/6098238/732596
    path_to_this_file = dirname(getfile(currentframe()))
    if not isdir(path_to_this_file):
        encoding = getfilesystemencoding()
        path_to_this_file = dirname(unicode(__file__, encoding))
    if not isdir(path_to_this_file):
        abspath(getsourcefile(lambda _: None))
    if not isdir(path_to_this_file):
        path_to_this_file = getcwd()
    assert isdir(path_to_this_file), path_to_this_file + ' is not a directory'
    return path_to_this_file
