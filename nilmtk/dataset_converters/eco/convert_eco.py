import pandas as pd
import numpy as np
import sys
from os import listdir, getcwd
from os.path import isdir, join, dirname, abspath
from pandas.tools.merge import concat
from nilmtk.utils import get_module_directory, check_directory_exists
from nilmtk.datastore import Key
from nilm_metadata import convert_yaml_to_hdf5
from inspect import currentframe, getfile, getsourcefile
from sys import getfilesystemencoding


"""
DATASET STRUCTURE:
------------------
On extracting all the dataset values, we should arrive at a similar directory structure as
mentioned.

ECO Dataset will have a folder '<i>_sm_csv' and '<i>_plug_csv' where i is the building no.

<i>_sm_csv has a folder 01
<i>_plug_csv has a folder 01, 02,....<n> where n is the plug numbers.

Each folder has a CSV file as per each day, with each day csv file containing
	86400 entries.
"""

sm_column_name = {1:('power', 'active'),
					2:('power', 'active'),
					3:('power', 'active'),
					4:('power', 'active'),
					5:('current', ''),
					6:('current', ''),
					7:('current', ''),
					8:('current', ''),
					9:('voltage', ''),
					10:('voltage', ''),
					11:('voltage', ''),
					12:('phase_angle', ''),
					13:('phase_angle', ''),
					14:('phase_angle', ''),
					15:('phase_angle', ''),
					16:('phase_angle', ''),
					};

plugs_column_name = {1:('power', 'active'),
                    };

def convert_eco(dataset_loc, hdf_filename, timezone):
    """
    Parameters:
    -----------
    dataset_loc: str
        The root directory where the dataset is located.
    hdf_filename: str
        The location where the hdf_filename is present. 
        The directory location has to contain the 
        hdf5file name for the converter to work.
    timezone: str
        specifies the timezone of the dataset.
    """

    # Creating a new HDF File
    store = pd.HDFStore(hdf_filename, 'w')
    
    check_directory_exists(dataset_loc)
    directory_list = [i for i in listdir(dataset_loc) if '.txt' not in i]
    directory_list.sort()
    print directory_list

    # Traversing every folder
    for folder in directory_list:

        if folder[0] == '.' or folder[-3:] == '.h5':
            print 'Skipping ', folder
            continue
        print 'Computing for folder',folder

        #Building number and meter_flag
        building_no = int(folder[:2])
        meter_flag = 'sm' if 'sm_csv' in folder else 'plugs'

        dir_list = [i for i in listdir(join(dataset_loc, folder)) if isdir(join(dataset_loc,folder,i))]
        dir_list.sort()
        print 'Current dir list:',dir_list

        for fl in dir_list:
            #Meter number to be used in key
            meter_num = 1 if meter_flag == 'sm' else int(fl) + 1

            print 'Computing for Meter no.',meter_num

            # if meter_num != 4:
            #     continue

            fl_dir_list = [i for i in listdir(join(dataset_loc,folder,fl)) if '.csv' in i]
            fl_dir_list.sort()

            # import ipdb
            # ipdb.set_trace()

            key = str(Key(building=building_no, meter=meter_num))

            for fi in fl_dir_list:

                #Getting dataframe for each csv file seperately
                df_fl = _get_df(join(dataset_loc,folder,fl),fi,meter_flag)
                # The ECO data set is already sorted by time
                # df_fl.sort_index(ascending=True,inplace=True)
                df_fl = df_fl.tz_convert(timezone)

                # If table not present in hdf5, create or else append to existing data
                if not key in store:
                    store.put(key, df_fl, format='Table')
                    print 'Building',building_no,', Meter no.',meter_num,'=> Done for ',fi[:-4]
                else:
                    store.append(key, df_fl, format='Table')
                    store.flush()
                    print 'Building',building_no,', Meter no.',meter_num,'=> Done for ',fi[:-4]

    print "Data storage completed."
    store.close()

    # Adding the metadata to the HDF5file
    print "Proceeding to Metadata conversion..."
    meta_path = join(_get_module_directory(), 'metadata')
    convert_yaml_to_hdf5(meta_path, hdf_filename)
    print "Completed Metadata conversion."

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

def _get_df(dir_loc, csv_name, meter_flag):
    """
    Parameters
    ----------
    dir_loc: str
        Location of the directory containing the csv file.
    csv_name: str
        Name of the .csv file whose values are being read.
    meter_flag: str
        Used to differentiate between a Smart Meter and a Plug file.
    """

    # Changing column length for Smart Meters and Plugs
    column_num = 16 if meter_flag == 'sm' else 1

    # Reading the CSV file and adding a datetime64 index to the Dataframe
    df = pd.read_csv(join(dir_loc,csv_name), names=[i for i in range(1,column_num+1)], dtype=np.float64)
    df.index = pd.DatetimeIndex(start=csv_name[:-4], freq='s', periods=86400, tz = 'GMT')

    if meter_flag == 'sm':
        df.rename(columns=sm_column_name, inplace=True)
    else:
        df.rename(columns=plugs_column_name, inplace=True)
    return df
