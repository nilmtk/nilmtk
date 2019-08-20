import pandas as pd
import numpy as np
import sys
from os import listdir, getcwd
from os.path import isdir, join, dirname, abspath
from pandas import concat
from nilmtk.utils import get_module_directory, check_directory_exists
from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilm_metadata import convert_yaml_to_hdf5

"""
DATASET STRUCTURE:
------------------
On extracting all the dataset values, we should arrive at a similar directory structure as
mentioned.

ECO Dataset will have a folder '<i>_sm_csv' and '<i>_plug_csv' where i is the building no.
Originally, the expected folder structure was:

- <i>_sm_csv has a folder <i>
- <i>_plug_csv has folders 01, 02,....<n> where n is the plug numbers.

This version also supports the following structure, which can be created by unpacking the 
ZIP files uniformly, creating a folder for each one:

- <i>_sm_csv has a folder <i>
- <i>_plug_csv has a folder <i>, and <i>_plug_csv/<i> has folders 01, 02,....<n>,
  where n is the plug numbers.


Each folder has a CSV file as per each day, with each day csv file containing
	86400 entries.
"""

plugs_column_name = {1: ('power', 'active')}

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
    store = pd.HDFStore(hdf_filename, 'w', complevel=9, complib='blosc')    
    
    check_directory_exists(dataset_loc)
    directory_list = [i for i in listdir(dataset_loc) if '.txt' not in i]
    directory_list.sort()
    print(directory_list)

    found_any_sm = False
    found_any_plug = False
    
    # Traversing every folder
    for folder in directory_list:
        if folder[0] == '.' or folder[-3:] == '.h5':
            print('Skipping ', folder)
            continue

        #Building number and meter_flag
        building_no = int(folder[:2])
        meter_flag = None 
        if 'sm_csv' in folder:
            meter_flag = 'sm'
        elif 'plugs' in folder:
            meter_flag = 'plugs'
        else:
            print('Skipping folder', folder)
            continue
            
        print('Computing for folder', folder)

        dir_list = [i for i in listdir(join(dataset_loc, folder)) if isdir(join(dataset_loc,folder,i))]
        dir_list.sort()
        
        if meter_flag == 'plugs' and len(dir_list) < 3:
            # Try harder to find the subfolders
            folder = join(folder, folder[:2])
            dir_list = [i for i in listdir(join(dataset_loc, folder)) if isdir(join(dataset_loc,folder,i))]
        
        print('Current dir list:', dir_list)

        for fl in dir_list:
            print('Computing for folder ', fl)
            
            fl_dir_list = [i for i in listdir(join(dataset_loc,folder,fl)) if '.csv' in i]
            fl_dir_list.sort()

            if meter_flag == 'sm':
                for fi in fl_dir_list:
                    found_any_sm = True
                    df = pd.read_csv(join(dataset_loc,folder,fl,fi), names=[i for i in range(1,17)], dtype=np.float32)
                    
                    for phase in range(1,4):
                        key = str(Key(building=building_no, meter=phase))
                        df_phase = df.loc[:,[1+phase, 5+phase, 8+phase, 13+phase]]

                        # get reactive power
                        power = df_phase.loc[:, (1+phase, 13+phase)].values
                        reactive = power[:,0] * np.tan(power[:,1] * np.pi / 180)
                        df_phase['Q'] = reactive
                        
                        df_phase.index = pd.date_range(start=fi[:-4], freq='s', periods=86400, tz='GMT')
                        df_phase = df_phase.tz_convert(timezone)
                        
                        sm_column_name = {
                            1+phase:('power', 'active'),
                            5+phase:('current', ''),
                            8+phase:('voltage', ''),
                            13+phase:('phase_angle', ''),
                            'Q': ('power', 'reactive'),
                        }
                        df_phase.columns = pd.MultiIndex.from_tuples([
                            sm_column_name[col] for col in df_phase.columns
                        ])
                        
                        power_active = df_phase['power', 'active']
                        tmp_before = np.size(power_active)
                        df_phase = df_phase[power_active != -1]
                        power_active = df_phase['power', 'active']
                        tmp_after = np.size(power_active)
                        
                        if tmp_before != tmp_after:
                            print('Removed missing measurements - Size before: ' + str(tmp_before) + ', size after: ' + str(tmp_after))
                        
                        df_phase.columns.set_names(LEVEL_NAMES, inplace=True)
                        if not key in store:
                            store.put(key, df_phase, format='Table')
                        else:
                            store.append(key, df_phase, format='Table')
                            store.flush()
                        print('Building', building_no, ', Meter no.', phase,
                              '=> Done for ', fi[:-4])
                
            else:
                #Meter number to be used in key
                meter_num = int(fl) + 3
                
                key = str(Key(building=building_no, meter=meter_num))

                current_folder = join(dataset_loc,folder,fl)
                if not fl_dir_list:
                    raise RuntimeError("No CSV file found in " + current_folder)
                    
                #Getting dataframe for each csv file seperately
                for fi in fl_dir_list:
                    found_any_plug = True
                    df = pd.read_csv(join(current_folder, fi), names=[1], dtype=np.float64)
                    df.index = pd.date_range(start=fi[:-4].replace('.', ':'), freq='s', periods=86400, tz='GMT')
                    df.columns = pd.MultiIndex.from_tuples(plugs_column_name.values())
                    df = df.tz_convert(timezone)
                    df.columns.set_names(LEVEL_NAMES, inplace=True)

                    tmp_before = np.size(df.power.active)
                    df = df[df.power.active != -1]
                    tmp_after = np.size(df.power.active)
                    if (tmp_before != tmp_after):
                        print('Removed missing measurements - Size before: ' + str(tmp_before) + ', size after: ' + str(tmp_after))
                    
                    # If table not present in hdf5, create or else append to existing data
                    if not key in store:
                        store.put(key, df, format='Table')
                        print('Building',building_no,', Meter no.',meter_num,'=> Done for ',fi[:-4])
                    else:
                        store.append(key, df, format='Table')
                        store.flush()
                        print('Building',building_no,', Meter no.',meter_num,'=> Done for ',fi[:-4])
            
            
    if not found_any_plug or not found_any_sm:
        raise RuntimeError('The files were not found! Please check the folder structure. Extract each ZIP file into a folder with its base name (e.g. extract "01_plugs_csv.zip" into a folder named "01_plugs_csv", etc.)')
        
    print("Data storage completed.")
    store.close()

    # Adding the metadata to the HDF5file
    print("Proceeding to Metadata conversion...")
    meta_path = join(
        get_module_directory(), 
        'dataset_converters',
        'eco',
        'metadata'
    )
    convert_yaml_to_hdf5(meta_path, hdf_filename)
    print("Completed Metadata conversion.")


