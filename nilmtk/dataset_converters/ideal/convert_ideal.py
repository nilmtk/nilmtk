import pandas as pd
import numpy as np
from copy import deepcopy
from os.path import join, isdir, isfile
from os import listdir
import re
import os
from sys import stdout
from pathlib import Path
from nilmtk.utils import get_datastore
from nilmtk.datastore import Key
from nilmtk.timeframe import TimeFrame
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import get_module_directory, check_directory_exists
from nilm_metadata import convert_yaml_to_hdf5, save_yaml_to_datastore

"""
TODO:
* The bottleneck appears to be CPU.  So could be sped up by using 
  multiprocessing module to use multiple CPU cores to load ideal channels in 
  parallel.
"""


def convert_ideal(ideal_path, output_filename, format='HDF'):
    """
    Convert the IDEAL dataset to NILMTK HDF5 format.
    From https://datashare.ed.ac.uk/handle/10283/3647 download these zips below:
        - household_sensors.zip (14.77Gb).
        - room_and_appliance_sensors.zip (9.317Gb).
    Both zips contain a folder called "sensorsdata".
    Create a new folder, e.g. called "ideal_dataset", and into it
        - Extract the folder "household_sensors.zip/sensordata" with the name 
          household_sensordata
        - Extract the folder "room_and_appliance_sensors/sensordata" with the 
          name rooms_appliance_sensensensordata

    Then run the function convert_ideal with ideal_path="ideal_dataset".

    Parameters
    ----------
    ideal_path : str
        The root path of the ideal low_freq dataset.
    output_filename : str
        The destination filename (including path and suffix).
    format : str
        format of output. Either 'HDF' or 'CSV'. Defaults to 'HDF'
    """
    

    def _ideal_measurement_mapping_func(house_id, chan_id,category_id):
        if(category_id=="electric-appliance"):
            ac_type='active'
            return [('power', ac_type)]
        else:
            ac_type='apparent'
            return [('power', ac_type)]
        
    # Open DataStore
    store = get_datastore(output_filename, format, mode='w')

    #household_sensordata contains mains reading
    #rooms_appliance_sensordata contains appliance reading
    folders=[]
    for root, dirs, files in os.walk(ideal_path):
        for folder in dirs:
            if(folder=="household_sensordata" or folder=="rooms_appliance_sensordata"):
                folders.append(folder)
    #valid_home_id are home ids which contain both mains and appliance reading            
    valid_home_id=mains_plus_appliance_home_id(ideal_path,folders)
    for folder in folders:
        input_path=join(ideal_path,folder)
        # Convert raw data to DataStore
        _convert(input_path, store, _ideal_measurement_mapping_func, 'Europe/London',valid_home_id)

    metadata_path = join(get_module_directory(),
                              'dataset_converters',
                              'ideal',
                              'metadata')


    # Add metadata
    save_yaml_to_datastore(metadata_path, store)
    store.close()

    print("Done converting ideal to HDF5!")

def _convert(input_path, store, measurement_mapping_func, tz,valid_home_id, sort_index=True, drop_duplicates=False):
    """
    Parameters
    ----------
    input_path : str
        The root path of the ideal low_freq dataset.
    store : DataStore
        The NILMTK DataStore object.
    measurement_mapping_func : function
        Must take these parameters:
            - house_id
            - chan_id
        Function should return a list of tuples e.g. [('power', 'active')]
    tz : str 
        Timezone e.g. 'US/Eastern'
    sort_index : bool
        Defaults to True
    drop_duplicates : bool
        Remove entries with duplicated timestamp (keeps the first value)
        Defaults to False for backwards compatibility.
    """

    check_directory_exists(input_path)
    #each file containg mains/appliance is linked with a house and sensor id
    filename,houses,sensor,category=_find_all_houses_sensor(input_path,valid_home_id)
    assert(len(houses)==len(sensor))
    for id in range(len(houses)):
        if(category[id]=='electric-appliance'):
            stdout.flush()
            key=Key(building=houses[id],meter=int(sensor[id]))
            csv_filename= join(input_path, filename[id])
            measurements = measurement_mapping_func(houses[id], sensor[id],category[id])
            df = _load_csv(csv_filename, measurements, tz, 
                sort_index=sort_index, 
                drop_duplicates=drop_duplicates
            )
            store.put(str(key), df)

        elif(category[id]=='electric-mains'):
            combined_meters=sensor[id].split('c')
            stdout.flush()
            key=Key(building=houses[id],meter=int(combined_meters[0]))
            csv_filename= join(input_path, filename[id])
            measurements = measurement_mapping_func(houses[id], sensor[id],category[id])
            df = _load_csv(csv_filename, measurements, tz, 
                sort_index=sort_index, 
                drop_duplicates=drop_duplicates
            )
            store.put(str(key), df)
        print("Instance number:"+str(id))    
        print("Loading for home id:"+ str(houses[id])+"and sensor id:" + sensor[id]+"........")


def mains_plus_appliance_home_id(ideal_path,folders):
    folder1=folders[1]
    input_path1=join(ideal_path,folder1)
    file_identifier='home*.csv.gz'
    input_path1=Path(input_path1)

    home_id_appliance = list()
    for file in input_path1.glob(file_identifier):
        home_, room_, sensor_, category_, subtype_ = file.name.split('_')
        if(str(category_)=="electric-mains" or str(category_)=='electric-appliance' ):
            home_id_appliance.append(int(re.sub('\D', '', home_)))

    folder2=folders[0]
    input_path2=join(ideal_path,folder2)
    input_path2=Path(input_path2)
    home_id_mains = list()
    for file in input_path2.glob(file_identifier):
        home_, room_, sensor_, category_, subtype_ = file.name.split('_')
        if(str(category_)=="electric-mains" or str(category_)=='electric-appliance' ):
            home_id_mains.append(int(re.sub('\D', '', home_)))

    return [homeid for homeid in home_id_mains if homeid in home_id_appliance]    

def _find_all_houses_sensor(input_path,valid_home_id):
    """
    Returns
    -------
    list of integers (house instances)
    """
    file_identifier='home*.csv.gz'
    input_path=Path(input_path)
    homeid = list()
    roomid = list()
    roomtype = list()
    sensorid = list()
    category = list()
    subtype = list()
    filename = list()
    for file in input_path.glob(file_identifier):
        home_, room_, sensor_, category_, subtype_ = file.name.split('_')
        if(str(category_)=="electric-mains" or str(category_)=='electric-appliance' ):
            if(int(re.sub('\D', '', home_)) in valid_home_id):
                filename.append(str(file.name))
                homeid.append(int(re.sub('\D', '', home_)))
                roomid.append(int(re.sub('\D', '', room_)))
                roomtype.append(str(re.sub('\d', '', room_)))
                category.append(str(category_))
                subtype.append(str(subtype_[:-7]))

                assert sensor_[:6] == 'sensor'
                sensorid.append(str(sensor_[6:]))

    return filename,homeid,sensorid,category




def _load_csv(filename, columns, tz, drop_duplicates=True, sort_index=False):
    """
    Parameters
    ----------
    filename : str
    columns : list of tuples (for hierarchical column index)
    tz : str 
        e.g. 'US/Eastern'
    sort_index : bool
        Defaults to True
    drop_duplicates : bool
        Remove entries with duplicated timestamp (keeps the first value)
        Defaults to False for backwards compatibility.

    Returns
    -------
    pandas.DataFrame
    """
    # Load data
    df = pd.read_csv(filename, header=None, names=columns)
    df.columns.set_names(LEVEL_NAMES, inplace=True)

    df.index = pd.DatetimeIndex(df.index.values)
    df.index = df.index.values.astype(np.int64) // 10 ** 9
    df.index = pd.to_datetime(df.index.values, unit='s', utc=True)
    df = df.tz_convert(tz)
  

    if sort_index:
        df = df.sort_index() # raw ideal data isn't always sorted
        
    if drop_duplicates:
        dups_in_index = df.index.duplicated(keep='first')
        if dups_in_index.any():
            df = df[~dups_in_index]

    return df
