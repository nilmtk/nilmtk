from os.path import join
import pandas as pd
import numpy as np
from nilmtk.utils import get_module_directory
from nilmtk import DataSet
from nilmtk.utils import get_datastore
from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilm_metadata import convert_yaml_to_hdf5
import tempfile, shutil
from sys import stderr
import yaml
"""
TODO
----
* convert HES appliance names to NILMTK standard
* what exactly is measured? Real power? Apparent?
* houses which have multiple mains: are they multiple 'splits' or phases or meters?
* dataset metadata
* some houses have both 2- and 10-minute data.  Might need a function to ignore 10 minute data.
* set up wiring to take into consideration the information in 
  'total_profiles.csv'  Sockets 1-11 are circuits monitored
  at the consumer unit which feed fall sockets around the dwelling.
* import the enormous amount of appliance metadata in 'appliance_data.csv', 
  especially channels which recorded multiple appliances
* use the metadata in 'ipsos.csv' and 'rdsap_data.csv' and 'rdsap_*.csv' for each Building
* Maybe email CAR to let them know that nilmtk can now import HES.
HES notes
---------
* As of 2018, the dataset is available from ukdataservice.ac.uk
* 14 homes recorded mains but only 5 were kept after cleaning
* circuit-level data from the consumer unit was recorded as 'sockets' for 216 houses
* 'total_profiles.csv' records pairs of <house>,<appliance> which are the 
  channels which need to be added to produce the whole-home total, which
  I think consists of all the circuit-level meters plus all appliances
  which are not also monitored at circuit level.
* appliance 2000 represents the calculated aggregate ???
* appliance 159 represents the difference between ???
  this and the sum of the known appliances
* appliance_codes.csv maps from <appliance code> to <appliance name>
* seasonal_adjustments.csv stores the trends in energy usage per appliance 
  activation over a year.
"""

FILENAMES = ['agd-{s}/appliance_group_data-{s}.csv'.format(s=s) for s in
             ['1a','1b','1c','1d','2','3']]
CHUNKSIZE = 1E6 # number of rows
COL_NAMES = ['interval id', 'house id', 'appliance code', 'date', 'data', 'time']
LAST_PWR_COLUMN = 250
NANOSECONDS_PER_TENTH_OF_AN_HOUR = 1E9 * 60 * 6
MAINS_CODES = [240, 241]
TEMPERATURE_CODES = list(range(251, 256))
CIRCUIT_CODES = list(range(208, 218)) + [222]
#E_MEASUREMENT = Measurement('energy', 'active')

def load_list_of_house_ids(data_dir):
    """Returns a list of house IDs in HES (ints)."""
    filename = join(data_dir, 'anonhes', 'ipsos-anonymised-corrected_310713.csv')
    series = pd.read_csv(filename, usecols=[0], index_col=False, squeeze=True)
    return series.tolist()


    """Load data from UK Government's Household Electricity Survey 
    (the cleaned version of the dataset released in summer 2013).
    """

    # TODO: re-use code from 
    # https://github.com/JackKelly/pda/blob/master/scripts/hes/load_hes.py

    """
    Broad approach:
    * load list of houses
    * create dataset.buildings dict with empty Buildings
    * the keys of `dataset.buildings` are the HES house IDs, which
      will be converted to the nilmtk standard after loading.
    * load CHUNK_SIZE of data from CSV into a DataFrame
    * convert datetime
    * get list of houses in the DF
    * for each house:
        * Load previously converted data from the HDFStore
        * append new data
        * save back to HDFStore
    * When all houses are complete, post-process:
        * sort all indicies
        * set timezone
        * convert energy to nilmtk standard energy unit (kWh?)
        * convert Wh to watts (retain energy) 
          (see 'convert_hes_to_watts.py' from pda)
        * convert keys of `dataset.buildings`
    """

def convert_hes(data_dir, output_filename, format='HDF', max_chunks=None):
    metadata = {
        'name': 'HES',
        'geographic_coordinates': (51.464462,-0.076544), # London
        'timezone': 'Europe/London'
    }
    
    # Open DataStore
    store = get_datastore(output_filename, format, mode='w')
    
    # load list of appliances
    hes_to_nilmtk_appliance_lookup = pd.read_csv(join(get_module_directory(), 
                                        'dataset_converters', 
                                        'hes', 
                                        'hes_to_nilmtk_appliance_lookup.csv'))

    # load list of houses
    hes_house_ids = load_list_of_house_ids(data_dir)
    nilmtk_house_ids = np.arange(1, len(hes_house_ids) + 1)
    hes_to_nilmtk_house_ids = dict(zip(hes_house_ids, nilmtk_house_ids))

    # array of hes_house_codes: nilmtk_building_code = house_codes.index(hes_house_code)
    house_codes = []
    
    # map 
    house_appliance_codes = dict()

    # Create a temporary metadata dir
    original_metadata_dir = join(get_module_directory(), 'dataset_converters', 'hes', 'metadata')
    tmp_dir = tempfile.mkdtemp()
    metadata_dir = join(tmp_dir, 'metadata')
    shutil.copytree(original_metadata_dir, metadata_dir)
    print("Using temporary dir for metadata:", metadata_dir)

    # Iterate over files
    for filename in FILENAMES:
        # Load appliance energy data chunk-by-chunk
        full_filename = join(data_dir, filename)
        print('Loading', full_filename)
        try:
            reader = pd.read_csv(full_filename, names=COL_NAMES, 
                                 index_col=False, chunksize=CHUNKSIZE)
        except IOError as e:
            print(e, file=stderr)
            continue

        # Iterate over chunks in file
        chunk_i = 0
        for chunk in reader:
            if max_chunks is not None and chunk_i >= max_chunks:
                break

            print(' processing chunk', chunk_i, 'of', filename)
            # Convert date and time columns to np.datetime64 objects
            dt = chunk['date'] + ' ' + chunk['time']
            del chunk['date']
            del chunk['time']
            chunk['datetime'] = pd.to_datetime(dt, format='%Y-%m-%d %H:%M:%S', utc=True)

            # Data is either tenths of a Wh or tenths of a degree
            chunk['data'] *= 10
            chunk['data'] = chunk['data'].astype(np.float32)

            # Iterate over houses in chunk
            for hes_house_id, hes_house_id_df in chunk.groupby('house id'):
                if hes_house_id not in house_codes:
                    house_codes.append(hes_house_id)
                    
                if hes_house_id not in house_appliance_codes.keys():
                    house_appliance_codes[hes_house_id] = []
                
                nilmtk_house_id = house_codes.index(hes_house_id)+1
                
                # Iterate over appliances in house
                for appliance_code, appliance_df in chunk.groupby('appliance code'):
                    if appliance_code not in house_appliance_codes[hes_house_id]:
                        house_appliance_codes[hes_house_id].append(appliance_code)
                    nilmtk_meter_id = house_appliance_codes[hes_house_id].index(appliance_code)+1
                    _process_meter_in_chunk(nilmtk_house_id, nilmtk_meter_id, hes_house_id_df, store, appliance_code)
                    
            chunk_i += 1
            
            
    print('houses with some data loaded:', house_appliance_codes.keys())
    
    store.close()
    
    # generate building yaml metadata
    for hes_house_id in house_codes:
        nilmtk_building_id = house_codes.index(hes_house_id)+1
        building_metadata = {}
        building_metadata['instance'] = nilmtk_building_id
        building_metadata['original_name'] = int(hes_house_id) # use python int
        building_metadata['elec_meters'] = {}
        building_metadata['appliances'] = []
        
        # initialise dict of instances of each appliance type
        instance_counter = {}
        
        for appliance_code in house_appliance_codes[hes_house_id]:
            nilmtk_meter_id = house_appliance_codes[hes_house_id].index(appliance_code)+1
            # meter metadata
            if appliance_code in MAINS_CODES:
                meter_metadata = {'device_model': 'multivoies',
                                  'site_meter': True}
                break
            elif appliance_code in CIRCUIT_CODES:
                meter_metadata = {'device_model': 'multivoies'}
                break
            elif appliance_code in TEMPERATURE_CODES:
                break
            else: # is appliance
                meter_metadata = {'device_model': 'wattmeter'}
                
            # only appliance meters at this point
            building_metadata['elec_meters'][nilmtk_meter_id] = meter_metadata
            # appliance metadata
            lookup_row = hes_to_nilmtk_appliance_lookup[hes_to_nilmtk_appliance_lookup.Code==appliance_code].iloc[0]
            appliance_metadata = {'original_name': lookup_row.Name, 
                                      'meters': [nilmtk_meter_id] }
            # appliance type
            appliance_metadata.update({'type': lookup_row.nilmtk_name})
            # TODO appliance room
            
            # appliance instance number
            if instance_counter.get(lookup_row.nilmtk_name) == None:
                instance_counter[lookup_row.nilmtk_name] = 0
            instance_counter[lookup_row.nilmtk_name] += 1 
            appliance_metadata['instance'] = instance_counter[lookup_row.nilmtk_name]
            
            building_metadata['appliances'].append(appliance_metadata)
            
            
        building = 'building{:d}'.format(nilmtk_building_id)
        
        yaml_full_filename = join(
            metadata_dir, building + '.yaml'
        )

        with open(yaml_full_filename, 'w') as outfile:
            #print(building_metadata)
            outfile.write(yaml.dump(building_metadata))
            
    # write yaml metadata to hdf5
    convert_yaml_to_hdf5(
        metadata_dir,
        output_filename
    )

    # remote the temporary dir when finished
    shutil.rmtree(tmp_dir)
    

def _process_meter_in_chunk(nilmtk_house_id, meter_id, chunk, store, appliance_code):
    data = chunk['data'].values
    index = chunk['datetime']
    df = pd.DataFrame(data=data, index=index)
    df.columns = pd.MultiIndex.from_tuples([('power', 'active')])
    
    # Modify the column labels to reflect the power measurements recorded.
    df.columns.set_names(LEVEL_NAMES, inplace=True)
    df = df.sort_index()

    key = Key(building=nilmtk_house_id, meter=meter_id)
    store.append(str(key), df)
