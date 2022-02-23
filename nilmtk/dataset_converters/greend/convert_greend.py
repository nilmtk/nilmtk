from os import listdir, getcwd
from os.path import join, isdir, isfile, dirname, abspath
import pandas as pd
import numpy as np
import datetime
import time
from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilm_metadata import convert_yaml_to_hdf5
import warnings
import numpy as np
from io import StringIO
from multiprocessing import Pool
from nilmtk.utils import get_module_directory

def _get_blocks(filename):
    '''
    Return a list of dataframes from a GREEND CSV file
    
    GREEND files can be interpreted as multiple CSV blocks concatenated into
    a single file per date. Since the columns of the individual blocks can 
    vary in a single file, they need to be read separately.
    
    There are some issues we need to handle in the converter:
    - the headers from the multiple blocks
    - corrupted data (lines with null chars, broken lines)
    - more fields than specified in header
    '''
    block_data = None
    dfs = []
    previous_header = None
    print(filename)
    # Use float64 for timestamps and float32 for the rest of the columns
    dtypes = {}
    dtypes['timestamp'] = np.float64
    
    def _process_block():
        if block_data is None:
            return
            
        block_data.seek(0)
        try:
            # ignore extra fields for some files
            error_bad_lines = not (
                ('building5' in filename and 'dataset_2014-02-04.csv' in filename)
            )
            df = pd.read_csv(block_data, index_col='timestamp', dtype=dtypes, error_bad_lines=error_bad_lines)
        except: #(pd.errors.ParserError, ValueError, TypeError):
            print("ERROR", filename)
            raise
            
        df.index = pd.to_datetime(df.index, unit='s')
        df = df.tz_localize("UTC").tz_convert("CET").sort_index()
        dfs.append(df)
        block_data.close()
    
    special_check = (
        ('dataset_2014-01-28.csv' in filename and 'building5' in filename) or
        ('dataset_2014-09-02.csv' in filename and 'building6' in filename)
    )
    
    with open(filename, 'r') as f:
        for line in f:
            # At least one file have a bunch of nulls present, let's clean the data
            line = line.strip('\0')
            if 'time' in line:
                # Found a new block
                if not line.startswith('time'):
                    # Some lines are corrupted, e.g. 1415605814.541311,0.0,NULL,NUtimestamp,000D6F00029C2918...
                    line = line[line.find('time'):]
                
                if previous_header == line.strip():
                    # Same exact header, we can treat it as the same block
                    # print('Skipping split')
                    continue
                    
                # Using a defaultdict for the dtypes didn't work with read_csv,
                # so we fill a normal dict when we find the columns
                cols = line.strip().split(',')[1:]
                for col in cols:
                    dtypes[col] = np.float32
                    
                # print('Found new block')
                _process_block()
                block_data = StringIO()
                previous_header = line.strip()

            
            if special_check:
                if ('0.072.172091508705606' in line or
                    '1409660828.0753369,NULL,NUL' == line):
                    continue

            block_data.write(line)
            
    # Process the remaining block
    _process_block()
    
    return (filename, dfs)

    
def _get_houses(greend_path):
    house_list = listdir(greend_path)
    return [h for h in house_list if isdir(join(greend_path,h))] 
    

def convert_greend(greend_path, hdf_filename, use_mp=True):
    """
    Parameters
    ----------
    greend_path : str
        The root path of the greend dataset.
    hdf_filename : str
        The destination HDF5 filename (including path and suffix).
    use_mp : bool 
        Defaults to True. Use multiprocessing to load the files for
        each building.
    
    The raw dataset can be downloaded from:
        https://docs.google.com/forms/d/e/1FAIpQLSf3Tbr7IDoSORNFw7dAGD2PB6kSO98RRiVpmKOWOZ52ULAMzA/viewform
    """
    store = pd.HDFStore(hdf_filename, 'w', complevel=5, complib='zlib')
    houses = sorted(_get_houses(greend_path))
    
    print('Houses found:', houses)
    if use_mp:
        pool = Pool()
    
    h = 1 # nilmtk counts buildings from 1 not from 0 as we do, so everything is shifted by 1
    
    for house in houses:
        print('Loading', house)
        abs_house = join(greend_path, house)
        dates = [d for d in listdir(abs_house) if d.startswith('dataset')]
        target_filenames = [join(abs_house, date) for date in dates]
        if use_mp:
            house_data = pool.map(_get_blocks, target_filenames)

            # Ensure the blocks are sorted by date and make a plain list
            house_data_dfs = []
            for date, data in sorted(house_data, key=lambda x: x[0]):
                house_data_dfs.extend(data)
        else:
            house_data_dfs = []
            for fn in target_filenames:
                house_data_dfs.extend(_get_blocks(fn)[1])
            
        overall_df = pd.concat(house_data_dfs, sort=False).sort_index()
        dups_in_index = overall_df.index.duplicated(keep='first')
        if dups_in_index.any():
            print("Found duplicated values in index, dropping them.")
            overall_df = overall_df[~dups_in_index]
        
        m = 1
        for column in overall_df.columns:
            print("meter {}: {}".format(m, column))
            key = Key(building=h, meter=m)
            print("Putting into store...")
            
            df = overall_df[column].to_frame() #.dropna(axis=0)
            
            # if drop_duplicates:
                # print("Dropping duplicated values in data...")
                # df = df.drop_duplicates()
            
            df.columns = pd.MultiIndex.from_tuples([('power', 'active')])
            df.columns.set_names(LEVEL_NAMES, inplace=True)
            
            store.put(str(key), df, format = 'table')
            m += 1
            # print('Flushing store...')
            # store.flush()
            
        h += 1

	# retrieve the dataset metadata in the metadata subfolder
    metadata_dir = join(get_module_directory(), 'dataset_converters', 'greend', 'metadata')
    convert_yaml_to_hdf5(metadata_dir, hdf_filename)
    
    # close h5
    store.close()

#is only called when this file is the main file... only test purpose
if __name__ == '__main__':
    t1 = time.time()
    convert_greend('GREEND_0-2_300615',
                   'GREEND_0-2_300615.h5')
    dt = time.time() - t1
    print()
    print()
    print('Time passed: {}:{}'.format(int(dt/60), int(dt%60)))
    