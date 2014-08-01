from __future__ import print_function, division
from os import listdir, getcwd
from os.path import join, isdir, isfile, dirname, abspath
import pandas as pd
import datetime
import time
from nilmtk.datastore import Key
import warnings
from nilm_metadata import convert_yaml_to_hdf5

warnings.filterwarnings("ignore")


def convert_greend(greend_path, hdf_filename):
    """
    Parameters
    ----------
    greend_path : str
        The root path of the greend dataset.
    hdf_filename : str
        The destination HDF5 filename (including path and suffix).
    """


    store = pd.HDFStore(hdf_filename, 'w', complevel=9, complib='zlib')
    houses = sorted(__get_houses(greend_path))
    print(houses)
    h = 1
    for house in houses:
        print('loading '+house+"'s house...")
        abs_house = join(greend_path, house)
        dates = [d for d in listdir(abs_house) if d.startswith('dataset')]
        house_data = pd.DataFrame()
        for date in dates:
            print('-----------------------',date)
            tmp_pandas = pd.DataFrame.from_csv(join(abs_house, date))
            tmp_pandas = tmp_pandas[tmp_pandas.index != 'timestamp']
            tmp_pandas = tmp_pandas.sort_index()
            c = 0 
            tmp_pandas.index = [__timestamp(t) for t in tmp_pandas.index]
            house_data = house_data.append(tmp_pandas)

            #for testing metadata files:
            #break
        m = 1 


        for meter in house_data:
            print("meter" + str(m)+': ')
            key = Key(building = h, meter=m)
            print("Putting into store...")
            store.put(str(key), house_data[meter], format = 'table')
            m += 1
            print('Flushing store...')
            store.flush()
        h += 1

    store.close()

    #needs to be edited
    convert_yaml_to_hdf5('/path/to/metadata', hdf_filename)
    

def __timestamp(t):
    res = 1
    try:
        res = datetime.datetime.fromtimestamp(int(float(t)))
    except ValueError:
        print('exception'+str(t))
    return res

def __get_houses(greend_path):
    house_list = listdir(greend_path)
    return [h for h in house_list if isdir(join(greend_path,h))] 
    

#is only called when this file is the main file... only test purpose
if __name__ == '__main__':
    t1 = time.time()
    convert_greend('/home/student/Desktop/greend', 
                   '/home/student/Desktop/greend.h5')
    dt = time.time()- t1
    print('\n\nTime passed:\n'+str(int(dt/60))+' : ' + str(dt%60))
