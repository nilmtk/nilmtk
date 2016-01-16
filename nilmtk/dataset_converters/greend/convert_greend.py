from __future__ import print_function, division
from os import listdir, getcwd
from os.path import join, isdir, isfile, dirname, abspath
import pandas as pd
import datetime
import time
from nilmtk.datastore import Key
import warnings
from nilm_metadata import convert_yaml_to_hdf5
import csv

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
        print('loading '+house)
        abs_house = join(greend_path, house)
        dates = [d for d in listdir(abs_house) if d.startswith('dataset')]
        house_data = []
        for date in dates:
            print('-----------------------',date)
            try:
                tmp_pandas = pd.read_csv(join(abs_house, date), na_values=['na'], error_bad_lines=False)
            except: # A CParserError is returned for malformed files (irregular column number)
                pass

            if "timestamp" in tmp_pandas.columns:
                   pass
            else:
                tmp_pandas["timestamp"] = tmp_pandas.index
            tmp_pandas.index = tmp_pandas["timestamp"].convert_objects(convert_numeric=True).values
            tmp_pandas = tmp_pandas.drop("timestamp",1)

            tmp_pandas = tmp_pandas.astype("float32")


            tmp_pandas.index = pd.to_datetime(tmp_pandas.index, unit='s')
            tmp_pandas = tmp_pandas.tz_localize("UTC").tz_convert("CET")
            tmp_pandas = tmp_pandas.drop_duplicates()
            #tmp_pandas = tmp_pandas.sort_index()
            house_data.append(tmp_pandas)
        overall_df = pd.concat(house_data)
        overall_df = overall_df.drop_duplicates()
        overall_df = overall_df.sort_index()

        m = 1

        for column in overall_df.columns:
            print("meter" + str(m)+': '+column)
            key = Key(building = h, meter=m)
            print("Putting into store...")
            store.put(str(key), overall_df[column], format = 'table')
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
    
def __preprocess_file(building_path, day_file):
    filename = join(building_path, day_file)
    csvfile = open(filename, 'rb')
    ff = csv.reader(csvfile, delimiter = ',', quotechar='|')
    from collections import defaultdict
    cols_nums = defaultdict(list)
    for f in ff: cols_nums[len(f)].append(f) # group by column number
    best_col_num = sorted( [(k, len(cols_nums[k])) for k in cols_nums.keys()] , key=lambda x:x[1], reverse=True) # sort rows by row_number DESC
    processed_rows = cols_nums[best_col_num[0][0]] # reject outliers (all rows with different column number)
    print("\t"+day_file+" has", best_col_num, "taking only rows with", best_col_num[0][0], "columns")    
    
    import io
    csvfile = io.BytesIO()
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerows(processed_rows) # print row to csv byte stream
    return csvfile.getvalue()

#is only called when this file is the main file... only test purpose
if __name__ == '__main__':
    t1 = time.time()
    convert_greend('/home/student/Downloads/GREEND_0-1_311014/', 
                   '/home/student/Desktop/greend.h5')
    dt = time.time()- t1
    print('\n\nTime passed:\n'+str(int(dt/60))+' : ' + str(dt%60))
