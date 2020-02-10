import requests
from os.path import join
import pandas as pd
import os
import time

def download():
    START_TIME_STR = "01-06-2014 0:0:0"
    END_TIME_STR = "01-07-2014 0:0:0"


    pattern = '%d-%m-%Y %H:%M:%S'
    START_TIME = int(time.mktime(time.strptime(START_TIME_STR, pattern)))*1000
    END_TIME = int(time.mktime(time.strptime(END_TIME_STR, pattern)))*1000

    SMAP_URL = raw_input("Enter SMAP URL")
    UUID_URL = join(SMAP_URL, "backend/api/query")

    MEASUREMENTS = ["Power", "Energy", "Current"]
    BASE_PATH = "/Users/nipunbatra/Desktop/iiitd/"

    academic_block = {'AHU': [0, 1, 2, 5],
                      'Building Total Mains': [0],
                      'Floor Total': [1, 2, 3, 4, 5],
                      'Lifts': [0],
                      'Light': [3],
                      'Power': [3],
                      'UPS': [3]
    }

    lecture_block = {'AHU-1': [0],
                     'AHU-2': [1],
                     'AHU-3': [2],
                     'Building Total Mains': [0],
                     'Floor Total': [0, 1, 2]
                     }

    load_renaming = {'AHU-1': 'AHU',
                     'AHU-2': 'AHU',
                     'AHU-3': 'AHU',
                     'Power': 'Power Sockets',
                     'UPS': 'UPS Sockets'}

    academic_building = {'Academic Block': academic_block, 'Lecture Block': lecture_block}

    query = """select *  where Metadata/Extra/Block = '{}' and (Metadata/SourceName = '{}') and Metadata/Extra/Type = '{}' and Metadata/Location/Floor = '{}' and Metadata/Extra/PhysicalParameter = '{}'"""

    for block_name, block in academic_building.items():
        for load, floors in block.items():
            for floor in floors:
                for measurement in MEASUREMENTS:
                    query_instance = query.format(block_name, "Academic Building", load, str(floor), measurement)
                    DATA_URL = join(SMAP_URL, "backend/api/data/uuid/{}?starttime={}&endtime={}")
                    uuid = requests.post(UUID_URL, query_instance).json()[0]['uuid']
                    data = requests.get(DATA_URL.format(uuid, START_TIME, END_TIME)).json()[0]["Readings"]
                    df = pd.DataFrame(data)
                    # Some loads like AHU-1, etc. need to be changed to only AHU
                    if load in load_renaming:
                        load_renamed = load_renaming[load]
                    else:
                        load_renamed = load
                    path_to_create = join(BASE_PATH, block_name, load_renamed, str(floor))
                    print(path_to_create)
                    if not os.path.exists(path_to_create):
                        os.makedirs(path_to_create)
                    df.to_csv(path_to_create+"/"+measurement+".csv", header=False, index=False)
