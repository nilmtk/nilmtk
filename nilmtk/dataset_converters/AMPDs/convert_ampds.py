from __future__ import print_function, division
import pandas as pd
import numpy as np
from pandas import *
from copy import deepcopy
from os.path import join, isdir, isfile
from os import listdir
import re
from sys import stdout
from nilmtk.datastore import Key
from nilmtk.timeframe import TimeFrame
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import get_module_directory
from nilm_metadata import convert_yaml_to_hdf5
from nilmtk.dataset import DataSet
from nilmtk.building import Building

# Column name mapping

columnNameMapping={ 'V':('voltage', ''),
                    'I':('current', ''),
                    'f':('frequency', ''),
                    'DPF': ('pf', 'd'),
                    'APF': ('pf', 'apparent'),
                    'P': ('power', 'active'),
                    'Pt':('energy', 'active'),
                    'Q':('power', 'reactive'),
                    'Qt':('energy', 'reactive'),
                    'S':('power','apparent'),
                    'St':('energy', 'apparent') }

# Appliance name mapping. It is not used currently, but I put it here just in case I need it later on

'''applianceNameMapping={ 'B1E': ('bedroom misc', 1), 'B2E': ('bedroom misc', 2),    'BME':('plugs', 1),    'CDE': ('dryer washer', 1),    'CWE': ('dryer washer', 2),    'DNE': ('plugs', 2), 'DWE': ('dishwasher', 1),    'EBE': ('workbench', 1),    'EQE':('security', 1),    'FGE': ('fridge', 1), 'FRE':('space heater', 1),    'GRE': ('misc', 3),    'HPE': ('air conditioner', 1),    'HTE': ('water heater', 1),'OFE': ('misc', 4),    'OUE': ('plugs', 3),    'TVE': ('entertainment', 1),    'UTE': ('plugs', 4),'WOE': ('oven', 1), 'UNE': ('unmetered', 1)}
'''

'''def readDataset(csvPath):
	files=[f for f in listdir(inputPath) if isfile (join(inputPath, f)) and '.csv' in f]
	return files	
'''
def convert(inputPath, hdfFilename, metadataPath='/'):

	'''
	Parameters: 
	-----------
	inputPath: str
		The path of the directory where all the csv files are supposed to be stored
	hdfFilename: str
		The path of the h5 file where all the standardized data is supposed to go. The path should refer to a particular file and not just a random directory in order for this to work.
	metadataPath: str
		The path of the directory where the metadata is stored. By default, it is the root directory.	
	
	'''


# This function contains the bulk of the code. The test() function can simply be ignored for now
# To do: Complete the metadata set. Then the convert_yaml_to_hdf5() function will stop throwing random errors.
	files=[f for f in listdir(inputPath) if isfile (join(inputPath, f)) and '.csv' in f and '.swp' not in f]
	print (files)
	assert isdir(inputPath)
#	print(files)
	store=HDFStore(hdfFilename)
#	fp=pd.read_csv(join(inputPath, sent))
	for i, csv_file in enumerate(files):  #range(len(files)):
		#sent=files[i]
		key=Key(building=1, meter=(i+1))
		print('Loading file #', (i+1), '. Please wait...')
		fp=pd.read_csv(join(inputPath, csv_file))
		fp.TS=fp.TS.astype('int')
		fp.index=pd.to_datetime((fp.TS.values*1e9).astype(int))
        	fp=fp.drop('TS', 1)
		fp.rename(columns=lambda x: columnNameMapping[x], inplace=True)
		fp.columns.set_names(LEVEL_NAMES, inplace=True)
		fp=fp.convert_objects(convert_numeric=True)
		fp=fp.dropna()
		fp=fp.astype(np.float32)
		store.put(str(key), fp, format='Table')
		store.flush()
		print("Done with file #", (i+1))
	store.close()
#	convert_yaml_to_hdf5(metadataPath, hdfFilename)
		
'''if 'electricity' in inputPath:
			if scheme==1:
				fp=pd.read_csv(join(inputPath, sent))
				key=join('electricity', 'WHE' + str(i+1))
				store.append(key, fp)
			else:
				fp=pd.read_csv(join(inputPath, sent))
                                key=join('electricity', str(sent)[:len(str(sent)) - 4] + str(i+1))
                                store.append(key, fp)
		if 'natural_gas' in inputPath:
			if scheme==1:
				fp=pd.read_csv(join(inputPath, sent))
                        	key=join('natural_gas', 'WHE' + str(i+1))
                        	store.append(key, fp)
			else:
				fp=pd.read_csv(join(inputPath, sent))
                                key=join('natural_gas', str(sent)[:len(str(sent)) - 4] + str(i+1))
                                store.append(key, fp)
		if 'water' in inputPath:
			if scheme==1:
				fp=pd.read_csv(join(inputPath, sent))
                        	key=join('water', 'WHE' + str(i+1))
                        	store.append(key, fp)
			else:
				fp=pd.read_csv(join(inputPath, sent))
                                key=join('water', str(sent)[:len(str(sent)) - 4] + str(i+1))
                                store.append(key, fp)'''
	
#	print(store)
#	print(store['/natural_gas/FRG1'])
#	metadata_path=join(inputPath, 'metadata.nilmtk')
#	print (metadataPath)
#	convert_yaml_to_hdf5(metadataPath, store)
#	store.close()
def test():
	inputPath='/Users/rishi/Documents/Master_folder/IIITD/5th_semester/Independent_Project/AMPds/electricity'
#	inputPath='/Users/rishi/Documents/Master_folder/IIITD/5th_semester/Independent_Project/Forked/nilmtk/nilmtk/dataset_converters/AMPDs'
	fileName='/Users/rishi/Documents/Master_folder/IIITD/5th_semester/Independent_Project/AMPds/electricity/store1.h5'
#	fileName='store.h5'
	ip1=join(inputPath, 'natural_gas')
	metadataPath='/Users/rishi/Documents/Master_folder/IIITD/5th_semester/Independent_Project/Forked/nilmtk/nilmtk/dataset_converters/AMPDs/metadata.nilmtk'
	convert(inputPath, fileName, metadataPath)

test()
