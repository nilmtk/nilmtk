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

def convert(inputPath, hdfFilename, scheme=1, metadataPath='/'):
# This function contains the bulk of the code. The test() function can simply be ignored for now
# To do: Complete the metadata set. Then the convert_yaml_to_hdf5() function will stop throwing random errors.
	files=[f for f in listdir(inputPath) if isfile (join(inputPath, f)) and '.csv' in f]
#	print(files)
	store=HDFStore(hdfFilename)
	for i in range(len(files)):
		sent=files[i]
		if 'electricity' in inputPath:
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
                                store.append(key, fp)
	
#	print(store)
#	print(store['/natural_gas/FRG1'])
#	metadata_path=join(inputPath, 'metadata.nilmtk')
#	print (metadataPath)
	convert_yaml_to_hdf5(metadataPath, store)
	store.close()
def test():
	inputPath='/Users/rishi/Documents/Master_folder/IIITD/5th_semester/Independent_Project/AMPds'
#	inputPath='/Users/rishi/Documents/Master_folder/IIITD/5th_semester/Independent_Project/Forked/nilmtk/nilmtk/dataset_converters/AMPDs'
	fileName='/Users/rishi/Documents/Master_folder/IIITD/5th_semester/Independent_Project/AMPds/electricity/store.h5'
#	fileName='store.h5'
	ip1=join(inputPath, 'natural_gas')
	metadataPath='/Users/rishi/Documents/Master_folder/IIITD/5th_semester/Independent_Project/Forked/nilmtk/nilmtk/dataset_converters/AMPDs/metadata.nilmtk'
	convert(ip1, fileName, 0, metadataPath)

test()
