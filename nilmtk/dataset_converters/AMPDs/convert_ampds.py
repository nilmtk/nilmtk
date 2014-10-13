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

def convert(inputPath, hdfFilename):
	files=[f for f in listdir(inputPath) if isfile (join(inputPath, f)) and '.csv' in f]
#	print(files)
	store=HDFStore(hdfFilename)
	for i in range(len(files)):
		sent=files[i]
		if 'electricity' in inputPath:
			fp=pd.read_csv(join(inputPath, sent))
			key=join('electricity', 'WHE' + str(i+1))
			store.append(key, fp)
		if 'natural_gas' in inputPath:
			fp=pd.read_csv(join(inputPath, sent))
                        key=join('natural_gas', 'WHE' + str(i+1))
                        store.append(key, fp)
		if 'water' in inputPath:
			fp=pd.read_csv(join(inputPath, sent))
                        key=join('water', 'WHE' + str(i+1))
                        store.append(key, fp)
#	print(store)
#inputPath='/Users/rishi/Documents/Master_folder/IIITD/5th_semester/Independent_Project/AMPds'
#fileName='/Users/rishi/Documents/Master_folder/IIITD/5th_semester/Independent_Project/AMPds/electricity/store.h5'
#fileName='store.h5'
#ip1=join(inputPath, 'natural_gas')
#convert(ip1, fileName)
