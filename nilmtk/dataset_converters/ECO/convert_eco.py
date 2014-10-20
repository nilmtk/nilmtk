import pandas as pd
import numpy as np
import sys
from os import listdir
from os.path import isdir, join
from pandas.tools.merge import concat
from nilmtk.utils import get_module_directory
from nilmtk.datastore import Key

"""
PROBLEMS:
---------
Refer to the blog @ nilmtkmridul.github.io to see current problems being faced.

"""

sm_column_name = {1:('power', 'apparent'),
					2:('power', 'apparent'),
					3:('power', 'apparent'),
					4:('power', 'apparent'),
					5:('current', ''),
					6:('current', ''),
					7:('current', ''),
					8:('current', ''),
					9:('voltage', ''),
					10:('voltage', ''),
					11:('voltage', ''),
					12:('phase angle', 'apparent'), #What property to assign to the phase angle?
					13:('phase angle', 'apparent'),
					14:('phase angle', 'apparent'),
					15:('phase angle', 'apparent'),
					16:('phase angle', 'apparent'),
					};

plugs_column_name = {1:('power', 'apparent'),	
					};

def _get_df(dir_loc, csv_name, meter_flag):

	#Changing column length for Smart Meters and Plugs
	column_num = 16 if meter_flag == 'sm' else 1

	#Reading the CSV file and adding a datetime64 index to the Dataframe
	df = pd.read_csv(join(dir_loc,csv_name), names=[i for i in range(1,column_num+1)])
	df_index = pd.date_range(csv_name[:-4], periods=86400, freq = 's', tz = 'GMT')
	df.index = pd.to_datetime(df_index)

	if meter_flag == 'sm':
		df.rename(columns=sm_column_name, inplace=True)
	else:
		df.rename(columns=plugs_column_name, inplace=True)

	#print 'Dataframe for file:',csv_name,'completed.'
	return df

def convert_eco(dataset_loc, hdf_file, timezone):
	"""
	Parameters:
	-----------
	dataset_loc: 	defined the root directory where the dataset is located.

	hdf_file: 		defines the location where the hdf_file is present. The name has 
					to be alongside the directory location for the converter to work.

	timezone:		specifies the timezone of the dataset
	"""

	#Creating a new HDF File
	store = pd.HDFStore(hdf_file, 'w')

	"""
	DATASET STRUCTURE:
	------------------
	ECO Dataset has folder '<i>_sm_csv' and '<i>_plug_csv' where i is the building no.

	<i>_sm_csv has a folder 01
	<i>_plug_csv has a folder 01, 02,....<n> where n is the plug numbers.

	Each folder has a CSV file as per each day, with each day csv file containing
		86400 entries.
	"""

	assert isdir(dataset_loc)
	directory_list = [i for i in listdir(dataset_loc) if '.txt' not in i]
	directory_list.sort()
	print directory_list

	#Traversing every folder
	for folder in directory_list:
		print 'Computing for folder',folder

		#Building number and meter_flag
		building_no = int(folder[:2])
		meter_flag = 'sm' if 'sm_csv' in folder else 'plugs'

		dir_list = [i for i in listdir(join(dataset_loc, folder)) if isdir(join(dataset_loc,folder,i))]
		dir_list.sort()
		print 'Current dir list:',dir_list
		for fl in dir_list:
			df = pd.DataFrame()

			#Meter number to be used in key
			meter_num = 1 if meter_flag == 'sm' else int(fl) + 1
			print 'Computing for Meter no.',meter_num

			fl_dir_list = [i for i in listdir(join(dataset_loc,folder,fl)) if '.csv' in i]
			fl_dir_list.sort()
			for fi in fl_dir_list:

				#Getting dataframe for each csv file
				df_fl = _get_df(join(dataset_loc,folder,fl),fi,meter_flag)

				#Merging with the current Dataframe
				df = concat([df,df_fl])
				print 'Done for ',fi[:-4]

			df.sort_index(ascending=True,inplace=True)
			df = df.tz_convert(timezone)
			print df[:5],'\n',df[-5:]

			#HDF5 file operations
			key = Key(building=building_no, meter=meter_num)
			store.put(str(key), df, format='Table')
			store.flush()
			#temp = raw_input()

	store.close()

	#Adding the metadata to the HDF5file
	meta_path = join(get_module_directory(), 'metadata')
	convert_yaml_to_hdf5(meta_path, hdf_file)

#Sample entries for checking
dataset_loc = '/home/mridul/nilmtkProject/ECODataset/Dataset'
hdf_file = '/home/mridul/nilmtkProject/ECODataset/hdf5store.h5'
convert_eco(dataset_loc, hdf_file ,'GMT')
