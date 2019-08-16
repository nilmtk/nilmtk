from nilmtk.dataset import DataSet
from nilmtk.metergroup import MeterGroup
import pandas as pd
from disaggregate import CombinatorialOptimisation, Mean, FHMM, Zero, DAE, Seq2Point, Seq2Seq, DSC#, AFHMM,AFHMM_SAC 
from disaggregate import Disaggregator
from six import iteritems
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
import numpy as np
import matplotlib.pyplot as plt
import datetime
class API():

	"""
	The API ia designed for rapid experimentation with NILM Algorithms. 

	"""

	def __init__(self,params):

		"""
		Initializes the API with default parameters
		"""
		self.power = {}
		self.sample_period = 1
		self.appliances = []
		self.methods = {}
		self.chunk_size = None
		self.method_dict={'CO':{},'FHMM':{},'Hart85':{},'DAE':{},'Mean':{},'Zero':{},'WindowGRU':{},'Seq2Point':{}, 'RNN':{},'Seq2Seq':{},'DSC':{},'AFHMM':{},'AFHMM_SAC':{}}
		self.pre_trained = False
		self.metrics = []
		self.train_datasets_dict = {}
		self.test_datasets_dict = {}
		self.artificial_aggregate = False
		self.train_submeters = []
		self.train_mains = pd.DataFrame()
		self.test_submeters = []
		self.test_mains = pd.DataFrame()
		self.gt_overall = {}
		self.pred_overall = {}
		self.classifiers=[]
		self.DROP_ALL_NANS = True
		self.mae = pd.DataFrame()
		self.rmse = pd.DataFrame()
		self.experiment(params)

	
	def initialise(self,params):

		"""
		Instantiates the API with the specified Parameters
		"""
		for elems in params['params']['power']:
			self.power = params['params']['power']
		self.sample_period = params['sample_rate']
		for elems in params['appliances']:
			self.appliances.append(elems)
		
		self.pre_trained = ['pre_trained']
		self.train_datasets_dict = params['train']['datasets']
		self.test_datasets_dict = params['test']['datasets']
		self.metrics = params['test']['metrics']
		self.methods = params['methods']
		self.artificial_aggregate = params.get('artificial_aggregate',self.artificial_aggregate)
		self.chunk_size = params.get('chunk_size',self.chunk_size)

	def experiment(self,params):
		"""
		Calls the Experiments with the specified parameters
		"""
		self.params=params
		self.initialise(params)

		if params['chunk_size']:
			# This is for training and Testing in Chunks
			self.load_datasets_chunks()
		else:
			# This is to load all the data from all buildings and use it for training and testing. This might not be possible to execute on computers with low specs
			self.load_datasets()
		
	def load_datasets_chunks(self):

		"""
		This function loads the data from buildings and datasets with the specified chunk size and trains on each of them. 

		After the training process is over, it tests on the specified testing set whilst loading it in chunks.

		"""
		# First, we initialize all the models
		self.store_classifier_instances()

		d=self.train_datasets_dict
		for model_name, clf in self.classifiers:

			# If the model is a neural net, it has an attribute n_epochs, Ex: DAE, Seq2Point
			if hasattr(clf,'n_epochs'):
				epochs = clf.n_epochs
			# If it doesn't have the attribute n_epochs, this is executed. Ex: Mean, Zero
			else:
				epochs = 1
			
			# If the model has the filename specified for loading the pretrained model, then we don't need to load training data
			if clf.load_model_path:
				print (clf.MODEL_NAME," is loading the pretrained model")
				continue

			for q in range(epochs):
				for dataset in d:
					print("Loading data for ",dataset, " dataset")			
					for building in d[dataset]['buildings']:
							train=DataSet(d[dataset]['path'])
							print("Loading building ... ",building)
							train.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
							mains_iterator = train.buildings[building].elec.mains().load(chunksize = self.chunk_size, physical_quantity='power', ac_type = self.power['mains'], sample_period=self.sample_period)
							print (self.appliances)
							appliance_iterators = [train.buildings[building].elec.select_using_appliances(type=app_name).load(chunksize = self.chunk_size, physical_quantity='power', ac_type=self.power['appliance'], sample_period=self.sample_period) for app_name in self.appliances]
							print(train.buildings[building].elec.mains())
							for chunk_num,chunk in enumerate (train.buildings[building].elec.mains().load(chunksize = self.chunk_size, physical_quantity='power', ac_type = self.power['mains'], sample_period=self.sample_period)):
								#Dummry loop for executing on outer level. Just for looping till end of a chunk
								print("starting enumeration..........")
								train_df = next(mains_iterator)
								appliance_readings = []
								for i in appliance_iterators:
									try:
										appliance_df = next(i)
									except StopIteration:
										pass
									appliance_readings.append(appliance_df)

								if self.DROP_ALL_NANS:
									train_df, appliance_readings = self.dropna(train_df, appliance_readings)
								
								if self.artificial_aggregate:
									print ("Creating an Artificial Aggregate")
									train_df = pd.DataFrame(np.zeros(appliance_readings[0].shape),index = appliance_readings[0].index,columns=appliance_readings[0].columns)
									for app_reading in appliance_readings:
										train_df+=app_reading
								train_appliances = []

								for cnt,i in enumerate(appliance_readings):
									train_appliances.append((self.appliances[cnt],[i]))

								self.train_mains = [train_df]
								self.train_submeters = train_appliances
								clf.partial_fit(self.train_mains,self.train_submeters)
								

		print("...............Finished the Training Process ...................")

		print("...............Started  the Testing Process ...................")

		d=self.test_datasets_dict
		for dataset in d:
			print("Loading data for ",dataset, " dataset")
			for building in d[dataset]['buildings']:
				test=DataSet(d[dataset]['path'])
				test.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
				mains_iterator = test.buildings[building].elec.mains().load(chunksize = self.chunk_size, physical_quantity='power', ac_type = self.power['mains'], sample_period=self.sample_period)
				appliance_iterators = [test.buildings[building].elec.select_using_appliances(type=app_name).load(chunksize = self.chunk_size, physical_quantity='power', ac_type=self.power['appliance'], sample_period=self.sample_period) for app_name in self.appliances]

				for chunk_num,chunk in enumerate (test.buildings[building].elec.mains().load(chunksize = self.chunk_size, physical_quantity='power', ac_type = self.power['mains'], sample_period=self.sample_period)):
					test_df = next(mains_iterator)
					appliance_readings = []
					for i in appliance_iterators:
						try:
							appliance_df = next(i)
						except StopIteration:
							appliance_df = pd.DataFrame()

						appliance_readings.append(appliance_df)

					if self.DROP_ALL_NANS:
						test_df, appliance_readings = self.dropna(test_df, appliance_readings)

					if self.artificial_aggregate:
						print ("Creating an Artificial Aggregate")
						test_df = pd.DataFrame(np.zeros(appliance_readings[0].shape),index = appliance_readings[0].index,columns=appliance_readings[0].columns)
						for app_reading in appliance_readings:
							test_df+=app_reading

					test_appliances = []

					for cnt,i in enumerate(appliance_readings):
						test_appliances.append((self.appliances[cnt],[i]))

					self.test_mains = [test_df]
					self.test_submeters = test_appliances
					print("Results for Dataset {dataset} Building {building} Chunk {chunk_num}".format(dataset=dataset,building=building,chunk_num=chunk_num))
					self.call_predict(self.classifiers)

					
	def dropna(self,mains_df, appliance_dfs):
		"""
		Drops the missing values in the Mains reading and appliance readings and returns consistent data by copmuting the intersection
		"""
		print ("Dropping missing values")

		# The below steps are for making sure that data is consistent by doing intersection across appliances
		mains_df = mains_df.dropna()
		for i in range(len(appliance_dfs)):
			appliance_dfs[i] = appliance_dfs[i].dropna()
		ix = mains_df.index
		for  app_df in appliance_dfs:
			ix = ix.intersection(app_df.index)
		mains_df = mains_df.loc[ix]
		new_appliances_list = []
		for app_df in appliance_dfs:
			new_appliances_list.append(app_df.loc[ix])
		return mains_df,new_appliances_list

	def load_datasets(self):

		# This function has a few issues, which should be addressed soon
		self.store_classifier_instances()
		d=self.train_datasets_dict
		
		print("............... Loading Data for training ...................")
		
		# store the train_main readings for all buildings
		for dataset in d:
			print("Loading data for ",dataset, " dataset")
			train=DataSet(d[dataset]['path'])
			for building in d[dataset]['buildings']:
				print("Loading building ... ",building)
				train.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
				self.train_mains=self.train_mains.append(next(train.buildings[building].elec.mains().load(physical_quantity='power', ac_type=self.power['mains'], sample_period=self.sample_period)))		
						


		# store train submeters reading
		train_buildings=pd.DataFrame()
		for appliance in self.appliances:
			train_df=pd.DataFrame()
			print("For appliance .. ",appliance)
			for dataset in d:
				print("Loading data for ",dataset, " dataset")
				train=DataSet(d[dataset]['path'])
				for building in d[dataset]['buildings']:
					print("Loading building ... ",building)
					
					# store data for submeters
					train.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
					train_df=train_df.append(next(train.buildings[building].elec.submeters().select_using_appliances(type=appliance).load(physical_quantity='power', ac_type=self.power['appliance'], sample_period=self.sample_period)))
					


			
			self.train_submeters.append((appliance,[train_df]))	
				
		# create instance of the training methods
		
		# train models

		# store data for mains
		self.train_mains = [self.train_mains]
		self.call_partial_fit()

		d=self.test_datasets_dict

		# store the test_main readings for all buildings
		for dataset in d:
			print("Loading data for ",dataset, " dataset")
			test=DataSet(d[dataset]['path'])
			for building in d[dataset]['buildings']:
				test.set_window(start=d[dataset]['buildings'][building]['start_time'],end=d[dataset]['buildings'][building]['end_time'])
				self.test_mains=(next(test.buildings[building].elec.mains().load(physical_quantity='power', ac_type=self.power['mains'], sample_period=self.sample_period)))		
				self.test_submeters=[]
				for appliance in self.appliances:
					test_df=next((test.buildings[building].elec.submeters().select_using_appliances(type=appliance).load(physical_quantity='power', ac_type=self.power['appliance'], sample_period=self.sample_period)))
					self.test_submeters.append((appliance,[test_df]))
				self.test_mains = [self.test_mains]
				self.call_predict(self.classifiers)
				
	
	def store_classifier_instances(self):

		"""
		This function is reponsible for initializing the models with the specified model parameters
		"""
		method_dict={}
		for i in self.method_dict:
			if i in self.methods:
				self.method_dict[i].update(self.methods[i])


		method_dict={'CO':CombinatorialOptimisation(self.method_dict['CO']),
					'FHMM':FHMM(self.method_dict['FHMM']),
					'DAE':DAE(self.method_dict['DAE']),
					'Mean':Mean(self.method_dict['Mean']),
					'Zero':Zero(self.method_dict['Zero']),
					'Seq2Seq':Seq2Seq(self.method_dict['Seq2Seq']),
					'Seq2Point':Seq2Point(self.method_dict['Seq2Point']),
					'DSC':DSC(self.method_dict['DSC']),
					# 'AFHMM':AFHMM(self.method_dict['AFHMM']),
					# 'AFHMM_SAC':AFHMM_SAC(self.method_dict['AFHMM_SAC'])				
					#'RNN':RNN(self.method_dict['RNN'])
					}

		for name in self.methods:
			if name in method_dict:
				clf=method_dict[name]
				self.classifiers.append((name,clf))
			else:
				print ("\n\nThe method {model_name} specied does not exist. \n\n".format(model_name=i))


	
	def call_predict(self,classifiers):

		"""
		This functions computers the predictions on the self.test_mains using all the trained models and then compares different learn't models using the metrics specified
		"""
		
		pred_overall={}
		gt_overall={}
		for name,clf in classifiers:
			gt_overall,pred_overall[name]=self.predict(clf,self.test_mains,self.test_submeters, self.sample_period,'Europe/London')

		self.gt_overall=gt_overall
		self.pred_overall=pred_overall

		for i in gt_overall.columns:
			plt.figure()
			plt.plot(gt_overall[i],label='truth')
			for clf in pred_overall:
				plt.plot(pred_overall[clf][i],label=clf)
			plt.title(i)
			plt.legend()

		if gt_overall.size==0:
			print ("No samples found in ground truth")
			return None

		for metric in self.metrics:
			
			if metric=='f1-score':
				f1_score={}
				
				for clf_name,clf in classifiers:
					f1_score[clf_name] = self.compute_f1_score(gt_overall, pred_overall[clf_name])
				f1_score = pd.DataFrame(f1_score)
				print("............ " ,metric," ..............")
				print(f1_score)	
				
			elif metric=='rmse':
				rmse = {}
				for clf_name,clf in classifiers:
					rmse[clf_name] = self.compute_rmse(gt_overall, pred_overall[clf_name])
				rmse = pd.DataFrame(rmse)
				self.rmse = rmse
				print("............ " ,metric," ..............")
				print(rmse)	

			elif metric=='mae':
				mae={}
				for clf_name,clf in classifiers:
					mae[clf_name] = self.compute_mae(gt_overall, pred_overall[clf_name])
				mae = pd.DataFrame(mae)
				self.mae = mae
				print("............ " ,metric," ..............")
				print(mae)  

			elif metric == 'rel_error':
				rel_error={}
				for clf_name,clf in classifiers:
					rel_error[clf_name] = self.compute_rel_error(gt_overall, pred_overall[clf_name])
				rel_error = pd.DataFrame(rel_error)
				print("............ " ,metric," ..............")
				print(rel_error)			
			else:
				print ("The requested metric {metric} does not exist.".format(metric=metric))
					
	def predict(self, clf, test_elec, test_submeters, sample_period, timezone):
		
		"""
		Generates predictions on the test dataset using the specified classifier.
		"""
		
		# "ac_type" varies according to the dataset used. 
		# Make sure to use the correct ac_type before using the default parameters in this code.   
		
			
		pred_list = clf.disaggregate_chunk(test_elec)

		# It might not have time stamps sometimes due to neural nets
		# It has the readings for all the appliances

		concat_pred_df = pd.concat(pred_list,axis=0)

		gt = {}
		for meter,data in test_submeters:
				concatenated_df_app = pd.concat(data,axis=1)
				index = concatenated_df_app.index
				gt[meter] = pd.Series(concatenated_df_app.values.flatten(),index=index)

		gt_overall = pd.DataFrame(gt, dtype='float32')
		
		pred = {}

		for app_name in concat_pred_df.columns:

			app_series_values = concat_pred_df[app_name].values.flatten()

			# Neural nets do extra padding sometimes, to fit, so get rid of extra predictions

			app_series_values = app_series_values[:len(gt_overall[app_name])]

			#print (len(gt_overall[app_name]),len(app_series_values))

			pred[app_name] = pd.Series(app_series_values, index = gt_overall.index)

		pred_overall = pd.DataFrame(pred,dtype='float32')


		#gt[i] = pd.DataFrame({k:v.squeeze() for k,v in iteritems(gt[i]) if len(v)}, index=next(iter(gt[i].values())).index).dropna()

		# If everything can fit in memory

		#gt_overall = pd.concat(gt)
		# gt_overall.index = gt_overall.index.droplevel()
		# #pred_overall = pd.concat(pred)
		# pred_overall.index = pred_overall.index.droplevel()

		# Having the same order of columns
		# gt_overall = gt_overall[pred_overall.columns]

		# #Intersection of index
		# gt_index_utc = gt_overall.index.tz_convert("UTC")
		# pred_index_utc = pred_overall.index.tz_convert("UTC")
		# common_index_utc = gt_index_utc.intersection(pred_index_utc)

		# common_index_local = common_index_utc.tz_convert(timezone)
		# gt_overall = gt_overall.loc[common_index_local]
		# pred_overall = pred_overall.loc[common_index_local]
		# appliance_labels = [m for m in gt_overall.columns.values]
		# gt_overall.columns = appliance_labels
		# pred_overall.columns = appliance_labels
		return gt_overall, pred_overall


	# metrics
	def compute_mae(self,gt,pred):
		"""
		Computes the Mean Absolute Error between Ground truth and Prediction
		"""

		mae={}
		for appliance in gt.columns:
			mae[appliance]=mean_absolute_error(gt[appliance],pred[appliance])
		return pd.Series(mae)


	def compute_rmse(self,gt, pred):
		"""
		Computes the Root Mean Squared Error between Ground truth and Prediction
		"""
		rms_error = {}
		for appliance in gt.columns:
			rms_error[appliance] = np.sqrt(mean_squared_error(gt[appliance], pred[appliance]))
		#print (gt['sockets'])
		#print (pred[])
		return pd.Series(rms_error)
	
	def compute_f1_score(self,gt, pred):
		"""
		Computes the F1 Score between Ground truth and Prediction
		"""	
		f1 = {}
		gttemp={}
		predtemp={}
		for appliance in gt.columns:
			gttemp[appliance] = np.array(gt[appliance])
			gttemp[appliance] = np.where(gttemp[appliance]<10,0,1)
			predtemp[appliance] = np.array(pred[appliance])
			predtemp[appliance] = np.where(predtemp[appliance]<10,0,1)
			f1[appliance] = f1_score(gttemp[appliance], predtemp[appliance])
		return pd.Series(f1)

	def compute_rel_error(self,gt,pred):

		"""
		Computes the Relative Error between Ground truth and Prediction
		"""
		rel_error={}
		for appliance in gt.columns:
			rel_error[appliance] = np.sum(np.sum(abs(gt[appliance]-pred[appliance]))/len(gt[appliance]))
		return pd.Series(rel_error)	
