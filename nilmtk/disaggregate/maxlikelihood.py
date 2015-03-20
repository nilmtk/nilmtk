import pandas as pd
from disaggregator import Disaggregator
#from datetime import datetime as datetime2
from datetime import timedelta

#from ipdb import set_trace as _breakpoint

class MaximumLikelihood(Disaggregator):
	"""
	Disaggregation of a single appliance based on its features and
	using the maximum likelihood for all features.

	Attributes
	----------
	model : dict
	    Each key is either the instance integer for an ElecMeter, 
	    or a tuple of instances for a MeterGroup.
	    Each value is a sorted list of power in different states.
	"""

	def __init__(self):
    	
		super(MaximumLikelihood, self).__init__()
		self.features = {}
 

	def load_features(self, **kwargs): 
		"""

		"""
		print "Updating parameters in features"
		for key in kwargs: 
			self.features[key] = kwargs[key]
			print key, kwargs[key]


#	def event_detector(self,chunk, units=('power','apparent'), power_limit=50, paired_event_limit=100):
#		"""
#		
#		"""
#
#		column_name = 'diff_' + units[1]
#		chunk[column_name] = chunk.loc[:,units].diff()
#		# Event detection 
#		onpower = (chunk[column_name] > power_limit)
#		offpower = (chunk[column_name] < -power_limit)
#		events = chunk[(onpower == True) | (offpower == True)]
#
#		return events[column_name]


	def disaggregate_chunk(self, chunk, units=('power','apparent'), power_limit=50, paired_limit=100, duration=400):
		"""
		Loads all of a DataFrame from disk.

		Parameters
		----------
		chunk : pd.DataFrame (in NILMTK format)
		
		Returns
		-------
		chunk : pd.DataFrame where each column represents a disaggregated appliance
		"""

		column_name = 'diff_' + units[1]
		chunk[column_name] = chunk.loc[:,units].diff()
		# Event detection 
		chunk['onpower'] = (chunk[column_name] > power_limit)
		chunk['offpower'] = (chunk[column_name] < -power_limit)
		events = chunk[(chunk.onpower == True) | (chunk.offpower == True)]

		kettle_detection = []
		# Max Likelihood algorithm (optimized): 
		for onevent in events[events.onpower==True].iterrows(): 
			start_date = onevent[0]
			delta = onevent[1][1]
			offevents = events[(events.offpower == True) & (events.index > start_date) & (events.index < start_date + timedelta(seconds=duration))]
			offevents = offevents[abs(delta - offevents[column_name].abs()) < paired_limit]
			#import ipdb; ipdb.set_trace()
			print "hi" 
		if not offevents.empty: 
			print "paired_event"
		#	pon = np.exp(onpower_gm.score([donvalue_n]))[0]
		#	for offevent in offevents.iterrows(): 
		#		offindex = offevent[0]
		#		end_date = offevent[1][0]
		#		doffvalue_n = offevent[1][3]
		#		
		#		duration = end_date - start_date
		#		poff = np.exp(offpower_gm.score([doffvalue_n]))[0]
		#		pduration = duration_poisson.pmf(duration.total_seconds())
		#		likelihood = pon * poff * pduration
		#		kettle_detection.append({'likelihood': likelihood, 'onindex':onindex, 'offindex': offindex})

	#	raise NotImplementedError("NotImplementedError")


	def disaggregate(self, mains, output_datastore, **load_kwargs):
		"""
		Disaggregate mains according to the model learnt previously.

		Parameters
		----------
		mains : nilmtk.ElecMeter or nilmtk.MeterGroup
		output_datastore : instance of nilmtk.DataStore subclass
		    For storing power predictions from disaggregation algorithm.
		output_name : string, optional
		    The `name` to use in the metadata for the `output_datastore`.
		    e.g. some sort of name for this experiment.  Defaults to 
		    "NILMTK_Hart85_<date>"
		resample_seconds : number, optional
		    The desired sample period in seconds.
		**load_kwargs : key word arguments
		    Passed to `mains.power_series(**kwargs)`
		"""

	#	raise NotImplementedError("NotImplementedError")

