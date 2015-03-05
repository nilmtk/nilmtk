import pandas as pd

class MaximumLikelihood(object):
    """
    Disaggregation of a single appliance based on its features. 

    Attributes
    ----------
    model : dict
        Each key is either the instance integer for an ElecMeter, 
        or a tuple of instances for a MeterGroup.
        Each value is a sorted list of power in different states.
    """

    def __init__(self):
    	
        self.model = {}
        self.features = {}
 

    def load_features(self, **kwargs): 
        """

        """
    for key in kwargs: 
        self.features[key] = kwargs[key]

    def pair(self):
        """

        """

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



