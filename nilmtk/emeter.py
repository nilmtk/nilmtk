from __future__ import print_function, division
from warnings import warn

class EMeter(object):
    """Represents a physical electricity meter.
    
    Attributes
    ----------
    loader : Loader
    
    metadata : dict.  Including keys:
        id : int, meter ID    
        submeter_of : int, ID of upstream meter       
        preprocessing : list of strings (why not actual Node objects?), 
          each describing a preprocessing Node.
          preprocessing to be applied before returning any stats answers; or before exporting.
          e.g. power normalisation or removing gaps.  Properties:
          - 'gaps_bookended_with_zeros': bool
          - 'energy_computed': bool
        device_model : string, the model name of the meter device.

        --------- THE FOLLOWING ATTRIBUTES ARE SET AUTOMATICALLY, ---------
        --------- i.e. THEY DO NOT EXIST IN THE ON-DISK METADATA. ---------

        device : dict (instantiated from meter_devices static class attribute)
    
    meter_devices : dict, static class attribute
        Keys are devices.  Values are dicts:
            manufacturer : string
            model : string, model name
            sample_period : float, seconds
            max_sample_period : float, seconds
            measurements : list of nilmtk.measurement objects, e.g.
                [Power('active'), Voltage()]
            measurement_limits : dict, e.g.:
                {Power('active'): [0, 3000]}

    """
    meter_devices = {}

    @classmethod
    def _load_meter_devices(cls, loader):
        dataset_metadata = loader.store.load_metadata()
        EMeter.meter_devices.update(dataset_metadata.get('meter_devices', {}))

    def load(self, loader):
        self.loader = loader
        self.metadata = self.loader.load_metadata()
        EMeter._load_meter_devices(loader)
        device_model = self.metadata['device_model']
        self.metadata['device'] = EMeter.meter_devices[device_model]

    def save(self, destination, key):
        """
        Convert all relevant attributes to a dict to be 
        saved as metadata in destination at location specified
        by key
        """
        # destination.write_metadata(key, self.metadata)
        raise NotImplementedError
                
    def power_series(self, measurement_preferences=None, 
                     required_measurement=None,
                     normalise=False, voltage_series=None, 
                     nominal_voltage=None):
        """Power timeseries.
        
        Set meter.loader parameters to configure chunk sizes, start date etc.
        
        The following cleaning steps will be run if the relevant entries
        in meter.cleaning are True:

        * remove implausable values
        * gaps will be bookended with zeros
        
        Parameters
        ----------
        measurement_preferences : list of Measurements, optional. Defaults to active > apparent > reactive
        required_measurements : Measurement, optional.  Raises MeasurementError if not available.
        normalise : boolean, optional, defaults to False
        voltage_series : EMeter object with voltage measurements available.  If not supplied and if normalise is True
            then will attempt to use voltage data from this meter.
        nominal_voltage : float

        
        Returns
        -------
        generator of pd.Series of power measurements.
        """
        raise NotImplementedError
        
    def voltage_series(self):
        """Returns a generator of pd.Series of voltage, if avilable."""
        raise NotImplementedError
        
    def total_energy(self, preprocessing=None):
        """returns an EnergyResults object"""
        raise NotImplementedError
        # nodes = [] if preprocessing is None else preprocessing
        # nodes.extend([RemoveImplausableValues(self.measurement_limits), 
        #               BookendGapsWithZeros(), 
        #               Energy()])
        # pipeline = Pipeline(self, nodes)
        # pipeline.run()
        # return pipline.results['energy']
        
    def dropout_rate(self):
        """returns a DropoutRateResults object."""
        raise NotImplementedError
        
    def gaps(self): 
        """returns Mask object"""
        raise NotImplementedError
        
    def contiguous_sections(self):
        """retuns Mask object"""
        raise NotImplementedError
        
    def clean_and_export(self, destination_datastore):
        """Apply all cleaning configured in meter.cleaning and then export.  Also identifies
        and records the locations of gaps.  Also records metadata about exactly which
        cleaning steps have been executed and some summary results (e.g. the number of
        implausible values removed)"""
        raise NotImplementedError
        
    def set_loader_attributes(self, **kwargs):
        """Provides a common interface to setting loader attributes.
        e.g. set_load_attributes(mask=Mask())
        """
        for key, value in kwargs.iteritems():
            self.loader.__setattr__(key, value)
        
    def reset_loader_attributes(self):
        self.loader.reset()
        
    def get_loader_attribute(self, attribute):
        return self.loader.__getattr__(attribute)
