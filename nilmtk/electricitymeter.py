from __future__ import print_function, division
from nilmtk.pipeline import Pipeline, ClipNode, EnergyNode, LocateGoodSectionsNode
from .hashable import Hashable
from warnings import warn
from collections import namedtuple

ElectricityMeterID = namedtuple('ElectricityMeterID',    
                                ['instance', 'building', 'dataset'])

class ElectricityMeter(Hashable):
    """Represents a physical electricity meter.
    
    Attributes
    ----------
    appliances : set of Appliance objects connected immediately downstream
      of this meter.  Will be empty set if no appliances are connected directly
      to this meter.

    mains : Mains (used so appliance methods can default to use
      the same measured parameter (active / apparent / reactive) 
      as Mains; and also for use in proportion of energy submetered
      and for voltage normalisation.)

    store : nilmtk.DataStore

    key : key into nilmtk.DataStore to access data

    identifier : ElectricityMeterID namedtuple with fields:
        instance : int, meter instance within this building, starting from 1
        building : int, building instance, starting from 1
        dataset : str e.g. 'REDD'
    
    metadata : dict.  Including keys:
        submeter_of : int, instance of upstream meter
        site_meter : bool, True if this is a site meter (i.e. furthest upstream meter)
        preprocessing : list of strings (why not actual Node objects?), 
          each describing a preprocessing Node.
          preprocessing to be applied before returning any stats answers; or before exporting.
          e.g. power normalisation or removing gaps.  Properties:
          - 'good_sections_located': bool
          - 'energy_computed': bool
        device_model : string, the model name of the meter device.
        dominant_appliance : reference to Appliance which is responsibly for 
          most of the power demand on this channel.

        --------- THE FOLLOWING ATTRIBUTES ARE SET AUTOMATICALLY, ---------
        --------- i.e. THEY DO NOT EXIST IN THE ON-DISK METADATA. ---------

        device : dict (instantiated from meter_devices static class attribute)
    

    STATIC ATTRIBUTES
    -----------------

    meter_devices : dict, static class attribute
        Keys are devices.  Values are dicts:
            manufacturer : string
            model : string, model name
            sample_period : float, seconds
            max_sample_period : float, seconds
            measurements : list of nilmtk.measurement objects, e.g.
                [Power('active'), Voltage()]
            measurement_limits : dict, e.g.:
                {Power('active'): {'lower': 0, 'upper': 3000}}

    meters : dict, static class attribute:
        Required for resolving `upstream_of` to an ElectricityMeter object
        Keys are ElectricityMeterID objects.
        Values are ElectricityMeter objects.
    """
    meter_devices = {}
    meters = {}

    def __init__(self, instance, building, dataset, 
                 appliances=None, metadata=None):
        assert isinstance(instance, int)
        assert isinstance(building, int)
        assert isinstance(dataset, str)
        self.identifier = ElectricityMeterID(instance, building, dataset)
        self.store = None
        self.key = None
        self.appliances = set() if appliances is None else set(appliances)
        self.metadata = {} if metadata is None else metadata
        self.mains = None
        self.dominant_appliance = None
        ElectricityMeter.meters[self.identifier] = self

    @property
    def upstream_meter(self):
        submeter_of = self.metadata.get('submeter_of')
        if submeter_of is None:
            return
        if submeter_of == 0:
            return self.mains 
        id_of_upstream = ElectricityMeterID(instance=submeter_of, 
                                            building=self.identifier.building, 
                                            dataset=self.identifier.dataset,)
        upstream_meter =  ElectricityMeter.meters[id_of_upstream]
        if upstream_meter.metadata.get('site_meter'):
            return self.mains
        else:
            return upstream_meter

    @classmethod
    def _load_meter_devices(cls, store):
        dataset_metadata = store.load_metadata()
        ElectricityMeter.meter_devices.update(dataset_metadata.get('meter_devices', {}))

    # TODO: why not just have this as __init__(store)???
    def load(self, store, key):
        self.store = store
        self.key = key
        self.metadata = self.store.load_metadata(self.key)
        ElectricityMeter._load_meter_devices(store)
        device_model = self.metadata['device_model']
        self.metadata['device'] = ElectricityMeter.meter_devices[device_model]

    def save(self, destination, key):
        """
        Convert all relevant attributes to a dict to be 
        saved as metadata in destination at location specified
        by key
        """
        # destination.write_metadata(key, self.metadata)
        raise NotImplementedError

    def __repr__(self):
        string = super(ElectricityMeter, self).__repr__()
        # Now add list of appliances...
        string = string[:-1] # remove last bracket
#        appliances = [a.type_and_instance_string() for a in self.appliances]
        string += ', appliances={})'.format(self.appliances)
        return string

    def matches(self, key):
        """
        Parameters
        ----------
        key : dict

        Returns
        -------
        True if all key:value pairs in `key` match any appliance
        in `self.appliances`.
        """
        for appliance in self.appliances:
            if appliance.matches(key):
                return True
        return False

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
        voltage_series : ElectricityMeter object with voltage measurements available.  If not supplied and if normalise is True
            then will attempt to use voltage data from this meter.
        nominal_voltage : float

        
        Returns
        -------
        generator of pd.Series of power measurements.
        """
        raise NotImplementedError
        
    def voltage_series(self):
        """Returns a generator of pd.Series of voltage, if available."""
        raise NotImplementedError
        
    def total_energy(self, **load_kwargs):
        """
        Returns
        -------
        nilmtk.pipeline.EnergyResults object
        """
        nodes = [ClipNode(), EnergyNode()]
        results = self._run_pipeline(nodes, **load_kwargs)
        return results['energy']
        
    def dropout_rate(self):
        """returns a DropoutRateResults object."""
        raise NotImplementedError
        
    def good_sections(self):
        """
        Returns
        -------
        sections: list of nilmtk.TimeFrame objects
        """
        nodes = [LocateGoodSectionsNode()]
        results = self._run_pipeline(nodes)
        return results['good_sections']
        
    def total_on_duration(self):
        """Return timedelta"""
        raise NotImplementedError
    
    def on_durations(self):
        raise NotImplementedError
    
    def activity_distribution(self, bin_size, timespan):
        raise NotImplementedError
    
    def when_on(self):
        """Return Series of bools"""
        raise NotImplementedError    

    def on_off_events(self):
        # use self.metadata.minimum_[off|on]_duration
        raise NotImplementedError
    
    def discrete_appliance_activations(self):
        """
        Return a Mask defining the start and end times of each appliance
        activation.
        """
        raise NotImplementedError
    
    def proportion_of_energy(self):
        # Mask out gaps from mains
        good_mains_timeframes = self.mains.good_timeframes()
        proportion_of_energy = (self.total_energy(timeframes=good_mains_timeframes) /
                                self. mains.total_energy())
        return proportion_of_energy 

    def contiguous_sections(self):
        """retuns Mask object"""
        raise NotImplementedError
        
    def clean_and_export(self, destination_datastore):
        """Apply all cleaning configured in meter.cleaning and then export.  Also identifies
        and records the locations of gaps.  Also records metadata about exactly which
        cleaning steps have been executed and some summary results (e.g. the number of
        implausible values removed)"""
        raise NotImplementedError
        
    def _run_pipeline(self, nodes, **load_kwargs): 
        if self.store is None:
            msg = ("'meter.loader' is not set!"
                   " Cannot process data without a loader!")
            raise RuntimeError(msg)
        pipeline = Pipeline(nodes)
        pipeline.run(meter=self, **load_kwargs)
        return pipeline.results        
