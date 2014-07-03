from __future__ import print_function, division
from .pipeline import Pipeline, Clip, TotalEnergy, GoodSections
from .hashable import Hashable
from .appliance import Appliance
from .datastore import Key
from .measurement import select_best_ac_type
from warnings import warn
from collections import namedtuple
from copy import deepcopy

ElecMeterID = namedtuple('ElecMeterID', ['instance', 'building', 'dataset'])

class ElecMeter(Hashable):
    """Represents a physical electricity meter.
    
    Attributes
    ----------
    appliances : list of Appliance objects connected immediately downstream
      of this meter.  Will be [] if no appliances are connected directly
      to this meter.

    store : nilmtk.DataStore

    key : string
        key into nilmtk.DataStore to access data.
    
    metadata : dict.
        See http://nilm-metadata.readthedocs.org/en/latest/dataset_metadata.html#elecmeter    

    STATIC ATTRIBUTES
    -----------------

    meter_devices : dict, static class attribute
        See http://nilm-metadata.readthedocs.org/en/latest/dataset_metadata.html#meterdevice

    meters : dict, static class attribute:
        Required for resolving `upstream_of` to an ElecMeter object.
        Keys are ElecMeterID objects.
        Values are ElecMeter objects.

    """

    meter_devices = {}
    meters = {}

    def __init__(self, store=None, metadata=None, meter_id=None):
        # Store and check parameters
        self.metadata = {} if metadata is None else metadata
        assert isinstance(self.metadata, dict)
        self.store = store
        if self.store is not None:
            assert not isinstance(self.store, dict)
        self.identifier = meter_id
        if self.identifier is not None:
            assert isinstance(meter_id, ElecMeterID)
            ElecMeter.meters[self.identifier] = self
        self.appliances = []

    @property
    def key(self):
        return self.metadata['data_location']

    @property
    def upstream_meter(self):
        submeter_of = self.metadata.get('submeter_of')
        if submeter_of is None:
            raise ValueError("This meter has no 'submeter_of' metadata attribute.")
        if submeter_of <= 0:
            raise ValueError("'submeter_of' must be >= 1.")
        upstream_meter_in_building = self.metadata.get('upstream_meter_in_building')
        if upstream_meter_in_building is None:
            upstream_meter_in_building = self.identifier.building
        id_of_upstream = ElecMeterID(instance=submeter_of, 
                                            building=upstream_meter_in_building,
                                            dataset=self.identifier.dataset)
        upstream_meter =  ElecMeter.meters[id_of_upstream]
        return upstream_meter

    @classmethod
    def load_meter_devices(cls, store):
        dataset_metadata = store.load_metadata('/')
        ElecMeter.meter_devices.update(dataset_metadata.get('meter_devices', {}))

    def save(self, destination, key):
        """
        Convert all relevant attributes to a dict to be 
        saved as metadata in destination at location specified
        by key
        """
        # destination.write_metadata(key, self.metadata)
        # then save data
        raise NotImplementedError

    @property
    def device(self):
        """
        Returns
        -------
        dict describing the MeterDevice for this meter (sample period etc).
        """
        device_model = self.metadata.get('device_model')
        if device_model:
            return ElecMeter.meter_devices[device_model]
        else:
            return {}

    def dominant_appliance(self):
        """Tries to find the most dominant appliance on this meter,
        and then returns that appliance object.  Will return None
        if there are no appliances on this meter.
        """
        n_appliances = len(self.appliances)
        if n_appliances == 0:
            return
        elif n_appliances == 1:
            return self.appliances[0]
        else:
            for app in self.appliances:
                if app.metadata.get('dominant_appliance'):
                    return app
            warn('Multiple appliances are associated with meter {}'
                 ' but none are marked as the dominant appliance. Hence'
                 ' returning the first appliance in the list.', RuntimeWarning)
            return self.appliances[0]

    def appliance_label(self):
        """
        Returns
        -------
        string : A label listing all the appliance types.
        """
        appliance_names = []
        dominant = self.dominant_appliance()
        for appliance in self.appliances:
            appliance_name = appliance.label()
            if appliance.metadata.get('dominant_appliance'):
                appliance_name = appliance_name.upper()
            appliance_names.append(appliance_name)
        label = ", ".join(appliance_names) 
        return label

    def available_ac_types(self):
        """Finds available alternating current types from power measurements.

        Returns
        -------
        list of strings e.g. ['apparent', 'active']
        """
        measurements = self.device['measurements']
        return [m['type'] for m in measurements 
                if m['physical_quantity'] == 'power']

    def __repr__(self):
        string = super(ElecMeter, self).__repr__()
        # Now add list of appliances...
        string = string[:-1] # remove last bracket
        string += ', appliances={}'.format(self.appliances)
        
        # METER ROOM
        room = self.metadata.get('room')
        if room:
            string += ', room={}'.format(room['name'])
            try:
                string += '{:d}'.format(room['instance'])
            except KeyError:
                pass

        string += ')'
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

    def power_series(self, measurement_ac_type_prefs=None, **load_kwargs):
        """Get power Series.
        
        Parameters
        ----------
        measurement_ac_type_prefs : list of strings, optional
            if provided then will try to select the best AC type from 
            self.available_ac_types which is also in measurement_ac_type_prefs.
            If none of the measurements from measurement_ac_type_prefs are 
            available then will raise a warning and will select another ac type.

        Returns
        -------
        generator of pd.Series of power measurements.

        TODO
        -----
        The following cleaning steps will be run if the relevant entries
        in meter.cleaning are True:

        * remove implausable values
        * gaps will be bookended with zeros

        required_measurements : Measurement, optional.  
            Raises MeasurementError if not available.
        normalise : boolean, optional, defaults to False
        voltage_series : ElecMeter object with voltage measurements available.
            If not supplied and if normalise is True
            then will attempt to use voltage data from this meter.
        nominal_voltage : float
        """
        best_ac_type = select_best_ac_type(self.available_ac_types(),
                                           measurement_ac_type_prefs)
        return self.store.load(key=self.key, cols=[best_ac_type], **load_kwargs)

    def voltage_series(self):
        """Returns a generator of pd.Series of voltage, if available."""
        raise NotImplementedError
        
    def total_energy(self, **load_kwargs):
        """
        Returns
        -------
        nilmtk.pipeline.EnergyResults object
        """
        nodes = [Clip(), TotalEnergy()]
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
        nodes = [GoodSections()]
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
    
    def proportion_of_energy(self, mains):
        # Mask out gaps from mains
        good_mains_timeframes = mains.good_timeframes()
        proportion_of_energy = (self.total_energy(timeframes=good_mains_timeframes) /
                                mains.total_energy(timeframes=good_mains_timeframes))
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
            msg = ("'meter.store' is not set!"
                   " Cannot process data without a DataStore!")
            raise RuntimeError(msg)
        pipeline = Pipeline(nodes)
        pipeline.run(meter=self, **load_kwargs)
        return pipeline.results        
