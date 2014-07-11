from __future__ import print_function, division
from warnings import warn
from collections import namedtuple
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
from .preprocessing import Clip
from .stats import TotalEnergy, GoodSections, DropoutRate
from .hashable import Hashable
from .appliance import Appliance
from .datastore import Key
from .measurement import select_best_ac_type
from .node import Node
from .electric import Electric
import nilmtk

ElecMeterID = namedtuple('ElecMeterID', ['instance', 'building', 'dataset'])

class ElecMeter(Hashable, Electric):
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
    """

    meter_devices = {}

    def __init__(self, store=None, metadata=None, meter_id=None):
        # Store and check parameters
        self.appliances = []
        self.metadata = {} if metadata is None else metadata
        assert isinstance(self.metadata, dict)
        self.store = store
        self.identifier = meter_id

        # Insert self into nilmtk.global_meter_group
        if self.identifier is not None:
            assert isinstance(self.identifier, ElecMeterID)
            if self not in nilmtk.global_meter_group.meters:
                nilmtk.global_meter_group.meters.append(self)

    @property
    def key(self):
        return self.metadata['data_location']

    def instance(self):
        return self._identifier_attr('instance')

    def building(self):
        return self._identifier_attr('building')

    def dataset(self):
        return self._identifier_attr('dataset')

    def _identifier_attr(self, attr):
        if self.identifier is None:
            return
        else:
            return getattr(self.identifier, attr)

    def upstream_meter(self):
        """
        Returns
        -------
        ElecMeterID of upstream meter or None if is site meter.
        """
        if self.is_site_meter():
            return

        submeter_of = self.metadata.get('submeter_of')

        # Sanity checks
        if submeter_of is None:
            raise ValueError("This meter has no 'submeter_of' metadata attribute.")
        if submeter_of < 0:
            raise ValueError("'submeter_of' must be >= 0.")
        upstream_meter_in_building = self.metadata.get('upstream_meter_in_building')
        if (upstream_meter_in_building is not None and
            upstream_meter_in_building != self.identifier.building):
            raise NotImplementedError(
                "'upstream_meter_in_building' not implemented yet.")

        id_of_upstream = ElecMeterID(instance=submeter_of, 
                                     building=self.identifier.building,
                                     dataset=self.identifier.dataset)

        return nilmtk.global_meter_group[id_of_upstream]

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
            return deepcopy(ElecMeter.meter_devices[device_model])
        else:
            return {}

    def sample_period(self):
        device = self.device
        if device:
            return device['sample_period']

    def is_site_meter(self):
        return self.metadata.get('site_meter', False)

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

    def available_power_ac_types(self):
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

        # Site meter
        if self.metadata.get('site_meter'):
            string += ', site_meter'

        # Appliances
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
        Bool
        """

        if not key:
            return True

        if not isinstance(key, dict):
            raise TypeError()

        match = True
        for k, v in key.iteritems():
            if hasattr(self.identifier, k):
                if getattr(self.identifier, k) != v:
                    match = False

            elif self.metadata.has_key(k):
                if self.metadata[k] != v:
                    match = False

            elif self.device.has_key(k):
                metadata_value = self.device[k]
                if (isinstance(metadata_value, list) and 
                    not isinstance(v, list)):
                    if v not in metadata_value:
                        match = False
                elif metadata_value != v:
                    match = False

            else:
                raise KeyError("'{}' not a valid key.".format(k))

        return match

    def power_series(self, **kwargs):
        """Get power Series.
        
        Parameters
        ----------
        measurement_ac_type_prefs : list of strings, optional
            if provided then will try to select the best AC type from 
            self.available_ac_types which is also in measurement_ac_type_prefs.
            If none of the measurements from measurement_ac_type_prefs are 
            available then will raise a warning and will select another ac type.
        preprocessing : list of Node subclass instances
        **kwargs :
            Any other key word arguments are passed to self.store.load()
        
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
        measurement_ac_type_prefs = kwargs.pop('measurement_ac_type_prefs', None)
        preprocessing = kwargs.pop('preprocessing', [])

        # Select power column:
        if not kwargs.has_key('cols'):
            best_ac_type = select_best_ac_type(self.available_power_ac_types(),
                                               measurement_ac_type_prefs)
            kwargs['cols'] = [('power', best_ac_type)]

        # Get source node
        last_node = self.get_source_node(**kwargs)
        generator = last_node.generator

        # Connect together all preprocessing nodes
        for node in preprocessing:
            node.upstream = last_node
            last_node = node
            generator = last_node.process()

        # Pull data through preprocessing pipeline
        for chunk in generator:
            series = chunk.icol(0).dropna()
            series.timeframe = getattr(chunk, 'timeframe', None)
            series.look_ahead = getattr(chunk, 'look_ahead', None)
            yield series

    def voltage_series(self):
        """Returns a generator of pd.Series of voltage, if available."""
        raise NotImplementedError

    def dry_run_metadata(self):
        return self.metadata

    def get_metadata(self):
        return self.metadata

    def get_source_node(self, **loader_kwargs):
        if self.store is None:
            raise RuntimeError("Cannot get source node if meter.store is None!")
        generator = self.store.load(key=self.key, **loader_kwargs)
        self.metadata['device'] = self.device
        return Node(self, generator=generator)
        
    def total_energy(self, **loader_kwargs):
        """
        Parameters
        ----------
        full_results : bool, default=False
        **loader_kwargs : key word arguments for DataStore.load()

        Returns
        -------
        if `full_results` is True then return TotalEnergyResults object
        else return either a single number of, if there are multiple
        AC types, then return a pd.Series with a row for each AC type.
        """
        full_results = loader_kwargs.pop('full_results', False)
        source_node = self.get_source_node(**loader_kwargs)
        clipped = Clip(source_node)
        total_energy = TotalEnergy(clipped)
        total_energy.run()
        if full_results: 
            return total_energy.results
        else:
            return total_energy.results.simple()
        
    def dropout_rate(self, **loader_kwargs):
        """
        Parameters
        ----------
        full_results : bool, default=False
        **loader_kwargs : key word arguments for DataStore.load()

        Returns
        -------
        DropoutRateResults object if `full_results` is True, 
        else float
        """
        full_results = loader_kwargs.pop('full_results', False)
        source_node = self.get_source_node(**loader_kwargs)
        dropout_rate = DropoutRate(source_node)
        dropout_rate.run()
        if full_results:
            return dropout_rate.results
        else:
            return dropout_rate.results.simple()
        
    def good_sections(self, **loader_kwargs):
        """
        Parameters
        ----------
        full_results : bool, default=False
        **loader_kwargs : key word arguments for DataStore.load()

        Returns
        -------
        if `full_results` is True then return nilmtk.stats.GoodSectionsResults 
        object otherwise return list of TimeFrame objects.
        """
        full_results = loader_kwargs.pop('full_results', False)
        source_node = self.get_source_node(**loader_kwargs)
        good_sections = GoodSections(source_node)
        good_sections.run()
        if full_results:
            return good_sections.results
        else:
            return good_sections.results.simple()

    def total_on_duration(self):
        """Return timedelta"""
        raise NotImplementedError
    
    def on_durations(self):
        raise NotImplementedError
    
    def activity_distribution(self, bin_size, timespan):
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
        """
        Parameters
        ----------
        mains : nilmtk.ElecMeter or MeterGroup
        """
        mains_good_sects = mains.good_sections()
        proportion_of_energy = (self.total_energy(timeframes=mains_good_sects) /
                                mains.total_energy(timeframes=mains_good_sects))
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
