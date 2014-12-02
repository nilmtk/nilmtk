from __future__ import print_function, division
from warnings import warn
from collections import namedtuple
from compiler.ast import flatten
from copy import deepcopy
from itertools import izip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial as ss
from scipy import fft
from pandas.tools.plotting import lag_plot, autocorrelation_plot
from scipy.special import digamma,gamma
from math import log,pi
import numpy.random as nr
import random
from .preprocessing import Clip
from .stats import TotalEnergy, GoodSections, DropoutRate
from .stats.totalenergyresults import TotalEnergyResults
from .hashable import Hashable
from .appliance import Appliance
from .datastore import Key
from .measurement import select_best_ac_type
from .node import Node
from .electric import Electric
from .timeframe import TimeFrame, list_of_timeframe_dicts
import nilmtk

MAX_SIZE_ENTROPY = 10000
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

    def get_timeframe(self):
        self._check_store()
        return self.store.get_timeframe(key=self.key)

    def _check_store(self):
        if self.store is None:
            raise RuntimeError("ElecMeter needs `store` attribute set to an"
                               " instance of a `nilmtk.DataStore` subclass")

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
            raise ValueError(
                "This meter has no 'submeter_of' metadata attribute.")
        if submeter_of < 0:
            raise ValueError("'submeter_of' must be >= 0.")
        upstream_meter_in_building = self.metadata.get(
            'upstream_meter_in_building')
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
        ElecMeter.meter_devices.update(
            dataset_metadata.get('meter_devices', {}))

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
        if self.is_site_meter():
            appliance_names.append('SITE METER')
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
        string = string[:-1]  # remove last bracket

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

    def power_series_all_columns(self, **kwargs):
        """Get all power parameters available"""

        preprocessing = kwargs.pop('preprocessing', [])

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
            series = chunk['power'].fillna(0)
            series.timeframe = getattr(chunk, 'timeframe', None)
            series.look_ahead = getattr(chunk, 'look_ahead', None)
            yield series

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
        measurement_ac_type_prefs = kwargs.pop(
            'measurement_ac_type_prefs', None)
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

    def plot_lag(self, lag=1):
        """
        Plots a lag plot of power data
        http://www.itl.nist.gov/div898/handbook/eda/section3/lagplot.htm

        Returns
        -------
        matplotlib.axis
        """
        fig, ax = plt.subplots()
        for power in self.power_series():
            lag_plot(power, lag, ax = ax)
        return ax

    def plot_spectrum(self):
        """
        Plots spectral plot of power data
        http://www.itl.nist.gov/div898/handbook/eda/section3/spectrum.htm

        Code borrowed from:
        http://glowingpython.blogspot.com/2011/08/how-to-plot-frequency-spectrum-with.html

        Returns
        -------
        matplotlib.axis
        """ 
        fig, ax = plt.subplots()
        Fs = 1.0/self.device.get('sample_period')
        for power in self.power_series():
            n = len(power.values) # length of the signal
            k = np.arange(n)            
            T = n/Fs
            frq = k/T # two sides frequency range
            frq = frq[range(n//2)] # one side frequency range

            Y = fft(power)/n # fft computing and normalization
            Y = Y[range(n//2)]

            ax.plot(frq,abs(Y)) # plotting the spectrum

        ax.set_xlabel('Freq (Hz)')
        ax.set_ylabel('|Y(freq)|')
        return ax
          

    def plot_autocorrelation(self):
        """
        Plots autocorrelation of power data 
        Reference: 
        http://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm

        Returns
        -------
        matplotlib.axis 
        """
        fig, ax = plt.subplots()
        for power in self.power_series():
            autocorrelation_plot(power, ax = ax)
        return ax


    def switch_times(self, threshold=40):
        """
        Returns an array of pd.DateTime when a switch occurs as defined by threshold

        Parameters
        ----------
        threshold: int, threshold in Watts between succcessive readings 
        to amount for an appliance state change
        """

        datetime_switches = []
        for power in self.power_series():
            delta_power = power.diff()
            delta_power_absolute = delta_power.abs()
            datetime_switches.append(delta_power_absolute[(delta_power_absolute>threshold)].index.values.tolist())
        return flatten(datetime_switches)

    def entropy(self, k=3, base=2):
        """ 
        This implementation is provided courtesy NPEET toolbox,
        the authors kindly allowed us to directly use their code.
        As a courtesy procedure, you may wish to cite their paper, 
        in case you use this function.
        This fails if there is a large number of records. Need to
        ask the authors what to do about the same! 
        The classic K-L k-nearest neighbor continuous entropy estimator
        x should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
        if x is a one-dimensional scalar and we have four samples
        """
        def kdtree_entropy(z):
            assert k <= len(z)-1, "Set k smaller than num. samples - 1"
            d = len(z[0])
            N = len(z)
            intens = 1e-10 #small noise to break degeneracy, see doc.
            z = [list(p + intens*nr.rand(len(z[0]))) for p in z]
            tree = ss.cKDTree(z)
            nn = [tree.query(point,k+1,p=float('inf'))[0][k] for point in z]
            const = digamma(N)-digamma(k) + d*log(2)
            return (const + d*np.mean(map(log,nn)))/log(base)

        out = []
        for power in self.power_series():
            x = power.values
            num_elements = len(x)
            x = x.reshape((num_elements, 1))            
            if num_elements>MAX_SIZE_ENTROPY:

                splits = num_elements/MAX_SIZE_ENTROPY + 1
                y = np.array_split(x, splits)
                for z in y:            
                    out.append(kdtree_entropy(z))                    
            else:
                out.append(kdtree_entropy(x))
        return sum(out)/len(out)

    def mutual_information(self, other, k=3, base=2):
        """ 
        Mutual information of two ElecMeters
        x,y should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
        if x is a one-dimensional scalar and we have four samples

        Parameters
        ----------
        other : ElecMeter or MeterGroup
        """
        def kdtree_mi(x, y, k, base):
            intens = 1e-10 #small noise to break degeneracy, see doc.
            x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
            y = [list(p + intens*nr.rand(len(y[0]))) for p in y]
            points = zip2(x,y)
            #Find nearest neighbors in joint space, p=inf means max-norm
            tree = ss.cKDTree(points)
            dvec = [tree.query(point,k+1,p=float('inf'))[0][k] for point in points]
            a,b,c,d = avgdigamma(x,dvec), avgdigamma(y,dvec), digamma(k), digamma(len(x)) 
            return (-a-b+c+d)/log(base)
            
        def zip2(*args):
            #zip2(x,y) takes the lists of vectors and makes it a list of vectors in a joint space
            #E.g. zip2([[1],[2],[3]],[[4],[5],[6]]) = [[1,4],[2,5],[3,6]]
            return [sum(sublist,[]) for sublist in zip(*args)]

        def avgdigamma(points,dvec):
            #This part finds number of neighbors in some radius in the marginal space
            #returns expectation value of <psi(nx)>
            N = len(points)
            tree = ss.cKDTree(points)
            avg = 0.
            for i in range(N):
                dist = dvec[i]
                #subtlety, we don't include the boundary point, 
                #but we are implicitly adding 1 to kraskov def bc center point is included
                num_points = len(tree.query_ball_point(points[i],dist-1e-15,p=float('inf'))) 
                avg += digamma(num_points)/N
            return avg

        out = []
        for power_x, power_y in izip(self.power_series(), other.power_series()):
            power_x_val = power_x.values
            power_y_val = power_y.values 
            num_elements = len(power_x_val)
            power_x_val = power_x_val.reshape((num_elements, 1))
            power_y_val = power_y_val.reshape((num_elements, 1))            
            if num_elements>MAX_SIZE_ENTROPY:
                splits = num_elements/MAX_SIZE_ENTROPY + 1
                x_split = np.array_split(power_x_val, splits)
                y_split = np.array_split(power_y_val, splits)
                for x, y  in izip(x_split, y_split):            
                    out.append(kdtree_mi(x, y, k, base))                    
            else:
                out.append(kdtree_mi(power_x_val, power_y_val, k, base))
        return sum(out)/len(out)
        

    def dry_run_metadata(self):
        return self.metadata

    def get_metadata(self):
        return self.metadata

    def get_source_node(self, **loader_kwargs):
        if self.store is None:
            raise RuntimeError(
                "Cannot get source node if meter.store is None!")
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
        else returns a pd.Series with a row for each AC type.
        """
        nodes = [Clip, TotalEnergy]
        return self._get_stat_from_cache_or_compute(
            nodes, TotalEnergy.results_class(), loader_kwargs)

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
        nodes = [DropoutRate]
        return self._get_stat_from_cache_or_compute(
            nodes, DropoutRate.results_class(), loader_kwargs)

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
        loader_kwargs.setdefault('n_look_ahead_rows', 10)
        nodes = [GoodSections]
        results_obj = GoodSections.results_class(self.device['max_sample_period'])
        return self._get_stat_from_cache_or_compute(
            nodes, results_obj, loader_kwargs)        

    def _get_stat_from_cache_or_compute(self, nodes, results_obj, loader_kwargs):
        """General function for computing statistics and/or loading them from 
        cache.

        Cached statistics lives in the DataStore at 
        'building<I>/elec/cache/meter<K>/<statistic_name>' e.g.
        'building1/elec/cache/meter1/total_energy'.  We store the 
        'full' statistic... i.e we store a representation of the `Results._data`
        DataFrame. Some times we need to do some conversion to store 
        `Results._data` on disk.  The logic for doing this conversion lives
        in the `Results` class or subclass.  The cache can be cleared by calling
        `ElecMeter.clear_cache()`.

        Parameters
        ----------
        nodes : list of nilmtk.Node classes
        results_obj : instance of nilmtk.Results subclass
        loader_kwargs : dict

        Returns
        -------
        if `full_results` is True then return nilmtk.Results subclass
        instance otherwise return nilmtk.Results.simple().

        See Also
        --------
        clear_cache
        _compute_stat
        key_for_cached_stat
        get_cached_stat
        """
        full_results = loader_kwargs.pop('full_results', False)

        # Prepare `sections` list
        sections = loader_kwargs.get('sections')
        if sections is None:
            tf = self.get_timeframe()
            tf.include_end = True
            sections = [tf]

        # Retrieve usable stats from cache
        key_for_cached_stat = self.key_for_cached_stat(results_obj.name)
        if loader_kwargs.get('preprocessing') is None:
            cached_stat = self.get_cached_stat(key_for_cached_stat)
            results_obj.import_from_cache(cached_stat, sections)
            
            # Get sections_to_compute
            results_obj_timeframes = results_obj.timeframes()
            sections_to_compute = set(sections) - set(results_obj_timeframes)
            sections_to_compute = list(sections_to_compute)
            sections_to_compute.sort()
        else:
            sections_to_compute = sections

        if not results_obj._data.empty:
            print("Using cached result.")

        # If we get to here then we have to compute some stats
        if sections_to_compute:
            loader_kwargs['sections'] = sections_to_compute
            computed_result = self._compute_stat(nodes, loader_kwargs)

            # Merge cached results with newly computed
            results_obj.update(computed_result.results)

            # Save to disk newly computed stats
            self.store.append(key_for_cached_stat,
                              computed_result.results.export_to_cache())

        return results_obj if full_results else results_obj.simple()

    def _compute_stat(self, nodes, loader_kwargs):
        """
        Parameters
        ----------
        nodes : list of nilmtk.Node subclass objects
        loader_kwargs : dict

        Returns
        -------
        Node subclass object

        See Also
        --------
        clear_cache
        _get_stat_from_cache_or_compute
        key_for_cached_stat
        get_cached_stat
        """
        results = self.get_source_node(**loader_kwargs)
        for node in nodes:
            results = node(results)
        results.run()
        return results

    def key_for_cached_stat(self, stat_name):
        """
        Parameters
        ----------
        stat_name : str

        Returns
        -------
        key : str

        See Also
        --------
        clear_cache
        _compute_stat
        _get_stat_from_cache_or_compute
        get_cached_stat
        """
        return ("building{:d}/elec/cache/meter{:d}/{:s}"
                .format(self.building(), self.instance(), stat_name))

    def clear_cache(self, verbose=False):
        """
        See Also
        --------
        _compute_stat
        _get_stat_from_cache_or_compute
        key_for_cached_stat        
        get_cached_stat
        """
        if self.store is not None:
            key_for_cache = self.key_for_cached_stat('')
            try:
                self.store.remove(key_for_cache)
            except KeyError:
                if verbose:
                    print("No existing cache for", key_for_cache)
            else:
                print("Removed", key_for_cache)

    def get_cached_stat(self, key_for_stat):
        """
        Parameters
        ----------
        key_for_stat : str

        Returns
        -------
        pd.DataFrame

        See Also
        --------
        _compute_stat
        _get_stat_from_cache_or_compute
        key_for_cached_stat        
        clear_cache
        """
        if self.store is None:
            return pd.DataFrame()
        try:
            stat_from_cache = self.store[key_for_stat]
        except KeyError:
            return pd.DataFrame()
        else:
            return stat_from_cache


    # def total_on_duration(self):
    #     """Return timedelta"""
    #     raise NotImplementedError

    # def on_durations(self):
    #     raise NotImplementedError

    # def activity_distribution(self, bin_size, timespan):
    #     raise NotImplementedError

    # def on_off_events(self):
    # use self.metadata.minimum_[off|on]_duration
    #     raise NotImplementedError

    # def discrete_appliance_activations(self):
    #     """
    #     Return a Mask defining the start and end times of each appliance
    #     activation.
    #     """
    #     raise NotImplementedError

    # def contiguous_sections(self):
    #     """retuns Mask object"""
    #     raise NotImplementedError

    # def clean_and_export(self, destination_datastore):
    #     """Apply all cleaning configured in meter.cleaning and then export.  Also identifies
    #     and records the locations of gaps.  Also records metadata about exactly which
    #     cleaning steps have been executed and some summary results (e.g. the number of
    #     implausible values removed)"""
    #     raise NotImplementedError
