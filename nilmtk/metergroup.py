from __future__ import print_function, division
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from warnings import warn
from sys import stdout
from collections import Counter
from copy import deepcopy
from .elecmeter import ElecMeter, ElecMeterID
from .appliance import Appliance
from .datastore.datastore import join_key
from .utils import (tree_root, nodes_adjacent_to_root, simplest_type_for,
                    flatten_2d_list, convert_to_timestamp)
from .plots import plot_series
from .measurement import (select_best_ac_type, AC_TYPES,
                          PHYSICAL_QUANTITIES_TO_AVERAGE)
from .exceptions import MeasurementError
from .electric import Electric
from .timeframe import TimeFrame
from .preprocessing import Apply

class MeterGroup(Electric):

    """A group of ElecMeter objects. Can contain nested MeterGroup objects.

    Implements many of the same methods as ElecMeter.

    Attributes
    ----------
    meters : list of ElecMeters or nested MeterGroups
    disabled_meters : list of ElecMeters or nested MeterGroups
    """

    def __init__(self, meters=None, disabled_meters=None):
        def _convert_to_list(list_like):
            return [] if list_like is None else list(list_like)
        self.meters = _convert_to_list(meters)
        self.disabled_meters = _convert_to_list(disabled_meters)

    def import_metadata(self, store, elec_meters, appliances, building_id):
        """
        Parameters
        ----------
        store : nilmtk.DataStore
        elec_meters : dict of dicts
            metadata for each ElecMeter
        appliances : list of dicts
            metadata for each Appliance
        building_id : BuildingID
        """
        # Sanity checking
        assert isinstance(elec_meters, dict)
        assert isinstance(appliances, list)
        assert isinstance(building_id, tuple)
        if not elec_meters:
            warn("Building {} has an empty 'elec_meters' object."
                 .format(building_id.instance), RuntimeWarning)
        if not appliances:
            warn("Building {} has an empty 'appliances' list."
                 .format(building_id.instance), RuntimeWarning)

        # Load static Meter Devices
        ElecMeter.load_meter_devices(store)

        # Load each meter
        for meter_i, meter_metadata_dict in elec_meters.iteritems():
            meter_id = ElecMeterID(instance=meter_i,
                                   building=building_id.instance,
                                   dataset=building_id.dataset)
            meter = ElecMeter(store, meter_metadata_dict, meter_id)
            self.meters.append(meter)

        # Load each appliance
        for appliance_md in appliances:
            appliance_md['dataset'] = building_id.dataset
            appliance_md['building'] = building_id.instance
            appliance = Appliance(appliance_md)
            meter_ids = [ElecMeterID(instance=meter_instance,
                                     building=building_id.instance,
                                     dataset=building_id.dataset)
                         for meter_instance in appliance.metadata['meters']]

            if appliance.n_meters == 1:
                # Attach this appliance to just a single meter
                meter = self[meter_ids[0]]
                if isinstance(meter, MeterGroup): # MeterGroup of site_meters
                    metergroup = meter
                    for meter in metergroup.meters:
                        meter.appliances.append(appliance)
                else:
                    meter.appliances.append(appliance)
            else:
                # DualSupply or 3-phase appliance so need a meter group
                metergroup = MeterGroup()
                metergroup.meters = [self[meter_id] for meter_id in meter_ids]
                for meter in metergroup.meters:
                    # We assume that any meters used for measuring
                    # dual-supply or 3-phase appliances are not also used
                    # for measuring single-supply appliances.
                    self.meters.remove(meter)
                    meter.appliances.append(appliance)
                self.meters.append(metergroup)

        # disable disabled meters
        meters_to_disable = [m for m in self.meters 
                             if isinstance(m, ElecMeter) 
                             and m.metadata.get('disabled')]
        for meter in meters_to_disable:
            self.meters.remove(meter)
            self.disabled_meters.append(meter)

    def union(self, other):
        """
        Returns
        -------
        new MeterGroup where its set of `meters` is the union of
        `self.meters` and `other.meters`.
        """
        if not isinstance(other, MeterGroup):
            raise TypeError()
        return MeterGroup(set(self.meters).union(other.meters))

    def dominant_appliance(self):
        dominant_appliances = [meter.dominant_appliance()
                               for meter in self.meters]
        n_dominant_appliances = len(set(dominant_appliances))
        if n_dominant_appliances == 0:
            return
        elif n_dominant_appliances == 1:
            return dominant_appliances[0]
        else:
            raise RuntimeError(
                "More than one dominant appliance in MeterGroup!"
                " (The dominant appliance per meter should be manually"
                " specified in the metadata. If it isn't and if there are"
                " multiple appliances for a meter then NILMTK assumes"
                " all appliances on that meter are dominant. NILMTK"
                " can't automatically distinguish between multiple"
                " appliances on the same meter (at least,"
                " not without using NILM!))")

    def nested_metergroups(self):
        return [m for m in self.meters if isinstance(m, MeterGroup)]

    def __getitem__(self, key):
        """Get a single meter using appliance type and instance unless
        ElecMeterID is supplied.

        These formats for `key` are accepted:

        Retrieve a meter using details of the meter:
        * `1` - retrieves meter instance 1, raises Exception if there are 
                more than one meter with this instance, raises KeyError
                if none are found.  If meter instance 1 is in a nested MeterGroup
                then retrieve the ElecMeter, not the MeterGroup.
        * `ElecMeterID(1, 1, 'REDD')` - retrieves meter with specified meter ID
        * `[ElecMeterID(1, 1, 'REDD')], [ElecMeterID(2, 1, 'REDD')]` - retrieves
          existing nested MeterGroup containing exactly meter instances 1 and 2.
        * `ElecMeterID(0, 1, 'REDD')` - instance `0` means `mains`. This returns
           a new MeterGroup of all site_meters in building 1 in REDD.
        * `ElecMeterID(1, 1, 'REDD')` - retrieves meter with specified meter ID
        * `ElecMeterID((1,2), 1, 'REDD')` - retrieve existing MeterGroup 
           which contains exactly meters 1 & 2.
        * `(1, 2, 'REDD')` - converts to ElecMeterID and treats as an ElecMeterID.
           Items must be in the order expected for an ElecMeterID.

        Retrieve a meter using details of appliances attached to the meter:
        * `'toaster'`    - retrieves meter or group upstream of toaster instance 1
        * `'toaster', 2` - retrieves meter or group upstream of toaster instance 2
        * `{'dataset': 'redd', 'building': 3, 'type': 'toaster', 'instance': 2}`
          - specify an appliance

        Returns
        -------
        ElecMeter or MeterGroup
        """
        if isinstance(key, str):
            # default to get first meter
            return self[(key, 1)]
        elif isinstance(key, ElecMeterID):
            if isinstance(key.instance, tuple):
                # find meter group from a key of the form
                # ElecMeterID(instance=(1,2), building=1, dataset='REDD')
                for group in self.nested_metergroups():
                    if (set(group.instance()) == set(key.instance) and
                            group.building() == key.building and
                            group.dataset() == key.dataset):
                        return group
                # Else try to find an ElecMeter with instance=(1,2)
                for meter in self.meters:
                    if meter.identifier == key:
                        return meter
            elif key.instance == 0:
                metergroup_of_building = self.select(
                    building=key.building, dataset=key.dataset)
                return metergroup_of_building.mains()
            else:
                for meter in self.meters:
                    if meter.identifier == key:
                        return meter
            raise KeyError(key)
        # find MeterGroup from list of ElecMeterIDs
        elif isinstance(key, list):
            if not all([isinstance(item, tuple) for item in key]):
                raise TypeError("requires a list of ElecMeterID objects.")
            for meter in self.meters:  # TODO: write unit tests for this
                # list of ElecMeterIDs.  Return existing MeterGroup
                if isinstance(meter, MeterGroup):
                    metergroup = meter
                    meter_ids = set(metergroup.identifier)
                    if meter_ids == set(key):
                        return metergroup
            raise KeyError(key)
        elif isinstance(key, tuple):
            if len(key) == 2:
                if isinstance(key[0], str):
                    return self[{'type': key[0], 'instance': key[1]}]
                else:
                    # Assume we're dealing with a request for 2 ElecMeters
                    return MeterGroup([self[i] for i in key])
            elif len(key) == 3:
                return self[ElecMeterID(*key)]
            else:
                raise TypeError()
        elif isinstance(key, dict):
            meters = []
            for meter in self.meters:
                if meter.matches_appliances(key):
                    meters.append(meter)
            if len(meters) == 1:
                return meters[0]
            elif len(meters) > 1:
                raise Exception('search terms match {} appliances'
                                .format(len(meters)))
            else:
                raise KeyError(key)
        elif isinstance(key, int) and not isinstance(key, bool):
            meters_found = []
            for meter in self.meters:
                if isinstance(meter.instance(), int):
                    if meter.instance() == key:
                        meters_found.append(meter)
                elif isinstance(meter.instance(), (tuple, list)):
                    if key in meter.instance():
                        if isinstance(meter, MeterGroup):
                            print("Meter", key, "is in a nested meter group."
                                  " Retrieving just the ElecMeter.")
                            meters_found.append(meter[key])
                        else:
                            meters_found.append(meter)
            n_meters_found = len(meters_found)
            if n_meters_found > 1:
                raise Exception('{} meters found with instance == {}: {}'
                                .format(n_meters_found, key, meters_found))
            elif n_meters_found == 0:
                raise KeyError(
                    'No meters found with instance == {}'.format(key))
            else:
                return meters_found[0]
        else:
            raise TypeError()

    def matches(self, key):
        for meter in self.meters:
            if meter.matches(key):
                return True
        return False

    def select(self, **kwargs):
        """Select a group of meters based on meter metadata.

        e.g. 
        * select(building=1, sample_period=6)
        * select(room='bathroom')

        If multiple criteria are supplied then these are ANDed together.

        Returns
        -------
        new MeterGroup of selected meters.

        Ideas for the future (not implemented yet!)
        -------------------------------------------

        * select(category=['ict', 'lighting'])
        * select([(fridge, 1), (tv, 1)]) # get specifically fridge 1 and tv 1
        * select(name=['fridge', 'tv']) # get all fridges and tvs
        * select(category='lighting', except={'room'=['kitchen lights']})
        * select('all', except=[('tv', 1)])

        Also: see if we can do select(category='lighting' | name='tree lights')
        or select(energy > 100)??  Perhaps using:
        * Python's eval function something like this:
          >>> s = pd.Series(np.random.randn(5))
          >>> eval('(x > 0) | (index > 2)', {'x':s, 'index':s.index})
          Hmm, yes, maybe we should just implement this!  e.g.
          select("(category == 'lighting') | (category == 'ict')")

          But what about:
          * select('total_energy > 100')
          * select('mean(hours_on_per_day) > 3')
          * select('max(hours_on_per_day) > 5')
          * select('max(power) > 2000')
          * select('energy_per_day > 2')
          * select('rank_by_energy > 5') # top_k(5)
          * select('rank_by_proportion > 0.2')
          Maybe don't bother.  That's easy enough
          to get with itemised_energy().  Although these are quite nice
          and shouldn't be too hard.  Would need to only calculate
          these stats if necessary though (e.g. by checking if 'total_energy'
          is in the query string before running `eval`)

        * or numexpr: https://github.com/pydata/numexpr
        * see Pandas.eval(): 
          * http://pandas.pydata.org/pandas-docs/stable/indexing.html#the-query-method-experimental
          * https://github.com/pydata/pandas/blob/master/pandas/computation/eval.py#L119
        """
        selected_meters = []
        exception_raised_every_time = True
        exception = None
        func = kwargs.pop('func', 'matches')
        for meter in self.meters:
            try:
                match = getattr(meter, func)(kwargs)
            except KeyError as e:
                exception = e
            else:
                exception_raised_every_time = False
                if match:
                    selected_meters.append(meter)

        if exception_raised_every_time and exception is not None:
            raise exception

        return MeterGroup(selected_meters)

    def select_using_appliances(self, **kwargs):
        """Select a group of meters based on appliance metadata.

        e.g. 
        * select(category='lighting')
        * select(type='fridge')
        * select(building=1, category='lighting')
        * select(room='bathroom')

        If multiple criteria are supplied then these are ANDed together.

        Returns
        -------
        new MeterGroup of selected meters.
        """
        return self.select(func='matches_appliances', **kwargs)

    def from_list(self, meter_ids):
        """
        Parameters
        ----------
        meter_ids : list or tuple
            Each element is an ElecMeterID or 
            a tuple of ElecMeterIDs (to make a nested MeterGroup)

        Returns
        -------
        MeterGroup
        """
        assert isinstance(meter_ids, (tuple, list))
        meter_ids = list(set(meter_ids))  # make unique
        meters = []
        for meter_id in meter_ids:
            if isinstance(meter_id, ElecMeterID):
                meters.append(self[meter_id])
            elif isinstance(meter_id, tuple):
                metergroup = self.from_list(meter_id)
                meters.append(metergroup)
            else:
                raise TypeError()
        return MeterGroup(meters)

    @classmethod
    def from_other_metergroup(cls, other, dataset):
        """Assemble a new meter group using the same meter IDs and nested 
        MeterGroups as `other`.  This is useful for preparing a ground truth
        metergroup from a meter group of NILM predictions.

        Parameters
        ----------
        other : MeterGroup
        dataset : string
            The `name` of the dataset for the ground truth.  e.g. 'REDD'

        Returns
        -------
        MeterGroup
        """
        other_identifiers = other.identifier
        new_identifiers = []
        for other_id in other_identifiers:
            new_id = other_id._replace(dataset=dataset)
            if isinstance(new_id.instance, tuple):
                nested = []
                for instance in new_id.instance:
                    new_nested_id = new_id._replace(instance=instance)
                    nested.append(new_nested_id)
                new_identifiers.append(tuple(nested))
            else:
                new_identifiers.append(new_id)
        return MeterGroup.from_list(new_identifiers)

    def __eq__(self, other):
        if isinstance(other, MeterGroup):
            return set(other.meters) == set(self.meters)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def appliances(self):
        appliances = set()
        for meter in self.meters:
            appliances.update(meter.appliances)
        return list(appliances)

    def get_appliance_labels(self, meter_ids):
        """Create human-readable appliance labels.

        Parameters
        ----------
        meter_ids : list of ElecMeterIDs (or 3-tuples in same order as ElecMeterID)

        Returns
        -------
        list of strings describing the appliances.
        """
        meters = [self[meter_id] for meter_id in meter_ids]
        labels = [meter.appliance_label() for meter in meters]
        return labels

    def __repr__(self):
        s = "{:s}(meters=\n".format(self.__class__.__name__)
        for meter in self.meters:
            s += "  " + str(meter).replace("\n", "\n  ") + "\n"
        s += ")"
        return s

    @property
    def identifier(self):
        """Returns tuple of ElecMeterIDs or nested tuples of ElecMeterIDs"""
        return tuple([meter.identifier for meter in self.meters])

    def instance(self):
        """Returns tuple of integers where each int is a meter instance."""
        return tuple([meter.instance() for meter in self.meters])

    def building(self):
        """Returns building instance integer(s)."""
        buildings = set([meter.building() for meter in self.meters])
        return simplest_type_for(buildings)

    def contains_meters_from_multiple_buildings(self):
        """Returns True if this MeterGroup contains meters from 
        more than one building."""
        building = self.building()
        try:
            n = len(building)
        except TypeError:
            return False
        else:
            return n > 1

    def dataset(self):
        """Returns dataset string(s)."""
        datasets = set([meter.dataset() for meter in self.meters])
        return simplest_type_for(datasets)

    def sample_period(self):
        """Returns max of all meter sample periods."""
        return max([meter.sample_period() for meter in self.meters])

    def wiring_graph(self):
        """Returns a networkx.DiGraph of connections between meters."""
        wiring_graph = nx.DiGraph()

        def _build_wiring_graph(meters):
            for meter in meters:
                if isinstance(meter, MeterGroup):
                    metergroup = meter
                    _build_wiring_graph(metergroup.meters)
                else:
                    upstream_meter = meter.upstream_meter()
                    # Need to ensure we use the same object
                    # if upstream meter already exists.
                    if upstream_meter is not None:
                        for node in wiring_graph.nodes():
                            if upstream_meter == node:
                                upstream_meter = node
                                break
                        wiring_graph.add_edge(upstream_meter, meter)
        _build_wiring_graph(self.meters)
        return wiring_graph

    def draw_wiring_graph(self):
        graph = self.wiring_graph()
        labels = {}
        for meter in graph.nodes():
            if isinstance(meter, ElecMeter):
                labels[meter] = meter.identifier.instance
            else:
                metergroup = meter
                meter_instances = [
                    m.identifier.instance for m in metergroup.meters]
                labels[meter] = meter_instances
        nx.draw(graph, labels=labels)

    def load(self, sample_period=None, **kwargs):
        """Returns a generator of DataFrames loaded from the DataStore.

        By default, `load` will load all available columns from the DataStore.  
        Specific columns can be selected in one or two mutually exclusive ways:

        1. specify a list of column names using the `cols` parameter.
        2. specify a `physical_quantity` and/or an `ac_type` parameter to ask 
           `load` to automatically select columns.

        Each meter in the MeterGroup will first be reindexed before being added.

        Note that we just use forward filling to reindex.

        Also note that the timeframe will be the *union* of the timeframes of
        all individual meters.

        Also note that `chunksize` refers to the chunksize for each individual
        meter.  If those chunks span a large timeframe then the timeframe of
        the returned data will be large.

        Parameters
        ----------
        sample_period : number, seconds, optional
            The sample_period to reindex all meters to.  If not specified then
            will use the max of all meters' sample_periods.

        physical_quantity : string or list of strings
            e.g. 'power' or 'voltage' or 'energy' or ['power', 'energy'].
            If a single string then load columns only for that physical quantity.
            If a list of strings then load columns for all those physical 
            quantities.

        ac_type : string or list of strings, defaults to None
            Where 'ac_type' is short for 'alternating current type'.  e.g. 
            'reactive' or 'active' or 'apparent'.
            If set to None then will load all AC types per physical quantity.
            If set to 'best' then load the single best AC type per 
            physical quantity.
            If set to a single AC type then load just that single AC type per 
            physical quantity, else raise an Exception.
            If set to a list of AC type strings then will load all those 
            AC types and will raise an Exception if any cannot be found.

        cols : list of tuples, using NILMTK's vocabulary for measurements.
            e.g. [('power', 'active'), ('voltage', ''), ('energy', 'reactive')]
            `cols` can't be used if `ac_type` and/or `physical_quantity` are set.

        preprocessing : list of Node subclass instances
            e.g. [Clip()]

        **kwargs : any other key word arguments to pass to `self.store.load()`

        Returns
        ---------
        Always return a generator of DataFrames (even if it only has a single 
        column).

        .. note:: Different AC types will be treated separately.
        """
        if kwargs.has_key('preprocessing'):
            warn("If you are using `preprocessing` to resample then please"
                 " do not!  Instead, please use the `sample_period` parameter.")

        if sample_period is None:
            sample_period = self.sample_period()

        # Load each generator and yield the sum or the mean
        _, generators = self._meter_generators(**kwargs)
        while True:
            chunk = combine_chunks_from_generators(generators, sample_period)
            if chunk.empty:
                break
            yield chunk

    def _meter_generators(self, **kwargs):
        """Returns (list of identifiers, list of generators)."""
        generators = []
        identifiers = []
        for meter in self.meters:
            try:
                generator = meter.load(**deepcopy(kwargs))
            except MeasurementError as e:
                warn("Ignoring meter '{}' because it does not have the correct"
                     " measurements.  The MeasurementError was: '{}'"
                     .format(meter.identifier, e))
            else:
                generators.append(generator)
                identifiers.append(meter.identifier)

        return identifiers, generators

    def plot_when_on(self, **load_kwargs):
        meter_identifiers = list(self.identifier)
        fig, ax = plt.subplots()
        for i, meter in enumerate(self.meters):
            id_meter = meter.identifier
            for chunk_when_on in meter.when_on(**load_kwargs):
                series_to_plot = chunk_when_on[chunk_when_on==True]
                if len(series_to_plot.index):
                    (series_to_plot+i-1).plot(ax=ax, style='k.')
        labels = self.get_appliance_labels(meter_identifiers)
        plt.yticks(range(len(self.meters)), labels)
        plt.ylim((-0.5, len(self.meters)+0.5))
        return ax

    def simultaneous_switches(self, threshold=40):
        """
        Parameters
        ----------
        threshold : number, threshold in Watts 

        Returns
        -------
        sim_switches : pd.Series of type {timestamp: number of 
        simultaneous switches}

        Notes
        -----
        This function assumes that the submeters in this MeterGroup
        are all aligned.  If they are not then you should align the
        meters, e.g. by using an `Apply` node with `resample`.
        """
        submeters = self.submeters().meters
        count = Counter()
        for meter in submeters:
            switch_time_meter = meter.switch_times(threshold)
            for timestamp in switch_time_meter:
                count[timestamp]+=1
        sim_switches = pd.Series(count)
        # Should be 2 or more appliances changing state at the same time
        sim_switches = sim_switches[sim_switches>=2]
        return sim_switches

    def mains(self):
        """
        Returns
        -------
        ElecMeter or MeterGroup or None
        """
        if self.contains_meters_from_multiple_buildings():
            msg = ("This MeterGroup contains meters from buildings '{}'."
                   " It only makes sense to get `mains` if the MeterGroup"
                   " contains meters from a single building."
                   .format(self.building()))
            raise RuntimeError(msg)
        site_meters = [meter for meter in self.meters if meter.is_site_meter()]
        n_site_meters = len(site_meters)
        if n_site_meters == 0:
            return
        elif n_site_meters == 1:
            return site_meters[0]
        else:
            return MeterGroup(meters=site_meters)

    def use_alternative_mains(self):
        """Swap present mains meter(s) for mains meter(s) in `disabled_meters`.
        This is useful if the dataset has multiple, redundant mains meters
        (e.g. in UK-DALE buildings 1, 2 and 5).
        """
        present_mains = [m for m in self.meters if m.is_site_meter()]
        alternative_mains = [m for m in self.disabled_meters if m.is_site_meter()]
        if not alternative_mains:
            raise RuntimeError("No site meters found in `self.disabled_meters`")
        for meter in present_mains:
            self.meters.remove(meter)
            self.disabled_meters.append(meter)
        for meter in alternative_mains:
            self.meters.append(meter)
            self.disabled_meters.remove(meter)

    def upstream_meter(self):
        """Returns single upstream meter.
        Raises RuntimeError if more than 1 upstream meter.
        """
        upstream_meters = []
        for meter in self.meters:
            upstream_meters.append(meter.upstream_meter())
        unique_upstream_meters = list(set(upstream_meters))
        if len(unique_upstream_meters) > 1:
            raise RuntimeError("{:d} upstream meters found for meter group."
                               "  Should be 1.".format(len(unique_upstream_meters)))
        return unique_upstream_meters[0]

    def meters_directly_downstream_of_mains(self):
        meters = nodes_adjacent_to_root(self.wiring_graph())
        assert isinstance(meters, list)
        return meters

    def submeters(self):
        """Returns new MeterGroup of all meters except site_meters"""
        submeters = [meter for meter in self.meters
                     if not meter.is_site_meter()]
        return MeterGroup(submeters)

    def is_site_meter(self):
        """Returns True if any meters are site meters"""
        return any([meter.is_site_meter() for meter in self.meters])

    def total_energy(self, **load_kwargs):
        """Sums together total meter_energy for each meter.

        Note that this function does *not* return the total aggregate
        energy for a building.  Instead this function adds up the total energy
        for all the meters contained in this MeterGroup.  If you want the total
        aggregate energy then please use `MeterGroup.mains().total_energy()`.

        Parameters
        ----------
        full_results : bool, default=False
        **loader_kwargs : key word arguments for DataStore.load()

        Returns
        -------
        if `full_results` is True then return TotalEnergyResults object
        else return a pd.Series with a row for each AC type.
        """
        self._check_kwargs(load_kwargs)
        full_results = load_kwargs.pop('full_results', False)

        meter_energies = self._collect_stats_on_all_meters(
            load_kwargs, 'total_energy', full_results)

        if meter_energies:
            total_energy_results = meter_energies[0]
            for meter_energy in meter_energies[1:]:
                if full_results:
                    total_energy_results.unify(meter_energy)
                else:
                    total_energy_results += meter_energy
            return total_energy_results

    def _collect_stats_on_all_meters(self, load_kwargs, func, full_results):
        collected_stats = []
        for meter in self.meters:
            single_stat = getattr(meter, func)(full_results=full_results,
                                               **load_kwargs)
            collected_stats.append(single_stat)
            if (full_results and len(self.meters) > 1 and
                    not meter.store.all_sections_smaller_than_chunksize):
                warn("at least one section requested from '{}' required"
                     " multiple chunks to be loaded into memory. This may cause"
                     " a failure when we try to unify results from multiple"
                     " meters.".format(meter))

        return collected_stats

    def dropout_rate(self, **load_kwargs):
        """Sums together total energy for each meter.

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
        self._check_kwargs(load_kwargs)
        full_results = load_kwargs.pop('full_results', False)

        dropout_rates = self._collect_stats_on_all_meters(
            load_kwargs, 'dropout_rate', full_results)

        if full_results and dropout_rates:
            dropout_rate_results = dropout_rates[0]
            for dr in dropout_rates[1:]:
                dropout_rate_results.unify(dr)
            return dropout_rate_results
        else:
            return np.mean(dropout_rates)

    def _check_kwargs(self, load_kwargs):
        if (load_kwargs.get('full_results ')
                and not load_kwargs.has_key('sections')
                and len(self.meters) > 1):
            raise RuntimeError("MeterGroup stats can only return full results"
                               " objects if you specify 'sections' to load. If"
                               " you do not specify periods then the results"
                               " from individual meters are likely to be for"
                               " different periods and hence"
                               " cannot be unified.")

    def good_sections(self, **kwargs):
        """Returns good sections for just the first meter.

        TODO: combine good sections from every meter.
        """
        if self.meters:
            if len(self.meters) > 1:
                warn("As a quick implementation we only get Good Sections from"
                     " the first meter in the meter group.  We should really"
                     " return the intersection of the good sections for all"
                     " meters.  This will be fixed...")
            return self.meters[0].good_sections(**kwargs)
        else:
            return []

    def dataframe_of_meters(self, sample_period=None, **kwargs):
        """
        Parameters
        ----------
        sample_period : number
            Number of seconds to reindex meters to.

        ac_type : string, defaults to 'best'

        physical_quantity: string, defaults to 'power'

        Returns
        -------
        DataFrame
            Each column is a meter.  We select the most appropriate measurement.
            All NaNs are zeroed out.  Note that column names are string 
            representations of ElecMeterIDs (because 'pd.concat' tries to use
            tuples to construct a multiindex)
        """
        if kwargs.has_key('preprocessing'):
            warn("If you are using `preprocessing` to resample then please"
                 " do not!  Instead, please use the `sample_period` parameter.")

        if sample_period is None:
            sample_period = self.sample_period()

        resample = lambda df: df.resample(rule='{}S'.format(sample_period))
        kwargs.setdefault('preprocessing', []).append(Apply(func=resample))
        kwargs.setdefault('ac_type', 'best')
        kwargs.setdefault('physical_quantity', 'power')
        identifiers, generators = self._meter_generators(**kwargs)

        segments = []
        while True:
            chunks = []
            ids = []
            index = None
            timeframe = None
            for meter_id, generator in zip(identifiers, generators):
                try:
                    chunk_from_next_meter = next(generator)
                except StopIteration:
                    continue

                if timeframe is None:
                    timeframe = chunk_from_next_meter.timeframe
                else:
                    timeframe = timeframe.union(chunk_from_next_meter.timeframe)

                # Extend 'index' if necessary
                index = extend_index(index, timeframe, sample_period)

                # Reindex chunk_from_next_meter
                chunk_from_next_meter = chunk_from_next_meter.reindex(
                    index, method='ffill', limit=1, fill_value=0)

                ids.append(meter_id)
                chunks.append(chunk_from_next_meter.icol(0))

            if chunks:
                df = pd.concat(chunks, axis=1)
                df.columns = ids
                segments.append(df)
            else:
                break

        return pd.concat(segments)

    def entropy_per_meter(self):
        """Finds the entropy of each meter in this MeterGroup.

        Returns
        -------
        pd.Series of entropy
        """
        return self.call_method_on_all_meters('entropy')

    def call_method_on_all_meters(self, method):
        """Calls `method` on each element in `self.meters`.

        Parameters
        ----------
        method : str
            Name of a stats method in `ElecMeter`.  e.g. 'correlation'.

        Returns
        -------
        pd.Series of result of `method` called on each element in `self.meters`.
        """
        meter_identifiers = list(self.identifier)
        result = pd.Series(index=meter_identifiers)
        for meter in self.meters:
            id_meter = meter.identifier
            result[id_meter] = getattr(meter, method)()
        return result

    def pairwise(self, method):
        """
        Calls `method` on all pairs in `self.meters`.

        Assumes `method` is symmetrical.

        Parameters
        ----------
        method : str
            Name of a stats method in `ElecMeter`.  e.g. 'correlation'.

        Returns
        -------
        pd.DataFrame of the result of `method` called on each 
        pair in `self.meters`.
        """
        meter_identifiers = list(self.identifier)
        result = pd.DataFrame(index=meter_identifiers, columns=meter_identifiers)
        for i, m_i in enumerate(self.meters):
            for j, m_j in enumerate(self.meters):
                id_i = m_i.identifier
                id_j = m_j.identifier
                if i > j:
                    result[id_i][id_j] = result[id_j][id_i]
                else:
                    result[id_i][id_j] = getattr(m_i, method)(m_j)
        return result

    def pairwise_mutual_information(self):
        """
        Finds the pairwise mutual information among different 
        meters in a MeterGroup.

        Returns
        -------
        pd.DataFrame of mutual information between
        pair of ElecMeters.
        """
        return self.pairwise('mutual_information')

    def pairwise_correlation(self):
        """
        Finds the pairwise correlation among different 
        meters in a MeterGroup.

        Returns
        -------
        pd.DataFrame of correlation between pair of ElecMeters.
        """
        return self.pairwise('correlation')

    def proportion_of_energy_submetered(self, **loader_kwargs):
        """
        Returns
        -------
        float [0,1]
        """
        mains = self.mains()
        downstream_meters = self.meters_directly_downstream_of_mains()
        proportion = 0.0
        for m in downstream_meters:
            print("Calculating proportion for", m)
            prop = m.proportion_of_energy(mains, **loader_kwargs)
            proportion += prop
            print("   {:.2%}".format(prop))
            
        return proportion

    def available_ac_types(self, physical_quantity):
        """Returns set of all available alternating current types for a 
        specific physical quantity.

        Parameters
        ----------
        physical_quantity : str

        Returns
        -------
        list of strings e.g. ['apparent', 'active']
        """
        all_ac_types = [meter.available_ac_types(physical_quantity)
                        for meter in self.meters]
        return list(set(flatten_2d_list(all_ac_types)))

    def available_physical_quantities(self):
        """
        Returns
        -------
        list of strings e.g. ['power', 'energy']
        """
        all_physical_quants = [meter.available_physical_quantities()
                               for meter in self.meters]
        return list(set(flatten_2d_list(all_physical_quants)))

    def energy_per_meter(self, **load_kwargs):
        """Returns pd.DataFrame where columns is meter.identifier and 
        each value is total energy.  Index is AC types.

        Does not care about wiring hierarchy.  Does not attempt to ensure all 
        channels share the same time sections.
        """
        energy_per_meter = pd.DataFrame(columns=self.instance(), index=AC_TYPES)
        n_meters = len(self.meters)
        for i, meter in enumerate(self.meters):
            print('\r{:d}/{:d} {}'.format(i+1, n_meters, meter), end='')
            stdout.flush()
            meter_energy = meter.total_energy(**load_kwargs)
            energy_per_meter[meter.identifier] = meter_energy
        return energy_per_meter.dropna(how='all')

    def fraction_per_meter(self, **load_kwargs):
        """Fraction of energy per meter.

        Return pd.Series.  Index is meter.instance.  
        Each value is a float in the range [0,1].
        """
        energy_per_meter = self.energy_per_meter(**load_kwargs).max()
        total_energy = energy_per_meter.sum()
        return energy_per_meter / total_energy

    def proportion_of_upstream_total_per_meter(self, **load_kwargs):
        prop_per_meter = pd.Series(index=self.identifier)
        n_meters = len(self.meters)
        for i, meter in enumerate(self.meters):
            proportion = meter.proportion_of_upstream(**load_kwargs)
            print('\r{:d}/{:d} {} = {:.3f}'
                  .format(i+1, n_meters, meter, proportion), end='')
            stdout.flush()
            prop_per_meter[meter.identifier] = proportion
        prop_per_meter.sort(ascending=False)
        return prop_per_meter

    def train_test_split(self, train_fraction=0.5):
        """
        Parameters
        ----------
        train_fraction

        Returns
        -------
        split_time: pd.Timestamp where split should happen
        """

        assert(
            0 < train_fraction < 1), "`train_fraction` should be between 0 and 1"

        # TODO: currently just works with the first mains meter, assuming
        # both to be simultaneosly sampled
        mains_first_meter = self.mains().meters[0]
        good_sections = mains_first_meter.good_sections()
        sample_period = mains_first_meter.device['sample_period']
        appx_num_records_in_each_good_section = [
            int((ts.end - ts.start).total_seconds() / sample_period) for ts in good_sections]
        appx_total_records = sum(appx_num_records_in_each_good_section)
        records_in_train = appx_total_records * train_fraction
        seconds_in_train = int(records_in_train * sample_period)
        if len(good_sections) == 1:
            # all data is contained in one good section
            split_point = good_sections[
                0].start + timedelta(seconds=seconds_in_train)
            return split_point
        else:
            # data is split across multiple time deltas
            records_remaining = records_in_train
            while records_remaining:
                for i, records_in_section in enumerate(appx_num_records_in_each_good_section):
                    if records_remaining > records_in_section:
                        records_remaining -= records_in_section
                    elif records_remaining == records_in_section:
                        # Next TimeFrame is the split point!!
                        split_point = good_sections[i + 1].start
                        return split_point
                    else:
                        # Need to split this timeframe
                        split_point = good_sections[
                            i].start + timedelta(seconds=sample_period * records_remaining)
                        return split_point

    ################## FUNCTIONS NOT YET IMPLEMENTED ###################
    # def init_new_dataset(self):
    #     self.infer_and_set_meter_connections()
    #     self.infer_and_set_dual_supply_appliances()
    # def infer_and_set_meter_connections(self):
    #     """
    #     Arguments
    #     ---------
    #     meters : list of Meter objects
    #     """
    # Maybe this should be a stand-alone function which
    # takes a list of meters???
    #     raise NotImplementedError
    # def infer_and_set_dual_supply_appliances(self):
    #     raise NotImplementedError
    # def total_on_duration(self):
    #     """Return timedelta"""
    #     raise NotImplementedError
    # def on_durations(self):
    # self.get_unique_upstream_meters()
    # for each meter, get the on time,
    # assuming the on-power-threshold for the
    # smallest appliance connected to that meter???
    #     raise NotImplementedError
    # def activity_distribution(self, bin_size, timespan):
    #     raise NotImplementedError
    # def cross_correlation(self):
    #     """Correlation between items."""
    #     raise NotImplementedError
    # def on_off_events(self, minimum_state_duration):
    #     raise NotImplementedError
    def select_top_k(self, k=5):
        """
        Returns
        -------
        MeterGroup containing top k meters.
        """
        # Filtering out mains to create a meter group of appliances
        appliance_meter_group = self.submeters()
        # Energy per appliance
        appliance_energy_per_meter = appliance_meter_group.energy_per_meter()

        # Removing appliances which may have no energy!
        # See https://github.com/nilmtk/nilmtk/issues/174
        appliances_to_ignore = []
        for appliance_id in appliance_energy_per_meter.columns:
            num_non_null_entries = appliance_energy_per_meter[
                appliance_id].isnull().sum()
            print(appliance_id, num_non_null_entries)
            if (num_non_null_entries == len(appliance_energy_per_meter[appliance_id].index)):
                appliances_to_ignore.append(appliance_id)

        print(appliances_to_ignore)

        appliances_to_consider = list(
            set(appliance_energy_per_meter.columns) - set(appliances_to_ignore))

        appliance_energy_per_meter = appliance_energy_per_meter[appliances_to_consider]

        # Finding the most relevant measurement to sort on. For now, this is a
        # simple function which considers the measurement having the most records
        # if there is only a single measurement, just take that
        if len(appliance_energy_per_meter.T.columns) == 1:
            measurement = appliance_energy_per_meter.T.columns[0]
        else:
            # temp find best measurement
            temp = appliance_energy_per_meter.T.isnull().sum()
            temp.sort()
            measurement = temp.head(1).index[0]
        print(measurement)
        top_k_appliance_index = flatten_2d_list(appliance_energy_per_meter.T.sort(
            columns=[measurement], ascending=False).head(k).index.tolist())
        meters_top_k = []
        for meter in self.meters:

            if isinstance(meter, MeterGroup):
                if meter.meters[0].instance() in top_k_appliance_index:
                    meters_top_k.append(meter)
            else:
                if meter.instance() in top_k_appliance_index:
                    meters_top_k.append(meter)
        return MeterGroup(meters_top_k)

    # def select_meters_contributing_more_than(self, threshold_proportion):
    #     """Return new MeterGroup with all meters whose proportion of
    #     energy usage is above threshold percentage."""
    # see prepb.filter_contribution_less_than_x(building, x)
    #     raise NotImplementedError


    # SELECTION FUNCTIONS NOT IMPLEMENTED YET

    # def groupby(self, **kwargs):
    #     """
    #     e.g. groupby('category')

    #     Returns
    #     -------
    #     A dict of MeterGroup objects e.g.:
    #       {'cold': MeterGroup, 'hot': MeterGroup}
    #     """
    #     raise NotImplementedError

    def get_timeframe(self):
        """
        Returns
        -------
        nilmtk.TimeFrame representing the timeframe which is the union
            of all meters in self.meters.
        """
        timeframe = None
        for meter in self.meters:
            if timeframe is None:
                timeframe = meter.get_timeframe()
            elif meter.get_timeframe().empty:
                pass
            else:
                timeframe = timeframe.union(meter.get_timeframe())
        return timeframe

    def plot(self, **kwargs):
        """
        Parameters
        ----------
        start, end : str or pd.Timestamp or datetime or None, optional
        width : int, optional
            Number of points on the x axis required
        ax : matplotlib.axes, optional
        plot_legend : boolean, optional
            Defaults to True.  Set to False to not plot legend.
        kind : {'separate lines', 'summed'}
        """

        """
        TODO: 
        Params
        ------
        kind : {'stacked', 'heatmap', 'lines', 'snakey'}

        pretty snakey:
        http://www.cl.cam.ac.uk/research/srg/netos/c-aware/joule/V4.00/
        """
        # Load data and plot each meter
        kind = kwargs.pop('kind', 'separate lines')
        if kind == 'separate lines':
            # Get start and end times for the plot
            start = convert_to_timestamp(kwargs.pop('start', None))
            end = convert_to_timestamp(kwargs.pop('end', None))
            if start is None or end is None:
                timeframe_for_group = self.get_timeframe()
                if start is None:
                    start = timeframe_for_group.start
                if end is None:
                    end = timeframe_for_group.end

            ax = kwargs.pop('ax', None)
            for meter in self.meters:
                ax = meter.plot(start=start, end=end, ax=ax, plot_legend=False, 
                                **kwargs)

            if kwargs.pop('plot_legend', True):
                plt.legend()

        elif kind == 'summed':
            ax = super(MeterGroup, self).plot(**kwargs)

        return ax

    def appliance_label(self):
        """
        Returns
        -------
        string : A label listing all the appliance types.
        """
        return ", ".join(set([meter.appliance_label() for meter in self.meters]))

    def clear_cache(self):
        for meter in self.meters:
            meter.clear_cache()
        
    def correlation_of_sum_of_submeters_with_mains(self, **load_kwargs):
        submeters = self.meters_directly_downstream_of_mains()
        submeters = MeterGroup(submeters)
        return self.mains().correlation(submeters, **load_kwargs)


def iterate_through_submeters_of_two_metergroups(master, slave):
    """
    Parameters
    ----------
    master, slave : MeterGroup

    Returns
    -------
    list of 2-tuples of the form (`master_meter`, `slave_meter`)
    """
    zipped = []
    for master_meter in master.submeters().meters:
        slave_identifier = master_meter.identifier._replace(
            dataset=slave.dataset())
        slave_meter = slave[slave_identifier]
        zipped.append((master_meter, slave_meter))
    return zipped


def combine_chunks_from_generators(generators, sample_period):
    """Combines chunks into a single DataFrame.

    Adds or averages columns, depending on whether each column is in
    PHYSICAL_QUANTITIES_TO_AVERAGE.

    Parameters
    ----------
    generators : list of generators of nilmtk DataFrames
    sample_period : number, seconds

    Returns
    -------
    DataFrame
    """
    # The approach is that we first add everything together
    # in the first for-loop, whilst also keeping a 
    # `columns_to_average_counter` DataFrame
    # which tells us what to divide by in order to compute the 
    # mean for PHYSICAL_QUANTITIES_TO_AVERAGE.

    chunk = pd.DataFrame()
    columns_to_average_counter = pd.DataFrame()
    timeframe = None

    # Go through each generator to try sum values together
    index = None
    for generator in generators:
        try:
            chunk_from_next_meter = next(generator)
        except StopIteration:
            continue

        if timeframe is None:
            timeframe = chunk_from_next_meter.timeframe
        else:
            timeframe = timeframe.union(chunk_from_next_meter.timeframe)

        # Extend 'index' if necessary
        index = extend_index(index, timeframe, sample_period)

        # Reindex chunk_from_next_meter
        chunk_from_next_meter = chunk_from_next_meter.reindex(
            index, method='ffill', limit=1, fill_value=0)

        # Add
        try:
            chunk = chunk.add(chunk_from_next_meter, fill_value=0)
        except ValueError as e:
            if str(e) != "cannot join with no level specified and no overlapping names":
                raise
            chunk = chunk.add(chunk_from_next_meter, fill_value=0, 
                              level='physical_quantity') 

        # Update columns_to_average_counter - this is necessary so we do not
        # add up columns like 'voltage' which should be averaged.
        physical_quantities = chunk_from_next_meter.columns.get_level_values('physical_quantity')
        columns_to_average = (set(PHYSICAL_QUANTITIES_TO_AVERAGE)
                              .intersection(physical_quantities))
        counter_increment = pd.DataFrame(1, columns=columns_to_average, 
                                         index=chunk_from_next_meter.index)
        columns_to_average_counter = columns_to_average_counter.add(
            counter_increment, fill_value=0)

    # Create mean values by dividing any columns which need dividing
    for column in columns_to_average_counter:
        chunk[column] /= columns_to_average_counter[column]

    chunk.timeframe = timeframe
    return chunk


def extend_index(index, timeframe, sample_period):
    """Extends `index` to size of timeframe, ensuring that we maintain
    a regular interval between each index element.

    Parameters
    ----------
    index : pd.DatetimeIndex or None
    timeframe : nilmtk.TimeFrame
    sample_period : number, seconds

    Returns
    -------
    pd.DatetimeIndex
    """
    freq = '{}S'.format(sample_period)
    if index is None:
        return pd.date_range(timeframe.start, timeframe.end, freq=freq)

    # extend beginning of index if needs be
    if index[0] > timeframe.start:
        seconds = (index[0] - timeframe.start).total_seconds()
        periods = int(np.ceil(seconds / sample_period))
        new_index = pd.date_range(start=None, end=index[0], freq=freq, 
                                  closed='right', periods=periods)
        index = new_index[:-1] + index

    # extend end of index if needs be
    if index[-1] < timeframe.end: 
        new_index = pd.date_range(start=index[-1], end=timeframe.end, 
                                  freq=freq, closed='left')
        index = index + new_index[1:]

    return index
