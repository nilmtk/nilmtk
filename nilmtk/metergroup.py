from __future__ import print_function, division
import networkx as nx
import pandas as pd
from warnings import warn
from .elecmeter import ElecMeter, ElecMeterID
from .appliance import Appliance
from .datastore import join_key
from .utils import (tree_root, nodes_adjacent_to_root, simplest_type_for,
                    flatten_2d_list)
from .measurement import select_best_ac_type, AC_TYPES

class MeterGroup(object):
    """A group of ElecMeter objects. Can contain nested MeterGroup objects.

    Implements many of the same methods as ElecMeter.
    
    Attributes
    ----------
    meters : list of ElecMeters
    """
    def __init__(self, meters=None):
        self.meters = [] if meters is None else list(meters)

    def load(self, store, elec_meters, appliances, building_id):
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
            appliance = Appliance(appliance_md)
            meter_ids = [ElecMeterID(instance=meter_instance,
                                     building=building_id.instance,
                                     dataset=building_id.dataset)
                         for meter_instance in appliance.metadata['meters']]

            if appliance.n_meters == 1:
                # Attach this appliance to just a single meter
                meter = self[meter_ids[0]]
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
            raise RuntimeError("More than one dominant appliance in MeterGroup!")

    def nested_metergroups(self):
        return [m for m in self.meters if isinstance(m, MeterGroup)]

    def __getitem__(self, key):
        """Get a single meter.
        
        These formats for `key` are accepted:
        * [1] - retrieves meter instance 1, raises Exception if there are 
                more than one meter with this instance, raises KeyError
                if none are found.  If meter instance 1 is in a nested MeterGroup
                then retrieve the ElecMeter, not the MeterGroup.
        * [ElecMeterID(1, 1, 'REDD')] - retrieves meter with specified meter ID
        * [[ElecMeterID(1, 1, 'REDD')], [ElecMeterID(2, 1, 'REDD')]] - retrieves
          MeterGroup containing meter instances 1 and 2.
        * [ElecMeterID((1,2), 1, 'REDD')] - retrieve MeterGroup with meters 1 & 2
        * ['toaster']    - retrieves meter or group upstream of toaster instance 1
        * ['toaster', 2] - retrieves meter or group upstream of toaster instance 2
        * [{'dataset': 'redd', 'building': 3, 'type': 'toaster', 'instance': 2}]

        Returns
        -------
        ElecMeter or MeterGroup
        """
        if isinstance(key, str):
            # default to get first meter
            return self[(key, 1)]
        elif isinstance(key, ElecMeterID):
            for meter in self.meters:
                if meter.identifier == key:
                    return meter
            if isinstance(key.instance, tuple):
                # find meter group from a key of the form
                # ElecMeterID(instance=(1,2), building=1, dataset='REDD')
                for group in self.nested_metergroups():
                    if (set(group.instance()) == set(key.instance) and
                        group.building() == key.building and
                        group.dataset() == key.dataset):
                        return group
            raise KeyError(key)
        elif isinstance(key, list): # find MeterGroup from list of ElecMeterIDs
            if not all([isinstance(item, tuple) for item in key]):
                raise TypeError("requires a list of ElecMeterID objects.")
            for meter in self.meters: # TODO: write unit tests for this
                if isinstance(meter, MeterGroup):
                    metergroup = meter
                    meter_ids = set([m.identifier for m in metergroup.meters
                                     if isinstance(m, ElecMeter)])
                    if meter_ids == set(key):
                        return meter
            raise KeyError(key)
        elif isinstance(key, tuple):
            if len(key) == 2:
                return self[{'type': key[0], 'instance': key[1]}]
            else:
                raise TypeError()
        elif isinstance(key, dict):
            for meter in self.meters:
                if meter.matches(key):
                    return meter
            raise KeyError(key)
        elif isinstance(key, int) and not isinstance(key, bool):
            meters_found = []
            for meter in self.meters:
                if isinstance(meter.instance(), int):
                    if meter.instance() == key:
                        meters_found.append(meter)
                elif isinstance(meter.instance(), (tuple, list)):
                    if key in meter.instance():
                        print("Meter", key, "is in a nested meter group."
                              " Retrieving just the ElecMeter.")
                        meters_found.append(meter[key])
            n_meters_found = len(meters_found)
            if n_meters_found > 1:
                raise Exception('{} meters found with instance == {}'
                                .format(n_meters_found, key))
            elif n_meters_found == 0:
                raise KeyError('No meters found with instance == {}'.format(key))
            else:
                return meters_found[0]
        else:
            raise TypeError()

    def matches(self, key):
        for meter in self.meters:
            if meter.matches(key):
                return True
        return False

    def select(self, *args, **kwargs):
        """Select a group of meters.

        e.g. 
        * select(category='lighting')
        * select(type='fridge')
        * select(building=1, category='lighting')
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
        for meter in self.meters:
            if meter.matches(kwargs):
                selected_meters.append(meter)

        return MeterGroup(selected_meters)

    @classmethod
    def from_all_meters_in_dataset(cls, meter_ids):
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
        meter_ids = list(set(meter_ids)) # make unique
        meters = []
        for meter_id in meter_ids:
            if isinstance(meter_id, ElecMeterID):
                meters.append(ElecMeter.all_meters[meter_id])
            elif isinstance(meter_id, tuple):
                metergroup = MeterGroup.from_all_meters_in_dataset(meter_id)
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
        return MeterGroup.from_all_meters_in_dataset(new_identifiers)

    def __eq__(self, other):
        if isinstance(other, MeterGroup):
            return other.meters == self.meters
        else:
            return False

    @property
    def appliances(self):
        appliances = set()
        for meter in self.meters:
            appliances.update(meter.appliances)
        return appliances

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
                    if upstream_meter is not None:
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
                meter_instances = [m.identifier.instance for m in metergroup.meters]
                labels[meter] = meter_instances
        nx.draw(graph, labels=labels)        

    def power_series(self, **kwargs):
        """Sum together all meters and return power Series.

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

        Note
        ----
        If meters do not align then resample (by passing in 
        `preprocessing=[Resample()]`) to get multiple meters to align.
        """

        # Get a list of generators
        generators = []
        for meter in self.meters:
            generators.append(meter.power_series(**kwargs))
        # Now load each generator and yield the sum
        while True:
            try:
                chunk = next(generators[0])
            except StopIteration:
                break

            timeframe = chunk.timeframe
            for generator in generators[1:]:
                another_chunk = next(generator)
                timeframe = timeframe.intersect(another_chunk.timeframe)
                chunk += another_chunk

            chunk = chunk.dropna()
            chunk.timeframe = timeframe
            yield chunk

    def mains(self):
        """
        Returns
        -------
        ElecMeter or MeterGroup or None
        """
        site_meters = [meter for meter in self.meters if meter.is_site_meter()]
        n_site_meters = len(site_meters)
        if n_site_meters == 0:
            return
        elif n_site_meters == 1:
            return site_meters[0]
        else:
            return MeterGroup(meters=site_meters)

    def meters_directly_downstream_of_mains(self):
        meters = nodes_adjacent_to_root(self.wiring_graph())
        assert isinstance(meters, list)
        return meters

    def submeters(self):
        """Returns list of all meters except site_meters"""
        return [meter for meter in self.meters
                if not meter.is_site_meter()]

    def is_site_meter(self):
        """Returns True if any meters are site meters"""
        return any([meter.is_site_meter() for meter in self.meters])

    def total_energy(self, **load_kwargs):
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
        full_results = load_kwargs.pop('full_results', False)
        total_energy = None
        for meter in self.meters:
            meter_energy = meter.total_energy(full_results=True, **load_kwargs)
            if total_energy is None:
                total_energy = meter_energy
            else:
                total_energy.unify(meter_energy)

        if full_results: 
            return total_energy
        else:
            return total_energy.simple()

    def dataframe_of_meters(self, rule='1T'):
        """
        Returns
        -------
        DataFrame
            Each column is a meter.  We select the most appropriate measurement.
            First column is mains.
            All NaNs are zeroed out.
            Chunks (which may not be consecutive) are put together. i.e. there 
            will be breaks in the index where there were holes in the mains data.

        Note
        ----
        * we use 'meters_directly_downsteam_of_mains' instead of most distal meters
        * think this was written when rushing to get disaggregation to
          work with NILMTK v0.2.  Might not need this function once we teach
          disaggregators to handle individual appliances, chunk by chunk.
        """
        submeters_dict = {}
        mains = self.mains()
        mains_good_sections = mains.good_sections()
        mains_energy = mains.total_energy(periods=mains_good_sections)
        energy_ac_type = select_best_ac_type(mains_energy.keys())
        energy_threshold = mains_energy[energy_ac_type] * 0.05

        # TODO: should iterate through 'most distal' meters
        for meter in [self.mains()] + self.meters_directly_downstream_of_mains():
            meter_energy = meter.total_energy(periods=mains_good_sections)
            meter_energy_ac_type = select_best_ac_type(meter_energy.keys(),
                                                       mains_energy.keys())
            if meter_energy[meter_energy_ac_type] < energy_threshold:
                continue

            # TODO: resampling etc should happen in pipeline
            chunks = []
            for chunk in meter.power_series(periods=mains_good_sections):
                chunk = chunk.resample(rule=rule, how='mean')

                # Protect against getting duplicate indicies
                if chunks and chunks[-1].index[-1] == chunk.index[0]:
                    chunks.append(chunk.iloc[1:])
                else:
                    chunks.append(chunk)
                
            power_series = pd.concat(chunks)

            # need to make sure 
            # resample stays in sync with mains power_series.  Maybe want reindex???
            # If we're careful then I think we can get power_series with index
            # in common with mains, without having to post-process it
            # like prepb.make_common_index(building)

            # TODO: insert zeros and then ffill
            power_series.fillna(value=0, inplace=True)
            submeters_dict[meter.identifier] = power_series
        return pd.DataFrame(submeters_dict)

    def proportion_of_energy_submetered(self):
        """
        Returns
        -------
        float [0,1]
        """
        mains = self.mains()
        good_mains_sections = mains.good_sections()
        print("number of good sections =", len(good_mains_sections))
        submetered_energy = 0.0
        common_ac_types = None
        for meter in self.meters_directly_downstream_of_mains():
            energy = meter.total_energy(periods=good_mains_sections)
            ac_types = set(energy.keys())
            ac_type = select_best_ac_type(ac_types, 
                                          mains.available_power_ac_types())
            submetered_energy += energy[ac_type]
            if common_ac_types is None:
                common_ac_types = ac_types
            else:
                common_ac_types = common_ac_types.intersection(ac_types)
        mains_energy = mains.total_energy()
        ac_type = select_best_ac_type(mains_energy.keys(), common_ac_types)
        return submetered_energy / mains_energy[ac_type]
    
    def available_power_ac_types(self):
        """Returns set of all AC types recorded by all meters"""
        all_ac_types = [meter.available_power_ac_types() for meter in self.meters]
        return set(flatten_2d_list(all_ac_types))

    def energy_per_meter(self, **load_kwargs):
        """Returns pd.DataFrame where columns is meter.instance and 
        each value is total energy.  Index is AC types.

        Does not care about wiring hierarchy.  Does not attempt to ensure all 
        channels share the same time periods.
        """ 
        energy_per_meter = pd.DataFrame(columns=self.instance(), index=AC_TYPES)
        for meter in self.meters:
            meter_energy = meter.total_energy(full_results=True, **load_kwargs)
            energy_per_meter[meter.instance()] = meter_energy.combined()
        return energy_per_meter.dropna(how='all')

    def fraction_per_meter(self, **load_kwargs):
        """Fraction of energy per meter.

        Return pd.Series.  Index is meter.instance.  
        Each value is a float in the range [0,1].
        """
        energy_per_meter = self.energy_per_meter(**load_kwargs).max()
        total_energy = energy_per_meter.sum()
        return energy_per_meter / total_energy

    ################## NOT IMPLEMENTED FUNCTIONS ###################
    def init_new_dataset(self):
        self.infer_and_set_meter_connections()
        self.infer_and_set_dual_supply_appliances()
            
    def infer_and_set_meter_connections(self):
        """
        Arguments
        ---------
        meters : list of Meter objects
        """
        # Maybe this should be a stand-alone function which
        # takes a list of meters???
        raise NotImplementedError
        
    def infer_and_set_dual_supply_appliances(self):
        raise NotImplementedError
    
    def plot(self, how='stacked'):
        """
        Arguments
        ---------
        stacked : {'stacked', 'heatmap', 'lines', 'snakey'}
        """
        # pretty snakey:
        # http://www.cl.cam.ac.uk/research/srg/netos/c-aware/joule/V4.00/
        raise NotImplementedError

    def total_on_duration(self):
        """Return timedelta"""
        raise NotImplementedError
    
    def on_durations(self):
        # self.get_unique_upstream_meters()
        # for each meter, get the on time, 
        # assuming the on-power-threshold for the 
        # smallest appliance connected to that meter???
        raise NotImplementedError
    
    def activity_distribution(self, bin_size, timespan):
        raise NotImplementedError
    
    def when_on(self, on_power_threshold):
        """Return Series of bools"""
        raise NotImplementedError
    
    def cross_correlation(self):
        """Correlation between items."""
        raise NotImplementedError
                    
    def on_off_events(self, minimum_state_duration):
        raise NotImplementedError
    
    def select_top_k(self, k=5):
        """
        Returns
        -------
        MeterGroup containing top k meters.
        """
        top_k = self.energy_per_meter().iloc[:k]
            
    def select_meters_contributing_more_than(self, threshold_proportion):
        """Return new MeterGroup with all meters whose proportion of
        energy usage is above threshold percentage."""
        # see prepb.filter_contribution_less_than_x(building, x)
        raise NotImplementedError
        
    
    # SELECTION FUNCTIONS NOT IMPLEMENTED YET

    def groupby(self, **kwargs):
        """
        e.g. groupby('category')
        
        Returns
        -------
        A dict of MeterGroup objects e.g.:
          {'cold': MeterGroup, 'hot': MeterGroup}
        """
        raise NotImplementedError


def iterate_through_submeters_of_two_metergroups(master, slave):
    """
    Parameters
    ---------
    master, slave : MeterGroup

    Returns
    -------
    list of 2-tuples of the form (`master_meter`, `slave_meter`)
    """
    zipped = []
    for master_meter in master.submeters():
        slave_identifier = master_meter.identifier._replace(
            dataset=slave.dataset())
        slave_meter = slave[slave_identifier]
        zipped.append((master_meter, slave_meter))
    return zipped
