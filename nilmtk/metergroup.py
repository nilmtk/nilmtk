import warnings
import gc
from sys import stdout
from collections import Counter
from copy import copy, deepcopy
from collections import namedtuple
from datetime import timedelta

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey 
from matplotlib.ticker import FuncFormatter
from matplotlib import MatplotlibDeprecationWarning

# NILMTK imports
from .elecmeter import ElecMeter, ElecMeterID
from .appliance import Appliance
from .datastore.datastore import join_key
from .utils import (tree_root, nodes_adjacent_to_root, simplest_type_for,
                    flatten_2d_list, convert_to_timestamp, normalise_timestamp,
                    print_on_line, convert_to_list, append_or_extend_list,
                    most_common, capitalise_first_letter)
from .plots import plot_series
from .measurement import (select_best_ac_type, AC_TYPES, LEVEL_NAMES,
                          PHYSICAL_QUANTITIES_TO_AVERAGE)
from nilmtk.exceptions import MeasurementError
from .electric import Electric
from .timeframe import TimeFrame, split_timeframes
from .preprocessing import Apply
from .datastore import MAX_MEM_ALLOWANCE_IN_BYTES
from nilmtk.timeframegroup import TimeFrameGroup

# MeterGroupID.meters is a tuple of ElecMeterIDs.  Order doesn't matter.
# (we can't use a set because sets aren't hashable so we can't use 
# a set as a dict key or a DataFrame column name.)
MeterGroupID = namedtuple('MeterGroupID', ['meters'])


class MeterGroup(Electric):

    """A group of ElecMeter objects. Can contain nested MeterGroup objects.

    Implements many of the same methods as ElecMeter.

    Attributes
    ----------
    meters : list of ElecMeters or nested MeterGroups
    disabled_meters : list of ElecMeters or nested MeterGroups
    name : only set by functions like 'groupby' and 'select_top_k'
    """

    def __init__(self, meters=None, disabled_meters=None):
        self.meters = convert_to_list(meters)
        self.disabled_meters = convert_to_list(disabled_meters)
        self.name = ""
    
    def __hash__(self):
        """
        Provide a hash based on the MeterGroup's name, meters and 
        disabled_meters
        """ 
        return hash((self.name, tuple(self.meters), tuple(self.disabled_meters)))

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
            warnings.warn("Building {} has an empty 'elec_meters' object."
                 .format(building_id.instance), RuntimeWarning)
        if not appliances:
            warnings.warn("Building {} has an empty 'appliances' list."
                 .format(building_id.instance), RuntimeWarning)

        # Load static Meter Devices
        ElecMeter.load_meter_devices(store)

        # Load each meter
        for meter_i, meter_metadata_dict in elec_meters.items():
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
                if isinstance(meter, MeterGroup):  # MeterGroup of site_meters
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
        dominant_appliances = list(set(dominant_appliances))
        n_dominant_appliances = len(dominant_appliances)
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
        * `MeterGroupID(meters=(ElecMeterID(1, 1, 'REDD')))` - retrieves
          existing nested MeterGroup containing exactly meter instances 1 and 2.
        * `[ElecMeterID(1, 1, 'REDD'), ElecMeterID(2, 1, 'REDD')]` - retrieves
          existing nested MeterGroup containing exactly meter instances 1 and 2.
        * `ElecMeterID(0, 1, 'REDD')` - instance `0` means `mains`. This returns
           a new MeterGroup of all site_meters in building 1 in REDD.
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
        elif isinstance(key, MeterGroupID):
            key_meters = set(key.meters)
            for group in self.nested_metergroups():
                if (set(group.identifier.meters) == key_meters):
                    return group
            raise KeyError(key)
        # find MeterGroup from list of ElecMeterIDs
        elif isinstance(key, list):
            if not all([isinstance(item, tuple) for item in key]):
                raise TypeError("requires a list of ElecMeterID objects.")
            for meter in self.meters:  # TODO: write unit tests for this
                # list of ElecMeterIDs.  Return existing MeterGroup
                if isinstance(meter, MeterGroup):
                    metergroup = meter
                    meter_ids = set(metergroup.identifier.meters)
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
        func = kwargs.pop('func', 'matches')

        def get(_kwargs):
            exception_raised_every_time = True
            exception = None
            no_match = True
            for meter in self.meters:
                try:
                    match = getattr(meter, func)(_kwargs)
                except KeyError as e:
                    exception = e
                else:
                    exception_raised_every_time = False
                    if match:
                        selected_meters.append(meter)
                        no_match = False
            if no_match:
                raise KeyError("'No match for {}'".format(_kwargs))
            if exception_raised_every_time and exception is not None:
                raise exception

        if len(kwargs) == 1 and isinstance(next(iter(kwargs.values())), list):
            attribute = next(iter(kwargs.keys()))
            list_of_values = next(iter(kwargs.values()))
            for value in list_of_values:
                get({attribute: value})
        else:
            get(kwargs)

        return MeterGroup(selected_meters)

    def select_using_appliances(self, **kwargs):
        """Select a group of meters based on appliance metadata.

        e.g. 
        * select_using_appliances(category='lighting')
        * select_using_appliances(type='fridge')
        * select_using_appliances(type=['fridge', 'kettle', 'toaster'])
        * select_using_appliances(building=1, category='lighting')
        * select_using_appliances(room='bathroom')

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
            Each element is an ElecMeterID or a MeterGroupID.

        Returns
        -------
        MeterGroup
        """
        meter_ids = list(meter_ids)
        meters = []

        def append_meter_group(meter_id):
            try:
                # see if there is an existing MeterGroup
                metergroup = self[meter_id]
            except KeyError:
                # there is no existing MeterGroup so assemble one
                metergroup = self.from_list(meter_id.meters)
            meters.append(metergroup)

        already_processed = set()
        for meter_id in meter_ids:
            if meter_id in already_processed:
                continue

            already_processed.add(meter_id)

            if isinstance(meter_id, ElecMeterID):
                meters.append(self[meter_id])
            elif isinstance(meter_id, MeterGroupID):
                append_meter_group(meter_id)
            elif isinstance(meter_id, tuple):
                meter_id = MeterGroupID(meters=meter_id)
                append_meter_group(meter_id)
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
        other_identifiers = other.identifier.meters
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
        metergroup = MeterGroup()
        metergroup.from_list(new_identifiers)
        return metergroup

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

    def dominant_appliances(self):
        appliances = set()
        for meter in self.meters:
            appliances.add(meter.dominant_appliance())
        return list(appliances)

    def values_for_appliance_metadata_key(self, key, 
                                          only_consider_dominant_appliance=True):
        """
        Parameters
        ----------
        key : str
            e.g. 'type' or 'categories' or 'room'
        
        Returns
        -------
        list
        """
        values = []
        if only_consider_dominant_appliance:
            appliances = self.dominant_appliances()
        else:
            appliances = self.appliances

        for appliance in appliances:
            value = appliance.metadata.get(key)
            append_or_extend_list(values, value)
            value = appliance.type.get(key)
            append_or_extend_list(values, value)
        return list(set(values))

    def get_labels(self, meter_ids, pretty=True):
        """Create human-readable meter labels.

        Parameters
        ----------
        meter_ids : list of ElecMeterIDs (or 3-tuples in same order as ElecMeterID)

        Returns
        -------
        list of strings describing the appliances.
        """
        meters = [self[meter_id] for meter_id in meter_ids]
        labels = [meter.label(pretty=pretty) for meter in meters]
        return labels

    def __repr__(self):
        s = "{:s}(meters=\n".format(self.__class__.__name__)
        for meter in self.meters:
            s += "  " + str(meter).replace("\n", "\n  ") + "\n"
        s += ")"
        return s

    @property
    def identifier(self):
        """Returns a MeterGroupID."""
        return MeterGroupID(meters=tuple([meter.identifier for meter in self.meters]))

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
                    upstream_meter = meter.upstream_meter(raise_warning=False)
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

    def draw_wiring_graph(self, show_meter_labels=True):
        graph = self.wiring_graph()
        
        try:
            # Try using graphviz layout...
            pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog='dot')
            used_graphviz = True
        except:
            # ...and fallback to shell layout if graphviz is not installed or
            # doesn't work
            pos = nx.shell_layout(graph)
            used_graphviz = False
            
        meter_labels = {meter: meter.label() for meter in graph.nodes()}
        if show_meter_labels:
            for meter, name in meter_labels.items():
                x, y = pos[meter]

                if used_graphviz:
                    if meter.is_site_meter():
                        delta_y = 5
                    else:
                        delta_y = -5
                    
                    plt.text(x, y + delta_y, s=name, bbox=dict(facecolor='red', alpha=0.5), horizontalalignment='center')
        

        with warnings.catch_warnings():
            #TODO: update networkx to 2.4 when it is released and remove this filter
            warnings.simplefilter('ignore', category=MatplotlibDeprecationWarning)
            if used_graphviz:
                # meter_labels already drawn
                nx.draw(graph, pos, arrows=False)
            else:
                nx.draw(graph, pos, labels=meter_labels, arrows=False)
            
        ax = plt.gca()
        
        return graph, ax

    def load(self, **kwargs):
        """Returns a generator of DataFrames loaded from the DataStore.

        By default, `load` will load all available columns from the DataStore.  
        Specific columns can be selected in one or two mutually exclusive ways:

        1. specify a list of column names using the `columns` parameter.
        2. specify a `physical_quantity` and/or an `ac_type` parameter to ask 
           `load` to automatically select columns.

        Each meter in the MeterGroup will first be resampled before being added.
        The returned DataFrame will include NaNs at timestamps where no meter
        had a sample (after resampling the meter).

        Parameters
        ----------
        sample_period : int or float, optional
            Number of seconds to use as sample period when reindexing meters.
            If not specified then will use the max of all meters' sample_periods.
        resample_kwargs : dict of key word arguments (other than 'rule') to 
            `pass to pd.DataFrame.resample()`
        chunksize : int, optional
            the maximum number of rows per chunk. Note that each chunk is 
            guaranteed to be of length <= chunksize.  Each chunk is *not*
            guaranteed to be exactly of length == chunksize.
        **kwargs : 
            any other key word arguments to pass to `self.store.load()` including:
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
        columns : list of tuples, using NILMTK's vocabulary for measurements.
            e.g. [('power', 'active'), ('voltage', ''), ('energy', 'reactive')]
            `columns` can't be used if `ac_type` and/or `physical_quantity` are set.
        preprocessing : list of Node subclass instances
            e.g. [Clip()]

        Returns
        ---------
        Always return a generator of DataFrames (even if it only has a single 
        column).

        .. note:: Different AC types will be treated separately.
        """
        # Handle kwargs
        sample_period = kwargs.setdefault('sample_period', self.sample_period())
        sections = kwargs.pop('sections', [self.get_timeframe()])
        chunksize = kwargs.pop('chunksize', MAX_MEM_ALLOWANCE_IN_BYTES)
        duration_threshold = sample_period * chunksize
        columns = pd.MultiIndex.from_tuples(
            self._convert_physical_quantity_and_ac_type_to_cols(**kwargs)['columns'],
            names=LEVEL_NAMES)
        freq = '{:d}S'.format(int(sample_period))
        verbose = kwargs.get('verbose')

        # Check for empty sections
        sections = [section for section in sections if section]
        if not sections:
            print("No sections to load.")
            yield pd.DataFrame(columns=columns)
            return

        # Loop through each section to load
        for section in split_timeframes(sections, duration_threshold):
            kwargs['sections'] = [section]
            start = normalise_timestamp(section.start, freq)
            tz = None if start.tz is None else start.tz.zone
            index = pd.date_range(
                start.tz_localize(None), section.end.tz_localize(None), tz=tz,
                closed='left', freq=freq)
            chunk = combine_chunks_from_generators(
                index, columns, self.meters, kwargs)
            yield chunk

    def _convert_physical_quantity_and_ac_type_to_cols(self, **kwargs):
        all_columns = set()
        kwargs = deepcopy(kwargs)
        for meter in self.meters:
            kwargs_copy = deepcopy(kwargs)
            new_kwargs = meter._convert_physical_quantity_and_ac_type_to_cols(**kwargs_copy)
            columns = new_kwargs.get('columns', [])
            for col in columns:
                all_columns.add(col)
        kwargs['columns'] = list(all_columns)
        return kwargs

    def _meter_generators(self, **kwargs):
        """Returns (list of identifiers, list of generators)."""
        generators = []
        identifiers = []
        for meter in self.meters:
            kwargs_copy = deepcopy(kwargs)
            generator = meter.load(**kwargs_copy)
            generators.append(generator)
            identifiers.append(meter.identifier)

        return identifiers, generators

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
                count[timestamp] += 1
        sim_switches = pd.Series(count)
        # Should be 2 or more appliances changing state at the same time
        sim_switches = sim_switches[sim_switches >= 2]
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
        """Returns new MeterGroup."""
        meters = list(nodes_adjacent_to_root(self.wiring_graph()))
        return MeterGroup(meters)

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
        self._check_kwargs_for_full_results_and_sections(load_kwargs)
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
            print_on_line("\rCalculating", func, "for", meter.identifier, "...   ")
            single_stat = getattr(meter, func)(full_results=full_results,
                                               **load_kwargs)
            collected_stats.append(single_stat)
            if (full_results and len(self.meters) > 1 and
                    not meter.store.all_sections_smaller_than_chunksize):
                warnings.warn("at least one section requested from '{}' required"
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
        self._check_kwargs_for_full_results_and_sections(load_kwargs)
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

    def _check_kwargs_for_full_results_and_sections(self, load_kwargs):
        if (load_kwargs.get('full_results')
                and 'sections' not in load_kwargs
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
                warnings.warn("As a quick implementation we only get Good Sections from"
                     " the first meter in the meter group.  We should really"
                     " return the intersection of the good sections for all"
                     " meters.  This will be fixed...")
            return self.meters[0].good_sections(**kwargs)
        else:
            return []

    def dataframe_of_meters(self, **kwargs):
        """
        Parameters
        ----------
        sample_period : int or float, optional
            Number of seconds to use as sample period when reindexing meters.
            If not specified then will use the max of all meters' sample_periods.
        resample : bool, defaults to True
            If True then resample to `sample_period`.
        **kwargs : 
            any other key word arguments to pass to `self.store.load()` including:
        ac_type : string, defaults to 'best'
        physical_quantity: string, defaults to 'power'

        Returns
        -------
        DataFrame
            Each column is a meter.
        """
        kwargs.setdefault('sample_period', self.sample_period())
        kwargs.setdefault('ac_type', 'best')
        kwargs.setdefault('physical_quantity', 'power')
        identifiers, generators = self._meter_generators(**kwargs)
        segments = []
        while True:
            chunks = []
            ids = []
            for meter_id, generator in zip(identifiers, generators):
                try:
                    chunk_from_next_meter = next(generator)
                except StopIteration:
                    continue

                if not chunk_from_next_meter.empty:
                    ids.append(meter_id)
                    chunks.append(chunk_from_next_meter.sum(axis=1))

            if chunks:
                df = pd.concat(chunks, axis=1)
                df.columns = ids
                segments.append(df)
            else:
                break

        if segments:
            return pd.concat(segments)
        else:
            return pd.DataFrame(columns=self.identifier.meters)

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
        meter_identifiers = list(self.identifier.meters)
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
        meter_identifiers = list(self.identifier.meters)
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
        float [0,1] or NaN if mains total_energy == 0
        """
        print("Running MeterGroup.proportion_of_energy_submetered...")
        mains = self.mains()
        downstream_meters = self.meters_directly_downstream_of_mains()
        proportion = 0.0
        verbose = loader_kwargs.get('verbose')
        all_nan = True
        for m in downstream_meters.meters:
            if verbose:
                print("Calculating proportion for", m)
            prop = m.proportion_of_energy(mains, **loader_kwargs)
            if not np.isnan(prop):
                proportion += prop
                all_nan = False
            if verbose:
                print("   {:.2%}".format(prop))
        
        if all_nan:
            proportion = np.NaN
        return proportion

    def available_ac_types(self, physical_quantity):
        """Returns set of all available alternating current types for a 
        specific physical quantity.

        Parameters
        ----------
        physical_quantity : str or list of strings

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

    def energy_per_meter(self, per_period=None, mains=None, 
                         use_meter_labels=False, **load_kwargs):
        """Returns pd.DataFrame where columns is meter.identifier and 
        each value is total energy.  Index is AC types.

        Does not care about wiring hierarchy.  Does not attempt to ensure all 
        channels share the same time sections.

        Parameters
        ----------
        per_period : None or offset alias
            If None then returns absolute energy used per meter.
            If a Pandas offset alias (e.g. 'D' for 'daily') then 
            will return the average energy per period.
        ac_type : None or str
            e.g. 'active' or 'best'.  Defaults to 'best'.
        use_meter_labels : bool
            If True then columns will be human-friendly meter labels.
            If False then columns will be ElecMeterIDs or MeterGroupIDs
        mains : None or MeterGroup or ElecMeter
            If None then will return DataFrame without remainder.
            If not None then will return a Series including a 'remainder'
            row which will be `mains.total_energy() - energy_per_meter.sum()`
            and an attempt will be made to use the correct AC_TYPE.

        Returns
        -------
        pd.DataFrame if mains is None else a pd.Series
        """
        meter_identifiers = list(self.identifier.meters)
        energy_per_meter = pd.DataFrame(columns=meter_identifiers, index=AC_TYPES)
        n_meters = len(self.meters)
        load_kwargs.setdefault('ac_type', 'best')
        for i, meter in enumerate(self.meters):
            print('\r{:d}/{:d} {}'.format(i+1, n_meters, meter), end='')
            stdout.flush()
            if per_period is None:
                meter_energy = meter.total_energy(**load_kwargs)
            else:
                load_kwargs.setdefault('use_uptime', False)
                meter_energy = meter.average_energy_per_period(
                    offset_alias=per_period, **load_kwargs)
            energy_per_meter[meter.identifier] = meter_energy

        energy_per_meters = energy_per_meter.dropna(how='all')

        if use_meter_labels:
            energy_per_meter.columns = self.get_labels(energy_per_meter.columns)

        if mains is not None:
            energy_per_meter = self._energy_per_meter_with_remainder(
                energy_per_meter, mains, per_period, **load_kwargs)

        return energy_per_meter

    def _energy_per_meter_with_remainder(self, energy_per_meter,
                                         mains, per_period, **kwargs):
        ac_types = energy_per_meter.keys()
        energy_per_meter = energy_per_meter.sum() # Collapse AC_TYPEs into Series

        # Find most common ac_type in energy_per_meter:
        most_common_ac_type = most_common(ac_types)
        mains_ac_types = mains.available_ac_types(
            ['power', 'energy', 'cumulative energy'])
        if most_common_ac_type in mains_ac_types:
            mains_ac_type = most_common_ac_type
        else:
            mains_ac_type = 'best'

        # Get mains energy_per_meter
        kwargs['ac_type'] = mains_ac_type
        if per_period is None:
            mains_energy = mains.total_energy(**kwargs)
        else:
            mains_energy = mains.average_energy_per_period(
                offset_alias=per_period, **kwargs)
        mains_energy = mains_energy[mains_energy.keys()[0]]

        # Calculate remainder
        energy_per_meter['Remainder'] = mains_energy - energy_per_meter.sum()
        energy_per_meter.sort_values(inplace=True, ascending=False)
        return energy_per_meter

    def fraction_per_meter(self, **load_kwargs):
        """Fraction of energy per meter.

        Return pd.Series.  Index is meter.instance.  
        Each value is a float in the range [0,1].
        """
        energy_per_meter = self.energy_per_meter(**load_kwargs).max()
        total_energy = energy_per_meter.sum()
        return energy_per_meter / total_energy

    def proportion_of_upstream_total_per_meter(self, **load_kwargs):
        prop_per_meter = pd.Series(index=self.identifier.meters)
        n_meters = len(self.meters)
        for i, meter in enumerate(self.meters):
            proportion = meter.proportion_of_upstream(**load_kwargs)
            print('\r{:d}/{:d} {} = {:.3f}'
                  .format(i+1, n_meters, meter, proportion), end='')
            stdout.flush()
            prop_per_meter[meter.identifier] = proportion
        prop_per_meter.sort_values(inplace=True, ascending=False)
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

        mains = self.mains()
        good_sections = self.mains().good_sections()
        sample_period = mains.sample_period()
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
    # def on_off_events(self, minimum_state_duration):
    #     raise NotImplementedError

    def select_top_k(self, k=5, by="energy", asc=False, group_remainder=False, **kwargs):
        """Only select the top K meters, according to energy.

        Functions on the entire MeterGroup.  So if you mean to select
        the top K from only the submeters, please do something like
        this:

        elec.submeters().select_top_k()

        Parameters
        ----------
        k : int, optional, defaults to 5
        by: string, optional, defaults to energy
            Can select top k by:
                * energy
                * entropy
        asc: bool, optional, defaults to False
            By default top_k is in descending order. To select top_k
            by ascending order, use asc=True
        group_remainder : bool, optional, defaults to False
            If True then place all remaining meters into a 
            nested metergroup.
        **kwargs : key word arguments to pass to load()

        Returns
        -------
        MeterGroup
        """
        function_map = {'energy': self.fraction_per_meter, 'entropy': self.entropy_per_meter}
        top_k_series = function_map[by](**kwargs)
        top_k_series.sort_values(inplace=True, ascending=asc)
        top_k_elec_meter_ids = list(top_k_series[:k].index)
        
        #TODO: investigate the root cause for missing namedtuple type, remove this workaround
        if top_k_elec_meter_ids and type(top_k_elec_meter_ids[0]) is tuple and len(top_k_elec_meter_ids[0]) == 3:
            top_k_elec_meter_ids = [ElecMeterID(*key) for key in top_k_elec_meter_ids]
        
        top_k_metergroup = self.from_list(top_k_elec_meter_ids)

        if group_remainder:
            remainder_ids = list(top_k_series[k:].index)
            if remainder_ids and type(remainder_ids[0]) is tuple and len(remainder_ids[0]) == 3:
                remainder_ids = [ElecMeterID(*key) for key in remainder_ids]
            
            remainder_metergroup = self.from_list(remainder_ids)
            remainder_metergroup.name = 'others'
            top_k_metergroup.meters.append(remainder_metergroup)
        return top_k_metergroup

    def groupby(self, key, use_appliance_metadata=True, **kwargs):
        """
        e.g. groupby('category')

        Returns
        -------
        MeterGroup of nested MeterGroups: one per group
        """
        if not use_appliance_metadata:
            raise NotImplementedError()

        values = self.values_for_appliance_metadata_key(key)
        groups = []
        for value in values:
            group = self.select_using_appliances(**{key: value})
            group.name = value
            groups.append(group)

        return MeterGroup(groups)

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

    def plot(self, kind='separate lines', **kwargs):
        """
        Parameters
        ----------
        width : int, optional
            Number of points on the x axis required
        ax : matplotlib.axes, optional
        plot_legend : boolean, optional
            Defaults to True.  Set to False to not plot legend.
        kind : {'separate lines', 'sum', 'area', 'snakey', 'energy bar'}
        timeframe : nilmtk.TimeFrame, optional
            Defaults to self.get_timeframe()
        """
        # Load data and plot each meter
        function_map = {
            'separate lines': self._plot_separate_lines,
            'sum': super(MeterGroup, self).plot,
            'area': self._plot_area,
            'sankey': self._plot_sankey,
            'energy bar': self._plot_energy_bar
        }
        try:
            ax = function_map[kind](**kwargs)
        except KeyError:
            raise ValueError("'{}' not a valid setting for 'kind' parameter."
                             .format(kind))
        return ax

    def _plot_separate_lines(self, ax=None, plot_legend=True, **kwargs):
        for meter in self.meters:
            if isinstance(meter, MeterGroup):
                ax = meter.plot(ax=ax, plot_legend=False, kind='sum', **kwargs)
            else:
                ax = meter.plot(ax=ax, plot_legend=False, **kwargs)
        if plot_legend:
            plt.legend()
        return ax

    def _plot_sankey(self):
        """
        Computes parameters like fraction of energy, labels and orientations
        from elecmeter object and calls matplotlib.sankey function to plot
        the data

        """

        # Use fraction_per_meter() function to get energy fraction values of
        # submeters
        fraction_per_meter = self.submeters().fraction_per_meter()

        # Calculate total number of devices
        total_devices = len(fraction_per_meter)

        # Define a list of energy_ratio
        energy_ratio = [i for i in fraction_per_meter]

        #  Define a list of labels of submeters
        labels = []

        for i in range(total_devices):
            # Get labels of appliances
            labels.append(self.submeters().meters[i].appliances[0].type['type'])

        # Append mains to last of both lists
        energy_ratio.append(-1)
        labels.append('mains')

        # Define orientations for plot
        orientations = np.ones(total_devices + 1)
        orientations[-1] = 0
        for i in range(int(total_devices / 2)):
            orientations[i] = -1

        # Plot
        Sankey(flows=energy_ratio, labels=labels,
            orientations=orientations).finish()
        plt.title("Sankey Diagram")
        

    def _plot_area(self, ax=None, timeframe=None, pretty_labels=True, unit='W',
                   label_kwargs=None, plot_kwargs=None, threshold=None,
                   **load_kwargs):
        """
        Parameters
        ----------
        plot_kwargs : dict of key word arguments for DataFrame.plot()
        unit : {kW or W}
        threshold : float or None
           if set to a float then any measured value under this threshold
           will be set to 0.

        Returns
        -------
        ax, dataframe
        """
        # Get start and end times for the plot
        timeframe = self.get_timeframe() if timeframe is None else timeframe
        if not timeframe:
            return ax

        load_kwargs['sections'] = [timeframe]
        load_kwargs = self._set_sample_period(timeframe, **load_kwargs)
        df = self.dataframe_of_meters(**load_kwargs)

        if threshold is not None:
            df[df <= threshold] = 0
            
        if unit == 'kW':
            df /= 1000

        if plot_kwargs is None:
            plot_kwargs = {}
        df.columns = self.get_labels(df.columns, pretty=pretty_labels)
        # Set a tiny linewidth otherwise we get lines even if power is zero
        # and this looks ugly when drawn above other lines.
        plot_kwargs.setdefault('linewidth', 0.0001)
        ax = df.plot(kind='area', **plot_kwargs)
        ax.set_ylabel("Power ({:s})".format(unit))
        return ax, df

    def plot_when_on(self, **load_kwargs):
        meter_identifiers = list(self.identifier.meters)
        fig, ax = plt.subplots()
        for i, meter in enumerate(self.meters):
            id_meter = meter.identifier
            for chunk_when_on in meter.when_on(**load_kwargs):
                series_to_plot = chunk_when_on[chunk_when_on==True]
                if len(series_to_plot.index):
                    (series_to_plot+i-1).plot(ax=ax, style='k.')
        labels = self.get_labels(meter_identifiers)
        plt.yticks(range(len(self.meters)), labels)
        plt.ylim((-0.5, len(self.meters)+0.5))
        return ax

    def plot_good_sections(self, ax=None, label_func='instance', 
                           include_disabled_meters=True, load_kwargs=None,
                           **plot_kwargs):
        """
        Parameters
        ----------
        label_func : str or None
            e.g. 'instance' (default) or 'label'
            if None then no labels will be produced.
        include_disabled_meters : bool
        """
        if ax is None:
            ax = plt.gca()

        if load_kwargs is None:
            load_kwargs = {}

        # Prepare list of meters
        if include_disabled_meters:
            meters = self.all_meters()
        else:
            meters = self.meters
        meters = copy(meters)
        meters.sort(key=meter_sorting_key, reverse=True)
        n = len(meters)

        labels = []
        for i, meter in enumerate(meters):
            good_sections = meter.good_sections(**load_kwargs)
            ax = good_sections.plot(ax=ax, y=i, **plot_kwargs)
            del good_sections
            if label_func:
                labels.append(getattr(meter, label_func)())

        # Just end numbers
        if label_func is None:
            labels = [n] + ([''] * (n-1))

        # Y tick formatting
        ax.set_yticks(np.arange(0, n) + 0.5)
        def y_formatter(y, pos):
            try:
                label = labels[int(y)]
            except IndexError:
                label = ''
            return label
        ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
        ax.set_ylim([0, n])

        return ax

    def _plot_energy_bar(self, ax=None, mains=None):
        """Plot a stacked bar of the energy per meter, in order.

        Parameters
        ----------
        ax : matplotlib axes
        mains : MeterGroup or ElecMeter, optional
            Used to calculate Remainder.

        Returns
        -------
        ax
        """
        energy = self.energy_per_meter(mains=mains, per_period='D',
                                        use_meter_labels=True)

        energy.sort(ascending=False)

        # Plot
        ax = pd.DataFrame(energy).T.plot(kind='bar', stacked=True, grid=True,
                                         edgecolor="none", legend=False, width=2)
        ax.set_xticks([])
        ax.set_ylabel('kWh\nper\nday', rotation=0, ha='center', va='center', 
                      labelpad=15)

        cumsum = energy.cumsum()
        text_ys = cumsum - (cumsum.diff().fillna(energy['Remainder']) / 2)
        for kwh, (label, y) in zip(energy.values, text_ys.items()):
            label += " ({:.2f})".format(kwh)
            ax.annotate(label, (0, y), color='white', size=8,
                        horizontalalignment='center', 
                        verticalalignment='center')

        return ax

    def plot_multiple(self, axes, meter_keys, plot_func, 
                      kwargs_per_meter=None, pretty_label=True, **kwargs):
        """Create multiple subplots.

        Parameters
        -----------
        axes : list of matplotlib axes objects.
            e.g. created using `fix, axes = plt.subplots()`
        meter_keys : list of keys for identifying ElecMeters or MeterGroups. 
            e.g. ['fridge', 'kettle', 4, MeterGroupID, ElecMeterID].  
            Each element is anything that MeterGroup.__getitem__() accepts.
        plot_func : string
            Name of function from ElecMeter or Electric or MeterGroup
            e.g. `plot_power_histogram`
        kwargs_per_meter : dict
            Provide key word arguments for the plot_func for each meter.
            each key is a parameter name for plot_func
            each value is a list (same length as `meters`) for specifying a value for
            this parameter for each meter. 
            e.g. {'range': [(0,100), (0,200)]}
        pretty_label : bool
        **kwargs : any key word arguments to pass the same values to the
           plot func for every meter.

        Returns
        -------
        axes (flattened into a 1D list)
        """
        axes = flatten_2d_list(axes)
        if len(axes) != len(meter_keys):
            raise ValueError("`axes` and `meters` must be of equal length.")

        if kwargs_per_meter is None:
            kwargs_per_meter = {}

        meters = [self[meter_key] for meter_key in meter_keys]
        for i, (ax, meter) in enumerate(zip(axes, meters)):
            kwargs_copy = deepcopy(kwargs)
            for parameter, arguments in kwargs_per_meter.items():
                kwargs_copy[parameter] = arguments[i]
            getattr(meter, plot_func)(ax=ax, **kwargs_copy)
            ax.set_title(meter.label(pretty=pretty_label))

        return axes

    def sort_meters(self):
        """Sorts meters by instance."""
        self.meters.sort(key=meter_sorting_key)

    def label(self, **kwargs):
        """
        Returns
        -------
        string : A label listing all the appliance types.
        """
        if self.name:
            label = self.name
            if kwargs.get('pretty'):
                label = capitalise_first_letter(label)
            return label
        return ", ".join(set([meter.label(**kwargs) for meter in self.meters]))

    def clear_cache(self):
        """Clear cache on all meters in this MeterGroup."""
        for meter in self.meters:
            meter.clear_cache()
        
    def correlation_of_sum_of_submeters_with_mains(self, **load_kwargs):
        print("Running MeterGroup.correlation_of_sum_of_submeters_with_mains...")
        submeters = self.meters_directly_downstream_of_mains()
        return self.mains().correlation(submeters, **load_kwargs)

    def all_meters(self):
        """Returns a list of self.meters + self.disabled_meters."""
        return self.meters + self.disabled_meters

    def describe(self, compute_expensive_stats=True, **kwargs):
        """Returns pd.Series describing this MeterGroup."""
        series = pd.Series()

        all_meters = self.all_meters()
        series['total_n_meters'] = len(all_meters)
        site_meters = [m for m in all_meters if m.is_site_meter()]
        series['total_n_site_meters'] = len(site_meters)
        if compute_expensive_stats:
            series['correlation_of_sum_of_submeters_with_mains'] = (
                self.correlation_of_sum_of_submeters_with_mains(**kwargs))
            series['proportion_of_energy_submetered'] = (
                self.proportion_of_energy_submetered(**kwargs))
            dropout_rates = self._collect_stats_on_all_meters(
                kwargs, 'dropout_rate', False)
            dropout_rates = np.array(dropout_rates)
            series['dropout_rates_ignoring_gaps'] = (
                "min={}, mean={}, max={}".format(
                    dropout_rates.min(), 
                    dropout_rates.mean(), 
                    dropout_rates.max()))

        series['mains_sample_period'] = self.mains().sample_period()
        series['submeter_sample_period'] = self.submeters().sample_period()
        timeframe = self.get_timeframe()
        series['timeframe'] = "start={}, end={}".format(timeframe.start, timeframe.end)
        series['total_duration'] = str(timeframe.timedelta)
        mains_uptime = self.mains().uptime(**kwargs)
        series['mains_uptime'] = str(mains_uptime)
        try:
            series['proportion_uptime'] = (mains_uptime.total_seconds() /
                                           timeframe.timedelta.total_seconds())
        except ZeroDivisionError:
            series['proportion_uptime'] = np.NaN
        series['average_mains_energy_per_day'] = self.mains().average_energy_per_period()

        return series


def replace_dataset(identifier, dataset):
    """
    Parameters
    ----------
    identifier : ElecMeterID or MeterGroupID

    Returns
    -------
    ElecMeterID or MeterGroupID with dataset replaced with `dataset`
    """
    if isinstance(identifier, MeterGroupID):
        new_meter_ids = [replace_dataset(id, dataset) for id in identifier.meters]
        new_id = MeterGroupID(meters=tuple(new_meter_ids))
    elif isinstance(identifier, ElecMeterID):
        new_id = identifier._replace(dataset=dataset)
    else:
        raise TypeError()

    return new_id


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
        slave_identifier = replace_dataset(master_meter.identifier, slave.dataset())
        slave_meter = slave[slave_identifier]
        zipped.append((master_meter, slave_meter))
    return zipped


def combine_chunks_from_generators(index, columns, meters, kwargs):
    """Combines chunks into a single DataFrame.

    Adds or averages columns, depending on whether each column is in
    PHYSICAL_QUANTITIES_TO_AVERAGE.

    Returns
    -------
    DataFrame
    """
    # Regarding columns (e.g. voltage) that we need to average:
    # The approach is that we first add everything together
    # in the first for-loop, whilst also keeping a 
    # `columns_to_average_counter` DataFrame
    # which tells us what to divide by in order to compute the 
    # mean for PHYSICAL_QUANTITIES_TO_AVERAGE.

    # Regarding doing an in-place addition:
    # We convert out cumulator dataframe to a numpy matrix.
    # This allows us to use np.add to do an in-place add.
    # If we didn't do this then we'd get horrible memory fragmentation.
    # See http://stackoverflow.com/a/27526721/732596

    DTYPE = np.float32
    cumulator = pd.DataFrame(np.NaN, index=index, columns=columns, dtype=DTYPE)
    cumulator_arr = cumulator.values
    columns_to_average_counter = pd.DataFrame(dtype=np.uint16)
    timeframe = None

    # Go through each generator to try sum values together
    for meter in meters:
        print_on_line("\rLoading data for meter", meter.identifier, "    ")
        kwargs_copy = deepcopy(kwargs)
        generator = meter.load(**kwargs_copy)
        try:
            chunk_from_next_meter = next(generator)
        except StopIteration:
            continue

        del generator
        del kwargs_copy
        gc.collect()

        if chunk_from_next_meter.empty or not chunk_from_next_meter.timeframe:
            continue

        if timeframe is None:
            timeframe = chunk_from_next_meter.timeframe
        else:
            timeframe = timeframe.union(chunk_from_next_meter.timeframe)

        # Add (in-place)
        for i, column_name in enumerate(columns):
            try:
                column = chunk_from_next_meter[column_name]
            except KeyError:
                continue

            aligned = column.reindex(index, copy=False).values
            del column
            cumulator_col = cumulator_arr[:,i]
            where_both_are_nan = np.isnan(cumulator_col) & np.isnan(aligned)
            np.nansum([cumulator_col, aligned], axis=0, out=cumulator_col, 
                      dtype=DTYPE)
            cumulator_col[where_both_are_nan] = np.NaN
            del aligned
            del where_both_are_nan
            gc.collect()

        # Update columns_to_average_counter - this is necessary so we do not
        # add up columns like 'voltage' which should be averaged.
        physical_quantities = chunk_from_next_meter.columns.get_level_values('physical_quantity')
        columns_to_average = (set(PHYSICAL_QUANTITIES_TO_AVERAGE)
                              .intersection(physical_quantities))
        if columns_to_average:
            counter_increment = pd.DataFrame(1, columns=columns_to_average, 
                                             dtype=np.uint16,
                                             index=chunk_from_next_meter.index)
            columns_to_average_counter = columns_to_average_counter.add(
                counter_increment, fill_value=0)
            del counter_increment

        del chunk_from_next_meter
        gc.collect()

    del cumulator_arr
    gc.collect()

    # Create mean values by dividing any columns which need dividing
    for column in columns_to_average_counter:
        cumulator[column] /= columns_to_average_counter[column]

    del columns_to_average_counter
    gc.collect()
    print()
    print("Done loading data all meters for this chunk.")
    cumulator.timeframe = timeframe
    return cumulator


meter_sorting_key = lambda meter: meter.instance()
