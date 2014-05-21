from __future__ import print_function, division
import networkx as nx
from .electricitymeter import ElecMeter
from .datastore import join_key

class MeterGroup(object):
    """A group of ElecMeter objects.

    Implements many of the same methods as Meter.
    
    Attributes
    ----------
    meters : list of ElectricityMeters
    """
    def __init__(self, meters=None):
        self.meters = [] if meters is None else list(meters)

    def load(self, store, elec_meters):
        """
        Parameters
        ----------
        store : nilmtk.DataStore
        elec_meters : list of dicts
            metadata for each ElecMeter
        """
        ElecMeter.load_meter_devices(store)
        for meter_i, meter_metadata_dict in elec_meters.iteritems():
            meter = ElecMeter(store, meter_metadata_dict, meter_i)
            self.meters.append(meter)

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

    def __getitem__(self, key):
        """Get a single meter.
        
        Three formats for `key` are accepted:
        * ['toaster']    - retrieves toaster instance 1
        * ['toaster', 2] - retrieves toaster instance 2
        * [{'dataset': 'redd', 'building': 3, 'type': 'toaster', 'instance': 2}]

        Returns
        -------
        Meter
        """
        if isinstance(key, str):
            # default to get first meter
            return self[(key, 1)]
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
        else:
            raise TypeError()

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

        Plans for the future (not implemented yet!)
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
        
    def groupby(self, **kwargs):
        """
        e.g. groupby('category')
        
        Returns
        -------
        A dict of MeterGroup objects e.g.:
          {'cold': MeterGroup, 'hot': MeterGroup}
        """
        raise NotImplementedError
        
    def __eq__(self, other):
        if isinstance(other, MeterGroup):
            return other.meters == self.meters
        else:
            return False

    def wiring_graph(self):
        """Returns a networkx.DiGraph of connections between meters.

        The root will be a Mains object.
        """
        wiring_graph = nx.DiGraph()
        for meter in self.meters:
            try:
                upstream_meter = meter.upstream_meter
            except ValueError:
                pass # no upstream meter
            else:
                wiring_graph.add_edge(meter.upstream_meter, meter)
        return wiring_graph

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
                
    def total_energy(self):
        # self.get_unique_upstream_meters()
        # adds energy on a meter-by-meter basis
        raise NotImplementedError
    
    def on_off_events(self, minimum_state_duration):
        raise NotImplementedError
    
    def top_k(self, k=5):
        """Return new MeterGroup?"""
        self.itemised_energy().ix[:k]
    
    def itemised_energy(self):
        """Needs to do it per-meter???  Return sorted.
        'kitchen lights': 234.5
        ['hall lights, bedroom lights'] : 32.1 
        need to subtract kitchen lights energy from lighting circuit!
        """ 
        # keys could be actual Appliance / MeterGroup objects?
        # e.g. when we want to select top_k Meters.
        raise NotImplementedError
        
    def proportion_above(self, threshold_proportion):
        """Return new MeterGroup with all meters whose proportion of
        energy usage is above threshold"""
        raise NotImplementedError
        
    def itemised_proportions(self):
        """Proportion of energy per meter. Return sorted."""
        raise NotImplementedError
    
    def power_series(self):
        # Get all upstream meters. Add series.  Return generator of series.
        # What happens if indicies don't match?  Automatically re-sample?  Or down-sample?
        # Probably best to raise exception and make user pre-process???  How?
        # lighting.resample('6S').power_series() ???? or
        # lighting.preprocessing = [Resample('6S')]
        # lighting.power_series()
        raise NotImplementedError
            
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
