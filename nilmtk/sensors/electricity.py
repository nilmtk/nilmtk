from collections import namedtuple
import copy

Measurement = namedtuple('Measurement', ['physical_quantity', 'type'])
"""
physical_quantity : string
    One of: {power, energy, voltage}
type : string
    One of: {active, reactive, apparent, ''}
"""

ApplianceName = namedtuple('ApplianceName', ['name', 'instance'])
MainsName = namedtuple('MainsName', ['split', 'meter'])
CircuitName = namedtuple('CircuitName', ['name', 'split', 'meter'])


class Electricity(object):

    """Store and process electricity for a building.

    Attributes
    ----------

    mains : dict of DataFrames, optional
        The power measurements taken from the level furthest upstream.
        Each key is a MainsName namedtuple with keys:
        * `split` is the phase or split.  Indexed from 1.
        * `meter` is the numeric ID of the meter. Indexed from 1.

        Each value is a DataFrame of shape (n_samples, n_features) 
        where each column name is a Measurement namedtuple (please
        definition of Measurement at the top of this file for a
        description of what can go into a Measurement namedtuple).

        DataFrame.index is a timezone-aware pd.DateTimeIndex.
        Power values are of type np.float32.
        DataFrame.name should be identical to the `mains` dict key which 
        maps to this DataFrame.

        For example, if we had a dataset recorded in the UK where the home has
        only a single phase supply but uses two separate meters, the first of
        which measures active and reactive power; the second of which measures only
        active, then we'd use:

        `mains = {MainsName(split=1, meter=1): 
                      DataFrame(columns=[Measurement('power','active'),
                                         Measurement('power','reactive')]),
                  MainsName(split=1, meter=2):
                      DataFrame(columns=[Measurement('power','active')])
                 }`

    circuits : dict of DataFrames, optional
        The power measurements taken from midstream.
        Each key is a CircuitName namedtuple with fields:

        * `name` is the standard name for this circuit.
        * `split` is the index for this circuit.  Indexed from 1.
        * `meter` is the numeric ID of the meter. Indexed from 1.
          Indexes into the `meters` dict.

        Each value is a DataFrame of shape (n_samples, n_features) 
        where each column name is a Measurement namedtuple (please
        definition of Measurement at the top of this file for a
        description of what can go into a Measurement namedtuple).

        DataFrame.index is a timezone-aware pd.DateTimeIndex.
        Power values are of type np.float32.
        DataFrame.name should be identical to the `circuits` dict key which 
        maps to this DataFrame.

        For example: 
        `circuits = {CircuitName(circuit='lighting', split=1):
                         DataFrame(columns=[Measurement('power','active')])}`

    appliances : dict of DataFrames, optional
        Each key is an ApplianceName namedtuple with fields:
        * `name` is a standard appliance name.  For a list of valid
           names, see nilmtk/docs/standard_names/appliances.txt
        * `instance` is the index for that appliance within this building.
           Indexed from 1.

        For example, if a house has two TVs, whose power is recorded separately,
        then use these two different keys: `('tv', 1)` and `('tv', 2)`

        If multiple appliances are monitored on one channel (e.g. tv + dvd)
        then use a tuple of appliances as the key, e.g.:

        `(('tv', 1), ('dvd player', 1))`
        
        Each value of the `appliances` dict is a DataFrame of 
        shape (n_samples, n_features) where each column name is either:

        * a Measurement namedtuple (please definition of Measurement at the top
          of this file for a description of what can go into 
          a Measurement namedtuple).
        * Or 'state' (where 0 is 'off'). This is used when the ground-truth of
          the appliance state is known; for example in the BLUEd dataset.

        DataFrame.index is a timezone-aware pd.DateTimeIndex.
        Power values are of type np.float32; `state` values are np.int32
        DataFrame.name should be identical to the `appliances` dict key which 
        maps to this DataFrame.

    appliance_estimates : Panel (3D matrix), optional
        Output from the NILM disaggregation algorithm.
        The index is a timezone-aware pd.DateTimeIndex
        The first two dimensions are time and ApplianceName.
        The third dimension describes, for each appliance and for each time:
        * `power` : np.float32. Estimated power consumption in Watts.
        * `state` : np.int32, optional.
        * `confidence` : np.float32 [0,1], optional.
        * `power_prob_dist` : object describing the probability dist, optional
        * `state_prob_dist` : object describing the probability dist, optional

    metadata : dict, optional

        nominal_mains_voltage : np.float32, optional

        appliances_in_each_room : dict, optional
            Each key is a (<room name>, <room instance>) tuple
            (as used in this `Building.rooms`).
            Each value is a list of (<appliance name>, <instance>) tuples
            e.g. `{('livingroom', 1): [('tv', 1), ('dvd', 2)]}`

        wiring : networkx.DiGraph
            Nodes are ApplianceNames or CircuitNames or MainsNames.
            Edges describe the wiring between mains, circuits and appliances.
            Edge direction indicates the flow of energy.  i.e. edges point
            towards loads.

        meters : dict, optional
            Maps from a tuple (<meter manufacturer>, <model>) to a list of 
            all the channels which use that type of meter.  Types of meters
            are described in `docs/standard_names/meters.json`.  e.g.:
            `{
               ('Current Cost', 'EnviR') : 
                 [
                    MainsName(split=1, meter=1, measurement='apparent'),
                    ApplianceName(name='boiler', instance=1, measurement='apparent')
                 ]
              }`

        appliances : dict of dicts, optional
            Metadata describing each appliance.
            Each key is an ApplianceName(<appliance name>, <instance>) tuple.
            Each value is dict describing metadata for that appliance.
            The permitted fields and values
            for each appliance name are described in `appliances.json`.  e.g.

            `{('tv', 1): 
                {'display': 'lcd', 
                 'backlight': 'led'
                 'screen size in inches': 42,
                 'year of manufacture': 2001,
                 'last time seen active': '3/4/2012'
                }
            }`
    """

    def __init__(self):
        self.mains = {}
        self.circuits = {}
        self.appliances = {}
        self.appliance_estimates = None
        self.metadata = {}

    def get_appliance(self, appliance_name, measurement="all"):
        """
        Arguments
        ---------
        appliance_name : string
        measurement : string or list of strings, optional
            apparent | active | reactive | voltage | all

        Returns
        -------
        appliance_data : DataFrame
        """
        raise NotImplementedError

    def count_appliances(self, appliance_name):
        """
        Returns
        -------
        n_appliances : int
        """
        raise NotImplementedError

    def get_vampire_power(self):
        raise NotImplementedError

    def get_diff_between_aggregate_and_appliances(self):
        raise NotImplementedError

    def plot_appliance_activity(self, source):
        """Plot a compact representation of all appliance activity."""
        raise NotImplementedError

    def __str__(self):
        return ""

    def to_json(self):
        representation = copy.copy(self.metadata)
        representation["mains"] = ""
        representation["appliances"] = ""
        representation["circuits"] = ""
        return json.dumps(representation)
