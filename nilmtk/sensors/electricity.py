from collections import namedtuple
import copy
import json

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
DualSupply = namedtuple('DualSupply', ['measurement', 'supply'])


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
        Power measurements taken from the furthest downstream in the dataset.
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
        * a DualSupply namedtuple with fields:
          * measurement : Measurement namedtuple
          * supply : int. Index of supply. Start from 1. Does not have to map
            directly to the index used to number the mains splits if this 
            information is not known.
          DualSupply is used for appliances like American ovens which are
          are single appliances but pull power from both "splits".
        * Or 'state' (where 0 is 'off'). This is used when the ground-truth of
          the appliance state is known; for example in the BLUEd dataset.

        DataFrame.index is a timezone-aware pd.DateTimeIndex.
        Power values are of type np.float32; `state` values are np.int32
        DataFrame.name should be identical to the `appliances` dict key which 
        maps to this DataFrame.

        Note that `appliances` always stores measurements from the meters
        in the dataset placed furthest downstream.  So, for example, in the case
        of the REDD dataset where we have measurements of 'mains' and 'circuits',
        these 'circuits' channels are put into `Electricity.appliances` because
        REDD's 'circuits' channels are the furthest downstream.

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
            The nominal mains voltage in volts.

        mains_wiring : networkx.DiGraph
            Nodes are ApplianceNames or CircuitNames or MainsNames.
            Edges describe the power wiring between mains, circuits and 
            appliances.
            Edge direction indicates the flow of energy.  i.e. edges point
            towards loads.

        control_connections : networkx.DiGraph
            Each node is an ApplianceName.
            Edges represent data / audio / video / control connections between
            appliances.  Edges can be single-directional or bi-directional.
            For example, a dvd player would have an edge pointing towards the
            tv.  A wifi router would have bi-directional edges to laptops,
            printers, smart TVs etc.

        appliances : dict of dicts, optional
            Metadata describing each appliance.
            Each key is an ApplianceName(<appliance name>, <instance>) tuple.
            Each value is list of dicts. Each dict describes metadata for that 
            specific appliance.  Multiple dicts are used to express replacing 
            appliances over time (in which case the items should be in 
            chronological order so the last element of the list is always the
            most recent.)  Each dict has 'general' appliance fields (which
            all appliances can have) and fields which are specific to that
            class of appliance.

            General fields
            --------------

            'room': (<room name>, <room instance>) tuple (as used in this `Building.rooms`).
            'meter': (<manufacturer>, <model>) tuple which maps into global Meters DB.

            DualSupply appliances may have a 'supply1' and 'supply2' key which 
            maps to a string describing the main component supplied by that
            supply.  e.g.
            {('washer dryer', 1): 
             {'supply1': 'motor', 'supply2': 'heating element'}
            }

            Appliance-specific fields
            -------------------------
            The permitted fields and values for each appliance name are 
            described in `nilmtk/docs/standard_names/appliances.txt`.  e.g.

            Appliances not directly metered
            -------------------------------
            Appliances which are not directly metered can be listed. For 
            example, if a dataset records the lighting circuit (but
            not each individual ceiling light) then we can specify each
            ceiling light in `metadata['appliances']` and then specify
            the wiring from the lighting circuit to each ceiling light in
            the `metadata['wiring']` graph.

            Example
            -------
            `{('tv', 1): 
                [{'original name in source dataset': 'Television LCD',
                  'display': 'lcd', 
                  'backlight': 'led'
                  'screen size in inches': 42,
                  'year of manufacture': 2001,
                  'active from': '3/4/2012',
                  'active until': '4/5/2013',
                  'quantity installed': 1,
                  'room': ('livingroom', 1),
                  'meter': ('Current Cost', 'IAM')
                }],
             ('lights', 1):
                [{'room': ('kitchen', 1),
                  'placing': 'ceiling',
                  'lamp': 'tungsten',
                  'dimmable': True,
                  'nominal Wattage each': 50,
                  'quantity installed': 10,
                  'meter': ('Current Cost', 'EnviR')
                 }]
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
