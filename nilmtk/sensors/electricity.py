class Electricity(object):
    """Store and process electricity for a building.

    Attributes
    ----------

    mains : DataFrame, shape (n_samples, n_features), optional
        The power measurements taken from the level furthest upstream.
        The index is a timezone-aware pd.DateTimeIndex
        Each column name is a MainsName namedtuple with fields:

        * `split` is the phase or split.  Indexed from 1.
        * `meter` is the numeric ID of the meter. Indexed from 1.
        * `measurement` is one of `apparent` | `active` etc (please see 
          /docs/standard_names/measurements.txt for the full list)

        For example, if we had a dataset recorded in the UK where the home has
        only a single phase supply but uses two separate meters, the first of
        which measures active and reactive; the second of which measures only
        active, then we'd use:
            * `MainsName(split=1, meter=1, measurement='active')`
            * `MainsName(split=1, meter=1, measurement='reactive')`
            * `MainsName(split=1, meter=2, measurement='active')`

    circuits : DataFrame, shape (n_samples, n_features), optional
        The power measurements taken downstream of the mains measurements but
        upstream of the appliances.
        The index is a timezone-aware pd.DateTimeIndex
        Each column name is a CircuitName namedtuple with fields:

        * `circuit` is the standard name for this circuit.
        * `split` is the index for this circuit.  Indexed from 1.
        * `meter` is the numeric ID of the meter. Indexed from 1.
          Indexes into the `meters` dict.
        * `measurement` is one of `apparent` | `active` etc (please see 
          /docs/standard_names/measurements.txt for the full list)

        For example: 
        `CircuitName(circuit='lighting', split=1, 
                     meter=3, measurement='apparent')`

    appliances : dict of DataFrames, optional
        Each key is an ApplianceName namedtuple with fields:
        * `name` is a standard appliance name.  For a list of valid
           names, see nilmtk/docs/standard_names/appliances.txt
        * `instance` is the index for that appliance within this building.
           Indexed from 1.
        * `measurement` is one of `apparent` | `active` etc (please see 
          /docs/standard_names/measurements.txt for the full list)

        For example, if a house has two TVs then use these two column names:
        `('tv', 1, 'active'), ('tv', 2, 'active')`
        Each value is a DataFrame shape (n_samples, n_features) where each
        column name is one of `apparent` | `active` | `reactive` | `voltage`
        and the index is a timezone-aware pd.DateTimeIndex.
        If multiple appliances are monitored on one channel (e.g. tv + dvd)
        then use a tuple of appliances as the column name, e.g.:
        `(('tv', 1, 'active'), ('dvd player', 1, 'active'))`

    appliance_estimates : Panel (3D matrix), optional
        Output from the NILM algorithm.
        The index is a timezone-aware pd.DateTimeIndex
        The first two dimensions are the same as for the `appliances` DataFrame
        The third dimension describes, for each appliance and for each time:
        * `power` : float. Estimated power consumption in Watts.
        * `state` : int, optional.
        * `confidence` : float [0,1], optional.
        * `power_prob_dist` : object describing the probability dist, optional
        * `state_prob_dist` : object describing the probability dist, optional

    nominal_mains_voltage : float, optional

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

    appliance_metadata : dict of dicts, optional
        Metadata describing each appliance.
        Each key is an (<appliance name>, <instance>) tuple.
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
        self.mains = None
        self.circuits = None
        self.appliances = None
        self.appliance_estimates = None
        self.nominal_mains_voltage = None
        self.appliances_in_each = {}
        self.wiring = None
        self.meters = {}
        self.appliance_metadata = {}

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
