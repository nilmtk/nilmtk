class Electricity(object):
    """Store and process electricity for a building.

    Attributes
    ----------

    mains : DataFrame, shape (n_samples, n_features), optional
        The power measurements taken from the level furthest upstream.
        The index is a timezone-aware pd.DateTimeIndex
        Use standard column names of the form
        `mains_<N>_meter_<K>_<measurement>` where:

        * `N` is the phase or split.  Indexed from 1.
        * `K` is the numeric ID of the meter. Indexed from 1.
           Indexes into the `meters` dict.
        * `measurement` is one of `apparent` | `active` | `reactive` | `voltage`
        For example, if we had a dataset recorded in the UK where the home has
        only a single phase supply but uses two separate meters, the first of
        which measures active and reactive; the second of which measures only
        active, then we'd use:
            * `mains_1_meter_1_active`
            * `mains_1_meter_1_reactive`
            * `mains_1_meter_2_active`

    circuits : DataFrame, shape (n_samples, n_features), optional
        The power measurements taken downstream of the mains measurements but
        upstream of the appliances.
        The index is a timezone-aware pd.DateTimeIndex
        Use standard column names of the form `<circuit>_<N>_meter_<K>
        _<measurement>`:
        * `circuit` is the standard name for this circuit.
        * `N` is the index for this circuit.  Indexed from 1.
        * `K` is the numeric ID of the meter. Indexed from 1.
          Indexes into the `meters` dict.
        * `measurement` is one of `apparent` | `active` | `reactive` | `voltage`
        For example: `lighting_1_apparent`

    appliances : dict of DataFrames, optional
        Each key is an appliance name string in the form `<appliance>_<N>`:
        * `appliance` is a standard appliance name.  For a list of valid
           names, see nilmtk/docs/standard_names/appliances.txt
        * `N` is the index for that appliance within this building.
           Indexed from 1.
        Each value is a DataFrame shape (n_samples, n_features) where each
        column name is one of `apparent` | `active` | `reactive` | `voltage`
        and the index is a timezone-aware pd.DateTimeIndex

    appliance_estimates : Panel (3D matrix), optional
        Output from the NILM algorithm.
        The index is a timezone-aware pd.DateTimeIndex
        The first two dimensions are the same as for the appliance DataFrame
        The third dimension describes, for each appliance and for each time:
        * `power` : float. Estimated power consumption in Watts.
        * `state` : int, optional.
        * `confidence` : float [0,1], optional.
        * `power_prob_dist` : object describing the probability dist, optional
        * `state_prob_dist` : object describing the probability dist, optional

    nominal_mains_voltage : float, optional

    map_appliance_to_room : dict, optional
        e.g. {`tv_1`    : `livingroom_1`,
              `fridge_1`: `kitchen_1`}

    wiring : networkx.DiGraph
        Nodes appliance names or circuit names or mains names.
        Edges describe the wiring between mains, circuits and appliances.
        Edge direction indicates the flow of energy.  i.e. edges point
        towards loads.

    meters : list of dicts, optional
        Each element describes a class of meter used to record mains or
        circuit data.  For example:

        [{'manufacturer': 'Current Cost',
          'model': 'EnviR',
          'type': 'CT',
          'accuracy_class': 'C',
          'used_by': [1, 2, 'fridge1', 'kettle1']}]

        `used_by` maps to a list of all the channels which use this
        class of meter.  This list may contain numeric meter IDs (the `K`
        parameter in mains and circuit labels) and/or appliance names.

    appliance_metadata : dict of dicts, optional
        Metadata describing each appliance.  e.g.
        {'television_1': {'type': 'CRT', 'year of manufacture': 2001}}
    """

    def __init__(self):
        self.mains = None
        self.circuits = None
        self.appliances = None
        self.appliance_estimates = None
        self.nominal_mains_voltage = None
        self.map_appliance_to_room = {}
        self.wiring = None
        self.meters = []

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
