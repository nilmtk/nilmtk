from collections import namedtuple

class Building(object):
    """Represent a physical building (e.g. a domestic house).

    Attributes
    ----------

    geographic_coordinates : pair of floats, optional
        (latitude, longitude)

    n_occupants : int, optional
         Max number of occupants.

    rooms : list of strings, optional
        A list of room names. Use standard names for each room

    electric: a collections.namedtuple (a little like a C struct) with fields:

        mains : DataFrame, shape (n_samples, n_features), optional
            The power measurements taken from the furthest upstream.
            The index is a timezone-aware pd.DateTimeIndex
            Use standard column names of the form `mains_<N>_<measurement>`:
            * `N` is the phase or split.  Indexed from 0.
            * `measurement` is one of `apparent` | `active` | `reactive` | `voltage`
            For example: `mains_0_apparent`

        circuits : DataFrame, shape (n_samples, n_features), optional
            The power measurements taken downstream of the mains measurements but
            upstream of the appliances.
            The index is a timezone-aware pd.DateTimeIndex
            Use standard column names of the form `<circuit>_<N>_<measurement>`:
            * `circuit` is the standard name for this circuit.
            * `N` is the index for this circuit.  Indexed from 0.
            * `measurement` is one of `apparent` | `active` | `reactive` | `voltage`
            For example: `lighting_0_apparent`

        appliances : dict of DataFrames, optinal
            Each key is an appliance name string in the form `<appliance>_<N>`:
            * `appliance` is a standard appliance name.  For a list of valid 
               names, see nilmtk/docs/appliance_names.txt
            * `N` is the index for that appliance within this building. 
               Indexed from 0.
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
            e.g. {`tv_0`    : `livingroom_0`, 
                  `fridge_0`: `kitchen_0`}

        map_appliance_to_upstream : dict
            Keys are appliance names of the form `<appliance>_<N>`
            Values are *either* the circuit *or* the mains to which this
            appliance is connected.

        map_circuit_to_mains : dict
            Each key is a circuit name of the form `<circuit>_<N>`
            Each value is a mains name.


    """

    def __init__(self):
        geographic_coordinates = None
        n_occupants = None
        rooms = None
        electric = namedtuple("electric", ["mains", "circuits", "appliances",
                                           "appliance_estimates", 
                                           "nominal_mains_voltage",
                                           "map_appliance_to_room",
                                           "map_appliance_to_upstream",
                                           "map_circuit_to_mains"])

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

    def crop(self, start, end): 
        """Reduce all timeseries to just these dates"""
        raise NotImplementedError

    def plot_appliance_activity(self, source):
        """Plot a compact representation of all appliance activity."""
        raise NotImplementedError
