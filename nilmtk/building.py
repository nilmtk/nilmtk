class Building(object):
    """Represent a physical building (e.g. a domestic house).

    Attributes
    ----------
    aggregate : DataFrame, shape (n_samples, n_features), optional
        Using standard column names of the form `mains_<N>_<measurement>` where:
        * `N` is the phase or split.  Indexed from 0.
        * `measurement` is one of `apparent` | `active` | `reactive` | `voltage`
        For example: `mains_0_apparent`

    appliances : DataFrame, shape (n_samples, n_features), optional
        Using standard column names of the form `<appliance>_<N>_<measurement>`:
        * `appliance` is a standard appliance name.  For a list of valid names,
           see nilmtk/docs/appliance_names.txt
        * `N` is the index for that appliance within this building. 
           Indexed from 0.
        * `measurement` is one of `apparent` | `active` | `reactive` | `voltage`
        For example: `tv_0_apparent` or `tv_2_apparent`

    appliance_estimates : Panel (3D matrix), optional
        Output from the NILM algorithm.
        The first two dimensions are the same as for the appliance DataFrame
        The third dimension describes, for each appliance and for each time:
        * `power` : float. Estimated power consumption in Watts.
        * `state` : int, optional.
        * `confidence` : float [0,1], optional.
        * `power_prob_dist` : object describing the probability dist, optional
        * `state_prob_dist` : object describing the probability dist, optional

    geographic_coordinates : pair of floats, optional
        (latitude, longitude)

    n_occupants : int, or pair of ints, optional
        Either the exact number of occupants (a single int) 
        or a range (pair of ints).

    nominal_mains_voltage : float, optional

    rooms : list of strings, optional
        A list of room names. Use standard names for each room

    map_appliance_to_room : dict
        e.g. {`tv_0`    : `livingroom_0`, 
              `fridge_0`: `kitchen_0`}

    """

    def get_appliance(self, appliance_name, measurement=None):
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
