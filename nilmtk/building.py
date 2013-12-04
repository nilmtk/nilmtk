class Building(object):
    """Represent a physical building (e.g. a domestic house).

    Attributes
    ----------
    aggregate : DataFrame

    appliances : DataFrame

    appliance_estimates : DataFrame

    geo_location

    n_occupants

    nominal_mains_voltage

    appliances_in_room

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
