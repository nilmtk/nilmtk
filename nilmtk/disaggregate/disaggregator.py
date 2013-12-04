from abc import ABCMeta, abstractmethod

class Disaggregator(object):
    """Abstract Base Class for all Disaggregators.  This class
    defines the common interface to all Disaggregators.
    """

    __metaclass__ = ABCMeta

    #------------- TRAINING ------------

    @abstractmethod
    def train_on_appliances(self, appliances):
        """Train on appliance data.

        Parameters
        ----------
        appliances : dict
            Each key of the dict is an appliance name, 
            each value of the dict is a list of filenames.

            For example:
            {'fridge': 'fridge1.csv', 'fridge2.csv',
             'kettle': 'kettle1.csv', 'kettle2.csv'}
        """
        return

    @abstractmethod
    def train_supervised_on_aggregate(self, aggregate, appliances):
        """Train on aggregate data, using simultaneously measured appliance
        data to label features in the aggregate data.

        Parameters
        ----------
        aggregate : pandas.DataFrame
            The column names must conform to the nilmtk standard
            for aggregate data.

        appliances : pandas.DataFrame
            The column names must conform to the nilmtk standard
            for appliance data.

        """
        return

    @abstractmethod
    def train_unsupervised_on_aggregate(self, aggregate):
        """Unsupervised training on aggregate data,
        without any labels.

        Parameters
        ----------
        aggregate : pandas.DataFrame
            The column names must conform to the nilmtk standard
            for aggregate data.
        """
        return

    @abstractmethod
    def disaggregate(self, building):
        """Runs non-intrusive load monitoring on the aggregate data from 
        the building.

        We pass in an entire Building object because some Disaggregators
        can take advantage of building metadata such a geo location to 
        improve the estimates.

        Parameters
        ----------
        building : Building

        Returns
        -------
        appliance_estimates: Panel, shape (n_samples, n_appliances, [1,3])
            Returns a 3D matrix (Panel). The third dimension represents
            estimated power, estimated state, confidence [0,1]
        """
        return
