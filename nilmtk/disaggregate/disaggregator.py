from abc import ABCMeta, abstractmethod

class Disaggregator(object):
    """Abstract Base Class for all Disaggregators.  This class
    defines the common interface to all Disaggregators.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, mains=None, appliances=None):
        """Train the disaggregation algorithm.

        There are three training modes.  The mode will be selected based
        on which arguments are provided.  The modes are:

        * Supervised training on appliance data
        * Supervised training on mains data, using simultaneously recorded
          appliance data as the labels
        * Unsupervised training on mains data only

        The parameters are designed to accept the standard mains and 
        appliance data structure used in nilmtk's `Electricity` class.

        `train` can be called more than once during the lifetime of
        the object to train on new data.

        Some subclasses of Disaggregator will be trainable using more
        than training mode.  In this case, call `train` multiple times
        using different combinations of arguments.

        Parameters
        ----------
        mains : pandas.DataFrame or pandas.Series, optional
            Whole-house, aggregate power data in Watts.
            index is a DataTimeIndex
            column names use the nilmtk standard for mains data

        appliances : dict of list of DataFrames, optional
            Keys are appliance names, using nilmtk standards
            Values are lists of DataFrames, one DataFrame per appliance, using
            standard nilmtk names in columns for recorded parameters.

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
            some combination of:
            * estimated power
            * estimated state
            * confidence [0,1]
        """
        return
