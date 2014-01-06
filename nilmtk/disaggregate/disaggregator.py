from abc import ABCMeta, abstractmethod
from nilmtk.sensors.electricity import Measurement


class Disaggregator(object):

    """Abstract Base Class for all Disaggregators.  This class
    defines the common interface to all Disaggregators.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, building, aggregate='mains', submetered='appliances',
              disagg_features=[Measurement('power', 'active')],
              environmental=None):
        """Train the disaggregation algorithm.

        There are three training modes:

        * Supervised training on appliance data
        * Supervised training on mains data, using simultaneously recorded
          appliance data as the labels
        * Unsupervised training on mains data only

        Some subclasses of Disaggregator will be trainable using more
        than one training mode.  In this case, `train` will attempt to use
        all training modes available, in order to produce the best models.

        `train` can be called more than once during the lifetime of
        the disaggregator object to train on new data.

        Parameters
        ----------
        building : a nilmtk Building object

        aggregate : 'mains' | 'circuits', optional

        submetered : 'appliances' | 'circuits', optional

        disagg_features : list of Measurements, optional
            default=[Measurement('power', 'active')]

        environmental :
            External sensor data, e.g. weather data from the local
            metoffice weather station.
        """
        return

    @abstractmethod
    def disaggregate(
        self, building, disagg_features=[Measurement('power', 'active')],
            environmental=None):
        """Runs non-intrusive load monitoring on the aggregate data from
        the building.

        We pass in an entire Building object because some Disaggregators
        can take advantage of building metadata such a geo location to
        improve the estimates.

        Parameters
        ----------
        building : Building

        disagg_features : list of Measurements

        Returns
        -------
        appliance_estimates: DataFrame or Panel, shape (n_samples, n_appliances, [1,3])
            Returns a 3D matrix (Panel). The third dimension represents
            some combination of:
            * estimated power
            * estimated state
            * confidence [0,1]

        `disaggregate` also sets the `appliance_estimates` attribute of
        `building.utility.electric`.
        """
        return
