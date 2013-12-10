from nilmtk.sensors.utility import Utility
from nilmtk.sensors.ambient import Ambient


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

    utility :  nilmtk Utility object

    ambient : nilmtk Ambient object
        Stores weather etc.

    external : DataFrame with measurements for external temperature,
        sunshine, rain fall, wind etc.  (This attribute is called `external`
        so that we can capture not just weather but any other external
        measurements that we might be interested in.)

    """

    def __init__(self):
        self.geographic_coordinates = None
        self.n_occupants = None
        self.rooms = []
        self.utility = Utility()
        self.ambient = Ambient()
        self.external = None

    def crop(self, start, end):
        """Reduce all timeseries to just these dates"""
        raise NotImplementedError

