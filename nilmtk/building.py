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

    rooms : dict of strings, optional
        Keys are room names. Use standard names for each room from
        docs/standard_names/rooms.txt.
        Values are the number of each type of room in this building.
        For example:
        room = {'kitchen': 1, 'bedroom': 3, ...}

    utility :  nilmtk Utility object

    ambient : dict of nilmtk Ambient objects
        Keys are pairs of the form
        (<room name>, <number of that room type (indexed from 0>)
        e.g. ('kitchen', 1) or ('bedroom', 3)
        If there are sensors which cover the whole house (or if
        a dataset gives a single set of ambient measurements for a single
        house without describing which room those measurements are taken)
        then use the key 'building'.
        Values are nilmtk Ambient objects.

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
        self.ambient = {}
        self.external = None

    def crop(self, start, end):
        """Reduce all timeseries to just these dates"""
        raise NotImplementedError

