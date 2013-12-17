from nilmtk.sensors.utility import Utility
from nilmtk.sensors.ambient import Ambient
import json, copy


class Building(object):

    """Represent a physical building (e.g. a domestic house).

    Attributes
    ----------

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

    metadata : dict
        geographical_coordinates : pair of floats, optional
            (latitude, longitude)
            Only specify this if the geo location of the building is known.
            Otherwise leave this blank and users should fall back to using
            the geo location specified for the dataset as a whole.

        n_occupants : int, optional
             Max number of occupants.

        rooms : dict of strings, optional
            Keys are room names. Use standard names for each room from
            docs/standard_names/rooms.txt.
            Values are the number of each type of room in this building.
            For example:
            room = {'kitchen': 1, 'bedroom': 3, ...}
    """

    def __init__(self):
        self.metadata = {}
        self.utility = Utility()
        self.ambient = Ambient()

    def crop(self, start, end):
        """Reduce all timeseries to just these dates"""
        raise NotImplementedError

    def to_json(self):
        representation = copy.copy(self.metadata)
        representation["utility"] = self.utility.to_json()
        representation["ambient"] = self.ambient.to_json()
        return json.dumps(representation)
