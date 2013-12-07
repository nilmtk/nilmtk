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

    electric : nilmtk Electricity object

    gas : Series, optional
        gas consumption in kWh

    water : Series, optional
        water consumption in cubic meters

    weather : DataFrame, optional
        index is a timezone-aware DateTimeIndex
        columns are names like `temperature` (degrees C) etc

    """

    def __init__(self):
        self.geographic_coordinates = None
        self.n_occupants = None
        self.rooms = []
        self.utility = Utility()
        self.ambient = Ambient()
        self.weather = None

    def crop(self, start, end):
        """Reduce all timeseries to just these dates"""
        raise NotImplementedError

