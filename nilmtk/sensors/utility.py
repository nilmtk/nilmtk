from nilmtk.sensors.electricity import Electricity
from nilmtk.sensors.gas import Gas
from nilmtk.sensors.water import Water
import json


class Utility(object):

    """Store and process utility information for a building.

    Attributes
    ----------
    electric : nilmtk Electricity object

    gas : Series, optional
        gas consumption in kWh

    water : Series, optional
        water consumption in cubic meters
    """

    def __init__(self):
        self.electric = Electricity()
        self.water = Water()
        self.gas = Gas()

    def to_json(self):
        representation = {}
        representation["electric"] = self.electric.to_json()
        representation["water"] = self.water.to_json()
        representation["gas"] = self.gas.to_json()
        return json.dumps(representation)
