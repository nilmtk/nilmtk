from electricity import Electricity
from water import Water
from gas import Gas


class Utility(object):
    """Store and process utility information for a building.

    Attributes
    ----------

    electric
    gas
    water
    """

    def __init__(self):
        self.electric = Electricity()
        self.water = Water()
        self.gas = Gas()
