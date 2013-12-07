from .pir import PIR
from .door import DoorStatus
from .temperature import Temperature


class Ambient(object):
    """Store and process utility information for a building.

    Attributes
    ----------

    pir
    door
    temperature
    """

    def __init__(self):
        self.pir = PIR()
        self.door = DoorStatus()
        self.temperature = Temperature()
