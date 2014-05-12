from .hashable import Hashable
from collections import namedtuple

MainsID = namedtuple('MainsID', ['building', 'dataset'])

class Mains(Hashable):
    """
    Attributes
    ----------
    meters : list of nilmtk.ElectricityMeter objects

    identifier : MainsID namedtuple with fields:
        building : int
        dataset : str

    metadata : dict
        nominal_voltage : float
    """

    # TODO: Perhaps mains should be a subclass of Meter?  Like
    # DualSupplyMeter??

    def __init__(self, building, dataset, meters=None):
        assert isinstance(building, int)
        assert isinstance(dataset, str)
        self.identifier = MainsID(building, dataset)
        self.meters = [] if meters is None else meters
        assert isinstance(self.meters, list)
        for meter in self.meters:
            assert meter.identifier.dataset == self.identifier.dataset
            assert meter.identifier.building == self.identifier.building
        
    def available_measurements(self):
        measurements = set(self.meters[0].metadata['device']['measurements'])
        for meter in self.meters[1:]:
            meter_measurements = set(meter.metadata['device']['measurements'])
            measurements = measurements.intersection(meter_measurements)
        return measurements

    def power_series(self, **kwargs):
        """Power series.  Sums together three phases / dual split power.
        
        Returns
        -------
        generator of pd.Series of power measurements.
        """
        # TODO: warn if any meter also measures an appliance
        # which isn't this appliance / mains.  User proper
        # Python warning: http://docs.python.org/2/library/warnings.html
        
        if len(self.meters) == 1:
            return self.meters[0].power_series(**kwargs)
        else:
            raise NotImplementedError
        
        # TODO: really not confident the code below is correct!!
        # power_generators = []
        # for meter in self.meters:
        #     power_generators.append(meter.power_series(**kwargs))
        # for generators in zip(power_generators):
        #     power_for_chunk = generators[0]
        #     for generator in generators[1:]:
        #         power_for_chunk += generator
        #     yield power_for_chunk
        
    def total_energy(self):
        """Returns EnergyResults object, as if it were a single meter"""
        if len(self.meters) == 1:
            return self.meters[0].total_energy()
        else:
            raise NotImplementedError

    def good_sections(self):
        """Returns good sections where all meters in self.meters are good."""
        if len(self.meters) == 1:
            return self.meters[0].good_sections()
        else:
            raise NotImplementedError

    def export(self, destination, key):
        # Exports metadata and 
        # list of meters as a list of meter ID ints
        raise NotImplementedError

    def load(self, store, key, meters_dict):
        # Import metadata
        # self.meters = list of relevant meter
        # objects taken from meters_dict
        raise NotImplementedError
    
    def set_mask(self, mask):
        for meter in self.meters:
            meter.mask = mask
            
