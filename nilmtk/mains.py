class Mains(object):
    """
    Attributes
    ----------
    meters : list of nilmtk.EMeter objects
    """
    def __init__(self, meters):
        self.meters = meters
        
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
    
    def set_loader_attributes(self, **kwargs):
        # TODO: is this really the best way to do this?
        # what attributes do we want to set?  Just mask?  If so
        # then maybe we should has a set_mask() method?

        # TODO: I guess each meter's loader needs to be setup so
        # we get chunks spanning the same time windows so, for example,
        # we can still calculate energy per time period?
        for meter in self.meters:
            meter.set_loader_attributes(**kwargs)
            
