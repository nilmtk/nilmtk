from __future__ import print_function, division
from .appliancegroup import ApplianceGroup

class Electricity(ApplianceGroup):
    """Represents mains circuit in a single building.
    
    Attributes
    ----------
    mains : Mains object (which has references to relevant meters)
        
    Assumptions
    -----------
    * exactly one Mains object per building
    """
    
    def meters_directly_downstream_of_mains(self):
        """
        Returns
        -------
        list of Meter objects which are directly downstream of mains; i.e. one hop from mains
        """
        raise NotImplementedError
        wiring_graph = self.wiring_graph()
        # return meters (not appliances) one hop from root (mains)

    def proportion_energy_submetered(self):
        good_mains_timeframes = self.mains.good_timeframes()
        submetered_energy = 0.0
        for meter in self.meters_directly_downstream_of_mains():
            energy = meter.total_energy(timeframes=good_mains_timeframes)
            submetered_energy += select_best_ac_type(energy.combined, mains)
        mains_energy = select_best_ac_type(self.mains.total_energy().combined)
        return submetered_energy / mains_energy
