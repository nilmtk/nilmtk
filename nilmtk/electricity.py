from __future__ import print_function, division
from .appliancegroup import MeterGroup

class Electricity(MeterGroup):
    """Represents mains circuit in a single building.
            
    Assumptions
    -----------
    * exactly one Mains object per building
    """    

    def proportion_energy_submetered(self):
        good_mains_sections = self.mains.good_sections()
        submetered_energy = 0.0
        for meter in self.meters_directly_downstream_of_mains():
            energy = meter.total_energy(timeframes=good_mains_sections)
            submetered_energy += select_best_ac_type(energy.combined, mains)
        mains_energy = select_best_ac_type(self.mains.total_energy().combined)
        return submetered_energy / mains_energy
