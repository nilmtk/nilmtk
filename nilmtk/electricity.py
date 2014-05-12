from __future__ import print_function, division
from .metergroup import MeterGroup
from .measurement import select_best_ac_type
from .utils import tree_root, nodes_adjacent_to_root

class Electricity(MeterGroup):
    """Represents mains circuit in a single building.
            
    Assumptions
    -----------
    * exactly one Mains object per building
    """    

    def mains(self):
        return tree_root(self.wiring_graph())

    def meters_directly_downstream_of_mains(self):
        return nodes_adjacent_to_root(self.wiring_graph())

    def proportion_energy_submetered(self):
        mains = self.mains()
        good_mains_sections = mains.good_sections()
        submetered_energy = 0.0
        for meter in self.meters_directly_downstream_of_mains():
            energy = meter.total_energy(periods=good_mains_sections)
            submetered_energy += select_best_ac_type(energy.combined, mains)
        mains_energy = select_best_ac_type(self.mains.total_energy().combined)
        return submetered_energy / mains_energy
