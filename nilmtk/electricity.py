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

    def proportion_of_energy_submetered(self):
        """
        Returns
        -------
        float [0,1]
        """
        mains = self.mains()
        good_mains_sections = mains.good_sections().combined
        submetered_energy = 0.0
        common_ac_types = None
        for meter in self.meters_directly_downstream_of_mains():
            energy = meter.total_energy(periods=good_mains_sections).combined
            ac_types = set(energy.keys())
            ac_type = select_best_ac_type(ac_types, 
                                          mains.available_ac_types())
            submetered_energy += energy[ac_type]
            if common_ac_types is None:
                common_ac_types = ac_types
            else:
                common_ac_types = common_ac_types.intersection(ac_types)
        mains_energy = mains.total_energy().combined
        ac_type = select_best_ac_type(mains_energy.keys(), common_ac_types)
        return submetered_energy / mains_energy[ac_type]
