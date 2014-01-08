import nilmtk.preprocessing.electricity.building as prepb
import nilmtk.preprocessing.electricity.single as prep
from nilmtk.stats.electricity.building import plot_missing_samples_using_bitmap

# assumes that `dataset` is already loaded
building = dataset.buildings[1]

# 1. sum together split mains and DualSupply appliances
building.utility.electric = building.utility.electric.sum_split_supplies()

# 2. downsample mains, circuits and appliances
building = prepb.downsample(building, rule='1T')

# 3. Fill large gaps in appliances with zeros and forward-fill small gaps
building = prepb.fill_appliance_gaps(building)

# 4. Intersection of mains and appliance datetime indicies

# 5. Contiguous blocks of datetimes
# could use nilmtk.stats.electricity.single.periods_with_sufficient_samples ?

plot_missing_samples_using_bitmap(building.utility.electric)
