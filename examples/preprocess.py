import nilmtk.preprocessing.electricity.building as prepb
import nilmtk.preprocessing.electricity.single as prep
from nilmtk.stats.electricity.building import plot_missing_samples_using_bitmap
from nilmtk.sensors.electricity import Measurement

from nilmtk.dataset import DataSet
dataset = DataSet()
#dataset.load_hdf5("/home/nipun/Dropbox/nilmtk_datasets/redd/low_freq")
dataset.load_hdf5("/home/nipun/Dropbox/nilmtk_datasets/iawe")
# assumes that `dataset` is already loaded
building = dataset.buildings[1]

# 1. sum together split mains and DualSupply appliances
building.utility.electric = building.utility.electric.sum_split_supplies()

# optional. remove samples where voltage outside range
# Fixing implausible voltage values
building = prepb.filter_out_implausible_values(
    building, Measurement('voltage', ''), 160, 260)

# Filtering out data. Note that this will remove motor as it does not have
# any data in this period
building = prepb.filter_datetime(
    building, '7-13-2013', '8-4-2013')

# 2. downsample mains, circuits and appliances
building = prepb.downsample(building, rule='1T')

# 3. Fill large gaps in appliances with zeros and forward-fill small gaps
building = prepb.fill_appliance_gaps(building)

building = prepb.prepend_append_zeros(building, '7-13-2013', '8-4-2013', '1T', 'Asia/Kolkata')

print "*" * 80
print "Before"
print "*" * 80
for appliance, appliance_df in building.utility.electric.appliances.iteritems():
    print appliance, appliance_df.index
print building.utility.electric.mains.values()[0].index


# 4. Intersection of mains and appliance datetime indicies
building = prepb.make_common_index(building)
print "*" * 80
print "After"
print "*" * 80
for appliance, appliance_df in building.utility.electric.appliances.iteritems():
    print appliance, appliance_df.index
print building.utility.electric.mains.values()[0].index
# 5. Contiguous blocks of datetimes
# could use nilmtk.stats.electricity.single.periods_with_sufficient_samples ?

# plot_missing_samples_using_bitmap(building.utility.electric)
