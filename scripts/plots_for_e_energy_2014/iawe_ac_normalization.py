"""Indian AC"""

DATA_PATH = "/home/nipun/Dropbox/nilmtk_datasets/iawe/"
FIG_SAVE_PATH = "/home/nipun/Desktop/abc.pdf"
from nilmtk.dataset import DataSet
import nilmtk.preprocessing.electricity.building as prepb
import nilmtk.preprocessing.electricity.single as prep
from nilmtk.sensors.electricity import Measurement
import matplotlib.pyplot as plt

ds = DataSet()
ds.load_hdf5(DATA_PATH)

# First building
building = ds.buildings[1]

# 1. sum together split mains and DualSupply appliances
building.utility.electric = building.utility.electric.sum_split_supplies()

# optional. (required for iAWE) remove samples where voltage outside range
# Fixing implausible voltage values
building = prepb.filter_out_implausible_values(
    building, Measurement('voltage', ''), 160, 260)

# optional. (required for iAWE) Note that this will remove motor as it does not have
# any data in this period
building = prepb.filter_datetime(
    building, '7-13-2013', '8-4-2013')

ac = building.utility.electric.appliances[('air conditioner', 1)]
ac_power = ac[('power', 'active')]
voltage = ac[('voltage', '')]

normalized_power = prep.normalise_power(ac_power, voltage, 230)

fig, axes = plt.subplots(ncols=2)
axes[0].hist(ac_power, bins=100)
axes[1].hist(normalized_power, bins=100)
axes[0].set_title("Before normlization")
axes[1].set_title("After normlization")

fig.tight_layout()
fig.savefig(FIG_SAVE_PATH)
