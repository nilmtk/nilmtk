from os.path import join

import matplotlib.pyplot as plt

from nilmtk import DataSet, HDFDataStore
from nilmtk.legacy.disaggregate import fhmm_exact
from nilmtk.metrics import f1_score

"""
This file replicates issue #376 (which should now be fixed)
https://github.com/nilmtk/nilmtk/issues/376
"""

data_dir = "/data"
building_number = 3
disag_filename = join(data_dir, "disag-fhmm" + str(building_number) + ".h5")

data = DataSet(join(data_dir, "redd.h5"))
print("Loading building " + str(building_number))
elec = data.buildings[building_number].elec

top_train_elec = elec.submeters().select_top_k(k=5)
fhmm = fhmm_exact.FHMM()
fhmm.train(top_train_elec)

output = HDFDataStore(disag_filename, "w")
fhmm.disaggregate(elec.mains(), output)
output.close()

### f1score fhmm
disag = DataSet(disag_filename)
disag_elec = disag.buildings[building_number].elec

f1 = f1_score(disag_elec, elec)
f1.index = disag_elec.get_labels(f1.index)
f1.plot(kind="barh")
plt.ylabel("appliance")
plt.xlabel("f-score")
plt.title("FHMM")
plt.tight_layout()
plt.savefig(join(data_dir, "f1-fhmm" + str(building_number) + ".png"))
disag.store.close()
####
print("Finishing building " + str(building_number))
