from nilmtk import *

ds = DataSet("/Users/nipunbatra/Downloads/nilm_gjw_data.hdf5")

elec = ds.buildings[1].elec

elec.plot()
