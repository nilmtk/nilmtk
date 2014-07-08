from __future__ import print_function, division
from nilmtk import HDFDataStore, DataSet
from nilmtk.disaggregate import CombinatorialOptimisation

datastore = HDFDataStore('/home/jack/workspace/python/nilmtk/notebooks/redd.h5')
dataset = DataSet()
dataset.load(datastore)
elec = dataset.buildings[1].elec
co = CombinatorialOptimisation()
co.train(elec)
print(co.model)
mains = elec.mains()

output = HDFDataStore('output.h5', 'w')
predictions = co.disaggregate(mains, output)
