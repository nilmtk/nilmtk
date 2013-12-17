from __future__ import print_function
from nilmtk.dataset import REDD
import time

redd = REDD()
t1 = time.time()
print (redd.load_building('/data/REDD/low_freq/', 'house_1'))
t2 = time.time()
print("Runtime = {:.2f}".format(t2-t1))

print('URL =', redd.metadata['urls'][0])
print()
print('Citation =', redd.metadata['citations'][0])
print()
print('Mains data:')

building = redd.buildings['house_1']
for mainsname, df in building.utility.electric.mains.iteritems():
    print(mainsname)
    print(df.describe())
