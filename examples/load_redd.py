

from __future__ import print_function
from nilmtk.dataset import REDD

redd = REDD()
print (redd.load_building('/data/REDD/low_freq/', 'house_1'))

print('URL =', redd.urls[0])
print()
print('Citation =', redd.citations[0])
print()
print('Mains data:')

building = redd.buildings['house_1']
print(building.electric.mains.describe())
