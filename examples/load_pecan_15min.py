from __future__ import print_function
from nilmtk.dataset import pecan

pecan_15min = pecan.Pecan_15min()
print ("re")
pecan_15min.load_building_names('/home/nipun/Desktop/PECAN/')
pecan_15min.load_building('/home/nipun/Desktop/PECAN/', 'Home 01')
building = pecan_15min.buildings['Home_01']

'''
print('URL =', redd.urls[0])
print()
print('Citation =', redd.citations[0])
print()
print('Mains data:')
'''


#building = pecan_15min.buildings['house_1']
#print(building.electric.mains.describe())
