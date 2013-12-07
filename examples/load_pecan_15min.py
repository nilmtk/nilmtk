from __future__ import print_function
from nilmtk.dataset import pecan

pecan_15min = pecan.Pecan_15min()

# Name of buildings
pecan_15min.load_building_names('/home/nipun/Desktop/PECAN/')

# Loading data for Home 01
pecan_15min.load_building('/home/nipun/Desktop/PECAN/', 'Home 01')

# Accessing Home 01 Building
building = pecan_15min.buildings['Home_01']

# Summary of Mains
print("Mains summary")
print(building.electric.mains.describe())

# Summary of appliances
print("Appliances summary")
for appliance in building.electric.appliances:
    print(building.electric.appliances[appliance].describe())

# Loading all buildings data
pecan_15min_complete = pecan.Pecan_15min()
pecan_15min_complete.load('/home/nipun/Desktop/PECAN/')

'''
print('URL =', redd.urls[0])
print()
print('Citation =', redd.citations[0])
print()
print('Mains data:')
'''


#building = pecan_15min.buildings['house_1']
#print(building.electric.mains.describe())
