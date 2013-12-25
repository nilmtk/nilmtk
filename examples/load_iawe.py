from __future__ import print_function
from nilmtk.dataset import iawe

#dataset = ampds.AMPDS()

PATH = '/home/nipun/Desktop/AMPds/'
'''
# Loading data for Home 01
ampds.load_electricity(PATH)
ampds.load_water(PATH)
ampds.load_gas(PATH)
'''

# Load everything
dataset = iawe.IAWE()
dataset.add_mains()
dataset.add_appliances()
