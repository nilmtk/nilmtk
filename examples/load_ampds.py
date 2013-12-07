from __future__ import print_function
from nilmtk.dataset import ampds

#dataset = ampds.AMPDS()

PATH = '/home/nipun/Desktop/AMPds/'
'''
# Loading data for Home 01
ampds.load_electricity(PATH)
ampds.load_water(PATH)
ampds.load_gas(PATH)
'''

# Load everything
dataset = ampds.AMPDS()
dataset.load(PATH)


