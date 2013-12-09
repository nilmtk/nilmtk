import os
import numpy as np


def get_immediate_subdirectories(dir):
    # From Richie Hindle's StackOverflow answer:
    # http://stackoverflow.com/a/800201/732596
    if dir:
        subdirs = [name for name in os.listdir(dir)
                   if os.path.isdir(os.path.join(dir, name))]
    else:
        subdirs = []
    return subdirs


def find_nearest_index(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    diff = array[idx] - value
    return [idx, -diff]