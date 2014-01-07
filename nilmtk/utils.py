import os
import numpy as np
import pandas as pd

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


def find_nearest_non_vectorized(known_array, test_array):
    indices = np.zeros(len(test_array))
    residual = np.zeros(len(test_array))
    for i in xrange(len(test_array)):
        [indices[i], residual[i]] = find_nearest(known_array, test_array[i])
    return [indices, residual]


def find_nearest_vectorized(known_array, test_array):

    # Recipe borrowed from
    # http://stackoverflow.com/questions/20780017/numpy-vectorize-finding-closest-value-in-an-array-for-each-element-in-another-a

    differences = (test_array.reshape(1, -1) - known_array.reshape(-1, 1))
    indices = np.abs(differences).argmin(axis=0)
    residual = np.diagonal(differences[indices, ])
    return [indices, residual]


def find_nearest_searchsorted(known_array, test_array):
    index_sorted = np.argsort(known_array)
    known_array_sorted = known_array[index_sorted]

    idx1 = np.searchsorted(known_array_sorted, test_array)
    idx2 = np.clip(idx1 - 1, 0, len(known_array_sorted) - 1)

    diff1 = known_array_sorted[idx1] - test_array
    diff2 = test_array - known_array_sorted[idx2]

    indices = index_sorted[np.where(diff1 <= diff2, idx1, idx2)]
    residual = test_array - known_array[indices]

    return [indices, residual]


def secs_per_period_alias(alias):
    """Seconds for each Pandas period alias."""
    dr = pd.date_range('00:00', periods=2, freq=alias)
    return (dr[-1] - dr[0]).total_seconds()

