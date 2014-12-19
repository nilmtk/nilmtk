from __future__ import print_function, division
import numpy as np
from warnings import warn


def histogram_from_generator(generator, bins=10, **kwargs):
    """Apart from 'generator', takes the same key word arguments 
    as numpy.histogram. And returns the same objects as np.histogram."""

    if 'density' in kwargs or 'normed' in kwargs:
        warn("This function is not designed to output densities.")

    histogram_cumulator = np.zeros(bins)
    for chunk in generator:
        hist, bins = np.histogram(chunk, bins=bins, **kwargs)
        histogram_cumulator += hist

    return histogram_cumulator, bins
