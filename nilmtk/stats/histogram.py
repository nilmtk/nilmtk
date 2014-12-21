from __future__ import print_function, division
import numpy as np
from warnings import warn


def histogram_from_generator(generator, bins=10, range=None, **kwargs):
    """Apart from 'generator', takes the same key word arguments 
    as numpy.histogram. And returns the same objects as np.histogram.
    
    Parameters
    ----------
    range : None or (min, max)
        range differs from np.histogram's interpretation of 'range' in 
        that either element can be None, in which case the min or max
        of the first chunk is used.
    """

    if 'density' in kwargs or 'normed' in kwargs:
        warn("This function is not designed to output densities.")

    histogram_cumulator = np.zeros(bins)
    for chunk in generator:
        if range is not None:
            if range[0] is None:
                range = (chunk.min(), range[1])
            if range[1] is None:
                range = (range[0], chunk.max())
        hist, bins = np.histogram(chunk, bins=bins, range=range, **kwargs)
        histogram_cumulator += hist

    return histogram_cumulator, bins
