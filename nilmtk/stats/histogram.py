import numpy as np
from warnings import warn


def histogram_from_generator(generator, bins=None, range=None, **kwargs):
    """Apart from 'generator', takes the same key word arguments 
    as numpy.histogram. And returns the same objects as np.histogram.
    
    Parameters
    ----------
    range : None or (min, max)
        range differs from np.histogram's interpretation of 'range' in 
        that either element can be None, in which case the min or max
        of the first chunk is used.
    bins : None or int
        if None then uses int(range[1]-range[0])
    """

    if 'density' in kwargs or 'normed' in kwargs:
        warn("This function is not designed to output densities.")

    histogram_cumulator = None
    for chunk in generator:
        if range is None:
            range = (chunk.min(), chunk.max())
        else:
            if range[0] is None:
                range = (chunk.min(), range[1])
            if range[1] is None:
                range = (range[0], chunk.max())
        if bins is None:
            bins = int(range[1] - range[0])
        hist, bins = np.histogram(chunk, bins=bins, range=range, **kwargs)
        if histogram_cumulator is None:
            histogram_cumulator = hist
        else:
            histogram_cumulator += hist

    return histogram_cumulator, bins
