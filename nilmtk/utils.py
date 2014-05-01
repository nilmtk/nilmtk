from __future__ import print_function, division
import numpy as np

def timedelta64_to_secs(timedelta):
    """Convert `timedelta` to seconds.

    Parameters
    ----------
    timedelta : np.timedelta64

    Returns
    -------
    float : seconds
    """
    return timedelta / np.timedelta64(1, 's')
