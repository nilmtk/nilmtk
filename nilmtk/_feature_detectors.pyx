"""
   Copyright 2013 Jack Kelly (aka Daniel)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

"""
RESOURCES:
http://wesmckinney.com/blog/?p=278
http://docs.cython.org/src/userguide/numpy_tutorial.html
"""

from __future__ import print_function, division
import numpy as np
cimport numpy as np
import pandas as pd

# Data types for timestamps (TS = TimeStamp)
TS_DTYPE = np.uint64
ctypedef np.uint64_t TS_DTYPE_t

# Data types for power data (PW = PoWer)
PW_DTYPE = np.float32
ctypedef np.float32_t PW_DTYPE_t

def _sanity_check_input_to_steady_state_detectors(
                  np.ndarray[PW_DTYPE_t, ndim=1] watts,
                  Py_ssize_t min_n_samples, 
                  PW_DTYPE_t max_range):
    if watts is None or min_n_samples is None or max_range is None:
        raise ValueError('Do not use None for any arguments.')
    if watts.size < min_n_samples:
        raise ValueError('watts array must have more than '
                         'min_n_samples={} elements!'.format(min_n_samples))
    if np.any(np.isnan(watts)):
        raise ValueError('Please remove all NaNs!')


def steady_states(series,
                  Py_ssize_t min_n_samples=3, 
                  PW_DTYPE_t max_range=15):
    """Steady_state detector based on the definition of steady states given
    in Hart 1992 [1]_

    Parameters
    ----------
    series : pd.Series
        Watts. Must be np.float_32

    min_n_samples : int, optional
        Defaults to 3. Minimum number of consecutive samples per steady state.
        Hart used 3.

    max_range : float, optional
        Defaults to 15 Watts. Maximum permissible range between the lowest
        and highest value per steady state. Hart used 15.
    

    Returns
    -------
    steady_states : pd.DataFrame
        Each row is a steady state.  Columns:
           * index: datetime of start of steady state
           * end: datetime of end of steady state
           * power: mean power for steady state

    References
    ----------
    .. [1] Hart, G. W. (1992). "Nonintrusive appliance load monitoring."
       Proceedings of the IEEE, 80(12), 1870–1891. doi:10.1109/5.192069
       page 1882, under the heading 'C. Edge Detection'

    """

    cdef:
        np.ndarray[PW_DTYPE_t, ndim=1] watts
        Py_ssize_t i, n, ss_start_i # steady state start index
        PW_DTYPE_t p, ss_max, ss_min # steady state max and mins

    watts = series.values
    _sanity_check_input_to_steady_state_detectors(watts, min_n_samples, max_range)

    n = len(watts) # number of samples
    idx = [] # list of timestamps; this will form the index for returned DataFrame
    ss = [] # steady states. What we return
    ss_start_i = 0 # steady state start index
    ss_min = ss_max = watts[ss_start_i] # min and max power values in the SS

    # Loop through all the power data, identifying steady states
    for i from 1 <= i < n-1:
        p = watts[i]

        if p > ss_max:
            ss_max = p
        if p < ss_min:
            ss_min = p

        if (ss_max - ss_min) > max_range: # finished a candidate steady state.
            if (i - ss_start_i) >= min_n_samples:
                idx.append(series.index[ss_start_i])
                ss.append({'end': series.index[i-1],
                           'power': watts[ss_start_i:i].mean()})

            ss_start_i = i
            ss_min = ss_max = watts[ss_start_i]

    # Were we tracking a steady state when we ran out of data?
    # If so, store that steady state at the end of the data.
    if (i - ss_start_i) >= min_n_samples:
        idx.append(series.index[ss_start_i])
        ss.append({'end': series.index[i-1],
                   'power': watts[ss_start_i:i].mean()})
            
    return pd.DataFrame(ss, index=idx)
