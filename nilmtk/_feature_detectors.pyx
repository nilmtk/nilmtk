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
from slicedpy.normal import Normal

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
    in Hart 1992, page 1882, under the heading 'C. Edge Detection'.

    Args:
        series (pd.Series): Watts. np.float_32
        min_n_samples (int): Optional. Defaults to 3. Minimum number of 
            consecutive samples per steady state.  Hart used 3.
        max_range (float): Optional. Defaults to 15 Watts. Maximum 
            permissible range between the lowest and highest value per
            steady state. Hart used 15.
    
    Returns:
        pd.DataFrame.  Each row is a steady state.  Columns:
           * index: datetime of start of steady state
           * end: datetime of end of steady state
           * power (slicedpy.Normal): summary stats describing power
    """

    cdef:
        np.ndarray[PW_DTYPE_t, ndim=1] watts
        Py_ssize_t i, n, ss_start_i # steady_state_start_index
        PW_DTYPE_t p, ss_max, ss_min # steady state max and mins

    watts = series.values
    _sanity_check_input_to_steady_state_detectors(watts, min_n_samples, max_range)

    n = len(watts)
    idx = [] # index for dataframe
    ss = [] # steady states. What we return
    ss_start_i = 0
    ss_min = ss_max = watts[ss_start_i]

    for i from 1 <= i < n-1:
        p = watts[i]

        if p > ss_max:
            ss_max = p
        if p < ss_min:
            ss_min = p

        if (ss_max - ss_min) > max_range: # Just left a candidate steady state.
            if (i - ss_start_i) >= min_n_samples:
                idx.append(series.index[ss_start_i])
                ss.append({'end': series.index[i-1],
                           'power': Normal(watts[ss_start_i:i])})

            ss_start_i = i
            ss_min = ss_max = watts[ss_start_i]

    if (i - ss_start_i) >= min_n_samples:
        idx.append(series.index[ss_start_i])
        ss.append({'end': series.index[i-1],
                   'power': Normal(watts[ss_start_i:i])})
            
    return pd.DataFrame(ss, index=idx)


def mean_steady_states(series,
                       Py_ssize_t min_n_samples=3,
                       PW_DTYPE_t max_range=10):
    """Steady_state detector where we calculate the mean of each steady state;
    if the next sample is more than max_range away from the mean then this is
    the end of the steady state.

    Args:
        series (pd.Series): Watts. np.float_32
        min_n_samples (int): Optional. Defaults to 3. Minimum number of
            consecutive samples per steady state. Hart used 3.
        max_range (float): Optional. 

    Returns:
        pd.DataFrame.  Each row is a steady state.  Columns:
           * index: datetime of start of steady state
           * end: datetime of end of steady state
           * power (slicedpy.Normal): summary stats describing power
    """

    cdef:
        np.ndarray[PW_DTYPE_t, ndim=1] watts
        Py_ssize_t i, n, ss_start_i # steady_state_start_index
        PW_DTYPE_t p, ss_mean, accumulator # steady state max and mins

    watts = series.values
    _sanity_check_input_to_steady_state_detectors(watts, min_n_samples, max_range)

    n = len(watts)
    idx = [] # index for dataframe
    ss = [] # list of steady states. What we return
    ss_start_i = 0
    accumulator = ss_mean = watts[ss_start_i]

    for i from 1 <= i < n-1:
        p = watts[i]
        delta = np.fabs(ss_mean - p)
        ss_length = i - ss_start_i
        if delta > max_range: # Just left a candidate steady state.
            if ss_length >= min_n_samples:
                idx.append(series.index[ss_start_i])
                ss.append({'end': series.index[i-1],
                           'power': Normal(watts[ss_start_i:i])})

            ss_start_i = i
            accumulator = ss_mean = watts[ss_start_i]
        else:
            accumulator += p
            ss_mean = accumulator / (ss_length + 1)

    if (i - ss_start_i) >= min_n_samples:
        idx.append(series.index[ss_start_i])
        ss.append({'end': series.index[i-1],
                   'power': Normal(watts[ss_start_i:i])})
            
    return pd.DataFrame(ss, index=idx)


def sliding_mean_steady_states(series,
                               Py_ssize_t min_n_samples=3, 
                               PW_DTYPE_t max_range=10,
                               Py_ssize_t window_size=10):
    """Steady_state detector where we calculate the mean *at most* the last
    window_size samples of each steady state;  if the next sample is more than
    max_range away from the mean then this is the end of the steady state.

    Args:
        series (pd.Series): Watts. np.float_32
        min_n_samples (int): Optional. Defaults to 3. Minimum number of 
            consecutive samples per steady state.  Hart used 3.
        max_range (float): Optional.
        window_size (int): number of samples
    
    Returns:
        pd.DataFrame.  Each row is a steady state.  Columns:
           * index: datetime of start of steady state
           * end: datetime of end of steady state
           * power (slicedpy.Normal): summary stats describing power
    """

    cdef:
        np.ndarray[PW_DTYPE_t, ndim=1] watts
        Py_ssize_t i, n, ss_start_i, ss_length # steady_state_start_index
        PW_DTYPE_t p, ss_mean, accumulator, delta # steady state max and mins

    watts = series.values
    _sanity_check_input_to_steady_state_detectors(watts, min_n_samples, max_range)

    n = len(watts)
    idx = [] # index for dataframe
    ss = [] # list of steady states. What we return
    ss_start_i = 0
    accumulator = watts[ss_start_i]
    ss_mean = watts[ss_start_i]

    for i from 1 <= i < n-1:
        p = watts[i]
        delta = np.fabs(ss_mean - p)
        ss_length = i - ss_start_i
        if delta > max_range: # Just left a candidate steady state.
            if ss_length >= min_n_samples:
                idx.append(series.index[ss_start_i])
                ss.append({'end': series.index[i-1],
                           'power': Normal(watts[ss_start_i:i])})
            ss_start_i = i
            accumulator = ss_mean = watts[ss_start_i]
        else:
            accumulator += p
            if ss_length >= window_size:
                accumulator -= watts[i-window_size]
                ss_mean = accumulator / window_size
            else:
                ss_mean = accumulator / (ss_length + 1)

    if (i - ss_start_i) >= min_n_samples:
        idx.append(series.index[ss_start_i])
        ss.append({'end': series.index[i-1],
                   'power': Normal(watts[ss_start_i:i])})
            
    return pd.DataFrame(ss, index=idx)
