from __future__ import print_function, division
import os, copy
import numpy as np
import pandas as pd
from scipy import stats

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


def is_namedtuple(obj, nt):
    """Returns true if obj is a namedtuple of type nt.

    Does what you might expect `isinstance(obj, nt)` to do, but doesn't.
    """
    # we can't use isinstance on NamedTuples like isinstance(col_name, DualSupply)
    # see http://bugs.python.org/issue7796
    try:
        for field in nt._fields:
            obj.__dict__[field]
    except (AttributeError, KeyError):
        return False
    else:
        return True


def recursive_resolve(obj, dict_name):
    """Returns `obj.dict_name` where `dict_name` is a string
    which may have periods e.g. `utility.electric.appliances`.

    Parameters
    ----------
    obj : object
        
    dict_name : string
        e.g. 'utility.electric.appliances'

    Returns
    -------
    obj.dict_name

    Examples
    --------
    To get buildling.utility.electric.appliances:

    >>> appliances = recursive_resolve(building, 'utility.electric.appliances')
    """
    partitions = dict_name.partition('.')
    if not partitions[0]:
        return
    elif not partitions[2]:
        # e.g. partitions = ('electric', '', '')
        return obj.__dict__[dict_name]
    else:
        return recursive_resolve(obj.__dict__[partitions[0]], partitions[2])



def apply_func_to_values_of_dicts(obj, func, dict_names):
    """Apply a generic function `func` to all values of a set dicts, 
    each of which is an attribute of an arbitrary object `obj`.

    Parameters
    ----------
    obj : object
        any object which has one or more dicts as attributes
    func : function
        the function to apply to each dict value
    dict_names : list of strings
        the attribute names of the dicts in `obj`

    Returns
    -------
    obj_copy : a deepcopy of `obj` with `func` applied to all `obj.<dict_names>`

    Examples
    --------
    For example, to apply `resample` to the `circuits` and `mains` dicts of
    an Electricity object:

    >>> resample = lambda df : pd.DataFrame.resample(df, rule='T')
    >>> electric = apply_func_to_values_of_dicts(electric, 
                                                 resample,
                                                 ['circuits', 'mains'])
    """

    # TODO: a lot of functions in nilmtk.preprocessing.electricity.buildling
    # could be simplified using `apply_to_values_of_dicts`

    obj_copy = copy.deepcopy(obj)
    for attribute in dict_names:
        dict_ = recursive_resolve(obj_copy, attribute)
        for key, value in dict_.iteritems():
            dict_[key] = func(value)
    return obj_copy


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


def summary_stats_string(data, fmt='{:>6.2f}'):
    data = np.array(data)
    s = ''
    # use eval, use loop
    # numpy stat_strings
    stat_strings = ['min', 'mean', 'mode', 'max', 'std']
    scipy_stats = ['mode']
    for stat_str in stat_strings:
        if stat_str in scipy_stats:
            stat = stats.__dict__[stat_str](data)[0][0]
        else:
            stat = data.__getattribute__(stat_str)()
        s += '  {:5s}'.format(stat_str) + ' = '
        s += fmt.format(stat)
        s += '\n'
    
    return s
