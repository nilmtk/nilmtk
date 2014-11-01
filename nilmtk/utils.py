from __future__ import print_function, division
import numpy as np
import pandas as pd
import networkx as nx
from copy import deepcopy
from os.path import isdir, dirname, abspath
from os import getcwd
from inspect import currentframe, getfile, getsourcefile
from sys import getfilesystemencoding

def timedelta64_to_secs(timedelta):
    """Convert `timedelta` to seconds.

    Parameters
    ----------
    timedelta : np.timedelta64

    Returns
    -------
    float : seconds
    """
    if len(timedelta) == 0:
        return np.array([])
    else:
        return timedelta / np.timedelta64(1, 's')


def tree_root(graph):
    """Returns the object that is the root of the tree.

    Parameters
    ----------
    graph : networkx.Graph
    """
    # from http://stackoverflow.com/a/4123177/732596
    assert isinstance(graph, nx.Graph)
    roots = [node for node,in_degree in graph.in_degree_iter() if in_degree==0]
    n_roots = len(roots)
    if n_roots > 1: 
        raise RuntimeError('Tree has more than one root!')
    if n_roots == 0: 
        raise RuntimeError('Tree has no root!')
    return roots[0]


def nodes_adjacent_to_root(graph):
    root = tree_root(graph)
    return graph.successors(root)


def index_of_column_name(df, name):
    for i, col_name in enumerate(df.columns):
        if col_name == name:
            return i
    raise KeyError(name)


def find_nearest(known_array, test_array):
    """Find closest value in `known_array` for each element in `test_array`.

    Parameters
    ----------
    known_array : numpy array
        consisting of scalar values only; shape: (m, 1)
    test_array : numpy array
        consisting of scalar values only; shape: (n, 1)

    Returns
    -------
    indices : numpy array; shape: (n, 1)
        For each value in `test_array` finds the index of the closest value
        in `known_array`.
    residuals : numpy array; shape: (n, 1)
        For each value in `test_array` finds the difference from the closest
        value in `known_array`.
    """
    # from http://stackoverflow.com/a/20785149/732596

    index_sorted = np.argsort(known_array)
    known_array_sorted = known_array[index_sorted]

    idx1 = np.searchsorted(known_array_sorted, test_array)
    idx2 = np.clip(idx1 - 1, 0, len(known_array_sorted)-1)
    idx3 = np.clip(idx1,     0, len(known_array_sorted)-1)

    diff1 = known_array_sorted[idx3] - test_array
    diff2 = test_array - known_array_sorted[idx2]

    indices = index_sorted[np.where(diff1 <= diff2, idx3, idx2)]
    residuals = test_array - known_array[indices]
    return indices, residuals


def container_to_string(container, sep='_'):
    if isinstance(container, str):
        string = container
    else:
        try:
            string = sep.join([str(element) for element in container])
        except TypeError:
            string = str(container)
    return string


def simplest_type_for(values):
    n_values = len(values)
    if n_values == 1:
        return list(values)[0]
    elif n_values == 0:
        return
    else:
        return tuple(values)


def flatten_2d_list(list2d):
    list1d = []
    for item in list2d:
        if isinstance(item, (list, set)):
            list1d.extend(item)
        else:
            list1d.append(item)
    return list1d


def get_index(data):
    """
    Parameters
    ----------
    data : pandas.DataFrame or Series or DatetimeIndex
    
    Returns
    -------
    index : the index for the DataFrame or Series
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        index = data.index
    elif isinstance(data, pd.DatetimeIndex):
        index = data
    else:
        raise TypeError('wrong type for `data`.')
    return index


def convert_to_timestamp(t):
    """
    Parameters
    ----------
    t : str or pd.Timestamp or datetime or None

    Returns
    -------
    pd.Timestamp or None
    """
    return None if t is None else pd.Timestamp(t)


def get_module_directory():
    # Taken from http://stackoverflow.com/a/6098238/732596
    path_to_this_file = dirname(getfile(currentframe()))
    if not isdir(path_to_this_file):
        encoding = getfilesystemencoding()
        path_to_this_file = dirname(unicode(__file__, encoding))
    if not isdir(path_to_this_file):
        abspath(getsourcefile(lambda _: None))
    if not isdir(path_to_this_file):
        path_to_this_file = getcwd()
    assert isdir(path_to_this_file), path_to_this_file + ' is not a directory'
    return path_to_this_file
