from __future__ import print_function, division
import numpy as np
import networkx as nx
from copy import deepcopy


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


def find_nearest_vectorized(known_array, test_array):
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
    # see http://stackoverflow.com/q/20780017/732596
    # TODO: might be much faster to use HYRY's method: 
    #       http://stackoverflow.com/a/20785149/732596
    differences = test_array.reshape(1, -1) - known_array.reshape(-1, 1)
    indices = np.abs(differences).argmin(axis=0)
    residuals = np.diagonal(differences[indices, ])
    return indices, residuals
