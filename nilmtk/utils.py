from __future__ import print_function, division
import numpy as np
import pandas as pd
import networkx as nx
from copy import deepcopy
from os.path import isdir, dirname, abspath
from os import getcwd
from inspect import currentframe, getfile, getsourcefile
from sys import getfilesystemencoding
from IPython.core.display import HTML, display
from collections import OrderedDict
import datetime


def dependencies_diagnostics():
    """Prints versions of various dependencies"""
    output= OrderedDict()
    output["Date"] = str(datetime.datetime.now())
    import sys 
    import platform
    output["Platform"] = str(platform.platform())
    system_information = sys.version_info
    output["System version"] = str(system_information.major) + "." +  str(system_information.minor)
    import nilmtk
    output["nilmtk version"] = nilmtk.__version__
    try:
        import nilm_metadata
        output["nilm_metadata version"] = nilm_metadata.__version__
    except ImportError:
        output["nilm_metadata version"] = "Not found"
    try:
        import numpy as np
        output["Numpy version"] = np.version.version
    except ImportError:
        output["Numpy version"] = "Not found"
    try:
        import matplotlib
        output["Matplotlib version"] = matplotlib.__version__
    except ImportError:
        output["Matplotlib version"] = "Not found"
    try:
        import pandas as pd 
        output["Pandas version"] = pd.__version__
    except ImportError:
        output["Pandas version"] = "Not found"
    try:
        import sklearn as sklearn
        output["Scikit-learn version"] = sklearn.__version__
    except ImportError:
        output["Scikit-learn version"] = "Not found"
    return output

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


def dict_to_html(dictionary):
    def format_string(value):
        try:
            if isinstance(value, basestring) and 'http' in value:
                html = '<a href="{url}">{url}</a>'.format(url=value)
            else:
                html = '{}'.format(value)
        except UnicodeEncodeError:
            html = ''
        return html

    html = '<ul>'
    for key, value in dictionary.iteritems():
        html += '<li><strong>{}</strong>: '.format(key)
        if isinstance(value, list):
            html += '<ul>'
            for item in value:
                html += '<li>{}</li>'.format(format_string(item))
            html += '</ul>'
        elif isinstance(value, dict):
            html += dict_to_html(value)
        else:
            html += format_string(value)
        html += '</li>'
    html += '</ul>'
    return html

    
def print_dict(dictionary):
    html = dict_to_html(dictionary)
    display(HTML(html))


def offset_alias_to_seconds(alias):
    """Seconds for each period length."""
    dr = pd.date_range('00:00', periods=2, freq=alias)
    return (dr[-1] - dr[0]).total_seconds()


def check_directory_exists(d):
    if not isdir(d):
        raise IOError("Directory '{}' does not exist.".format(d))


def tz_localize_naive(timestamp, tz):
    if tz is None:
        return timestamp
    elif pd.isnull(timestamp):
        return pd.NaT
    else:
        return timestamp.tz_localize('UTC').tz_convert(tz)


def get_tz(df):
    index = df.index
    try:
        tz = index.tz
    except AttributeError:
        tz = None
    return tz
