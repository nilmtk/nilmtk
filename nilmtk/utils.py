from os.path import isdir, dirname, abspath
from os import getcwd
from inspect import currentframe, getfile, getsourcefile
from sys import getfilesystemencoding, stdout
from collections import OrderedDict, defaultdict
import datetime, re
import pytz
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from IPython.core.display import HTML, display
from sklearn.metrics import mean_squared_error

from nilmtk.datastore import HDFDataStore, CSVDataStore

def show_versions():
    """Prints versions of various dependencies"""
    output = OrderedDict()
    output["Date"] = str(datetime.datetime.now())
    import sys
    import platform
    output["Platform"] = str(platform.platform())
    system_information = sys.version_info
    output["System version"] = "{}.{}".format(system_information.major,
                                              system_information.minor)

    PACKAGES = [
        "nilmtk", "nilm_metadata", "numpy", "matplotlib", "pandas", "sklearn",
        "hmmlearn"]
    for package_name in PACKAGES:
        key = package_name + " version"
        try:
            exec("import " + package_name)
        except ImportError:
            output[key] = "Not found"
        else:
            output[key] = eval(package_name + ".__version__")

    try:
        print(pd.show_versions())
    except:
        pass
    else:
        print("")

    for k, v in output.items():
        print("{}: {}".format(k, v))


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
    roots = [node for node, in_degree in graph.in_degree()
             if in_degree == 0]
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
        if isinstance(item, str):
            list1d.append(item)
        else:
            try:
                len(item)
            except TypeError:
                list1d.append(item)
            else:
                list1d.extend(item)
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
            if isinstance(value, str) and 'http' in value:
                html = re.sub(r'(http[^\s\)]+)', r'<a href="\1">\1</a>', value)
            else:
                html = '{}'.format(value)
        except UnicodeEncodeError:
            html = ''
        return html

    html = '<ul>'
    for key, value in dictionary.items():
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
    if timestamp is None or pd.isnull(timestamp):
        return pd.NaT

    timestamp = pd.Timestamp(timestamp)
    if timestamp_is_naive(timestamp):
        timestamp = timestamp.tz_localize('UTC')

    return timestamp.tz_convert(tz)


def get_tz(df):
    index = df.index
    try:
        tz = index.tz
    except AttributeError:
        tz = None
    return tz


def timestamp_is_naive(timestamp):
    """
    Parameters
    ----------
    timestamp : pd.Timestamp or datetime.datetime

    Returns
    -------
    True if `timestamp` is naive (i.e. if it does not have a
    timezone associated with it).  See:
    https://docs.python.org/3/library/datetime.html#available-types
    """
    if timestamp.tzinfo is None:
        return True
    elif timestamp.tzinfo.utcoffset(timestamp) is None:
        return True
    else:
        return False


def get_datastore(filename, format=None, mode='r'):
    """
    Parameters
    ----------
    filename : string
    format : 'CSV' or 'HDF', default: infer from filename ending.
    mode : 'r' (read-only), 'a' (append) or 'w' (write), default: 'r'

    Returns
    -------
    metadata : dict
    """
    if not format:
        if filename.endswith(".h5"):
            format = "HDF"
        elif filename.endswith(".csv"):
            format = "CSV"

    if filename is not None:
        if format == "HDF":
            return HDFDataStore(filename, mode)
        elif format == "CSV":
            return CSVDataStore(filename)
        else:
            raise ValueError('format not recognised')
    else:
        ValueError('filename is None')


def normalise_timestamp(timestamp, freq):
    """Returns the nearest Timestamp to `timestamp` which would be
    in the set of timestamps returned by pd.DataFrame.resample(freq=freq)
    """
    timestamp = pd.Timestamp(timestamp)
    series = pd.Series(np.NaN, index=[timestamp])
    resampled = series.resample(freq).mean()
    return resampled.index[0]


def print_on_line(*strings):
    print(*strings, end="")
    stdout.flush()


def append_or_extend_list(lst, value):
    if value is None:
        return
    elif isinstance(value, list):
        lst.extend(value)
    else:
        lst.append(value)


def convert_to_list(list_like):
    return [] if list_like is None else list(list_like)


def most_common(lst):
    """Returns the most common entry in lst."""
    lst = list(lst)
    counts = {item: lst.count(item) for item in set(lst)}
    counts = pd.Series(counts)
    counts.sort()
    most_common = counts.index[-1]
    return most_common


def capitalise_first_letter(string):
    return string[0].upper() + string[1:]


def capitalise_index(index):
    labels = list(index)
    for i, label in enumerate(labels):
        labels[i] = capitalise_first_letter(label)
    return labels


def capitalise_legend(ax):
    legend_handles = ax.get_legend_handles_labels()
    labels = capitalise_index(legend_handles[1])
    ax.legend(legend_handles[0], labels)
    return ax


def compute_rmse(ground_truth, predictions, pretty=True):
    """
    Compute the RMS error between the time-series of appliance
    ground truth values and predicted values.

    Parameters
    ----------
    ground_truth : `pandas.DataFrame` containing the ground truth series 
                  for the appliances.
    
    predictions : `pandas.DataFrame` containing the predicted time-series
                  for each appliance. If a appliance is present in 
                  `ground_truth` but absent in `predictions` (or only
                  contains NA values), it is not listed in the output.
    
    pretty : If `True`, tries to use the appliance labels if possible. If
             a type of appliance is present more than once, resulting in
             duplicate labels, building and instance number are added 
             to differentiate them. 
    
    Returns
    -------
    pandas.Series with the RMSe for each appliance
    """ 
    
    # This was initially added to simplify examples, see #652.
    
    rms_error = []
    app_counts = defaultdict(int)
    for app_idx, app in enumerate(ground_truth.columns):
        if pretty:
            try:
                app_label = app.label()
            except:
                pretty = False
                app_label = app
        else:
            app_label = app
        
        gt_app = ground_truth.iloc[:, app_idx]
        pred_app = predictions.iloc[:, app_idx]
        if pred_app.empty: 
            continue
            
        df_app = pd.DataFrame({'gt': gt_app, 'pr': pred_app}, index=gt_app.index).dropna()
        
        if not df_app.empty:
            app_rms_error = np.sqrt(mean_squared_error(df_app['gt'], df_app['pr']))
        else:
            app_rms_error = np.NaN

        if pretty:
            app_counts[app_label] += 1
            
        rms_error.append([app, app_label, app_rms_error])

    if pretty:
        for current_label, current_count in app_counts.items():
            if current_count < 2:
                continue

            # A loop should be fine for such small lists
            for app_data in rms_error:
                if app_data[1] != current_label:
                    continue

                app = app_data[0]
                app_data[1] = '{} ({}, {})'.format(
                    current_label,
                    app.building,
                    app.instance
                )
            
    return pd.Series(dict(
        (item[1], item[2]) for item in rms_error
    ))


def safe_resample(data, **resample_kwargs):
    if data.empty:
        return data
    
    def _resample_chain(data, all_resample_kwargs):
        """_resample_chain provides a compatibility function for 
        deprecated/removed DataFrame.resample kwargs"""
        

        rule = all_resample_kwargs.pop('rule')
        axis = all_resample_kwargs.pop('axis', None)
        on = all_resample_kwargs.pop('on', None)
        level = all_resample_kwargs.pop('level', None)

        resample_kwargs = {}
        if axis is not None: resample_kwargs['axis'] = axis
        if on is not None: resample_kwargs['on'] = on
        if level is not None: resample_kwargs['level'] = level

        fill_method_str = all_resample_kwargs.pop('fill_method', None)
        if fill_method_str:
            limit = all_resample_kwargs.pop('limit', None)
            fill_method = lambda df: getattr(df, fill_method_str)(limit=limit)
        else:
            fill_method = lambda df: df
            
        how_str = all_resample_kwargs.pop('how', None)
        if how_str:
            how = lambda df: getattr(df, how_str)()
        else:
            how = lambda df: df

        if all_resample_kwargs:
            warnings.warn("Not all resample_kwargs were consumed: {}".format(repr(all_resample_kwargs))) 
        
        return fill_method(how(data.resample(rule, **resample_kwargs)))
       

    try:
        dups_in_index = data.index.duplicated(keep='first')

        #TODO: remove this after validation tests are ready
        if dups_in_index.any():
            warnings.warn("Found duplicate index. Keeping first value")
            data = data[~dups_in_index]
            
        data = _resample_chain(data, resample_kwargs)
    except pytz.AmbiguousTimeError:
        # Work-around for
        # https://github.com/pydata/pandas/issues/10117
        tz = data.index.tz.zone
        data = data.tz_convert('UTC')
        data = _resample_chain(data, resample_kwargs)
        
        data = data.tz_convert(tz)
    return data
