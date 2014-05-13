from __future__ import print_function, division
import pandas as pd
from nilmtk.timeframe import TimeFrame

MAX_MEM_ALLOWANCE_IN_BYTES = 1E9

class DataStore(object):
    """
    Provides a common interface to all physical data stores.  
    Supports hierarchical stores.
    
    The DataStore class lives in the bottom layer of NILMTK.  It loads
    a single chunk at a time from physical location and returns a
    DataFrame.

    * Deals with: retrieving data from disk / network / direct from a meter
    * Optimised for: handling large amounts of data
    * Services it provides: delivering a generator of pd.DataFrames of data given a
      specific time span and columns
    * Totally agnostic about what the data 'means'. It could be voltage,
      current, temperature, PIR readings etc.
    * could have subclasses for NILMTK HDF5, NILMTK CSV, Xively, REDD, iAWE,
      UKPD, etc; MetOffice XLS data, Current Cost meters etc.  
    * One DataStore per HDF5 file or folder or CSV files or Xively
      feed etc 

    Attributes
    ----------
    window : nilmtk.TimeFrame
        Defines the timeframe we are interested in.
    """
    def __init__(self):
        self.window = TimeFrame()


class HDFDataStore(DataStore):
    def __init__(self, filename):
        """
        Parameters
        ----------
        filename : string
        """
        self.store = pd.HDFStore(filename, 'r')
        super(HDFDataStore, self).__init__()

    def load(self, key, cols=None, periods=None, n_look_ahead_rows=10):
        """
        Parameters
        ----------
        key : string, the location of a table within the DataStore.
        cols : list of Measurements or ['index'], optional
            e.g. [Power('active'), Power('reactive'), Voltage()]
            if not provided then will return all columns from the table.
        periods : list of nilmtk.TimeFrame objects, optional
            defines the time periods to load.  If `self.window` is enabled
            then each `period` will be intersected with `self.window`.
        n_look_ahead_rows : int, optional, defaults to 10

        Returns
        ------- 
        Returns a generator of DataFrame objects.  Each DataFrame is:

        If `cols==['index']` then 
            each DF is a pd.DatetimeIndex
        else 
            returns a pd.DataFrame with extra attributes:
                - timeframe : TimeFrame of period intersected with self.window
                - look_ahead : pd.DataFrame:
                    with `n_look_ahead_rows` rows.  The first row will be for
                    `period.end`.  `look_ahead` stores data which appears on 
                    disk immediately after `period.end`; i.e. it ignores
                    the next `period.start`.

            Returns an empty DataFrame if no data is available for the
            specified period (or if the period.intersect(self.window)
            is empty).

        Raises
        ------
        MemoryError if we try to load too much data.
        """
        self._check_key(key)
        self._check_columns(key, cols)
        periods = [TimeFrame()] if periods is None else periods

        start_row = 0 # row to start search in table
        for period in periods:
            window_intersect = self.window.intersect(period)
            if window_intersect.empty:
                data = pd.DataFrame()
            else:
                terms = window_intersect.query_terms('window_intersect')
                coords = self.store.select_as_coordinates(
                    key=key, where=terms, 
                    start=None if start_row is 0 else start_row,
                    stop=None if start_row is 0 else -1)
                if len(coords) > 0:
                    self._check_data_will_fit_in_memory(
                        key=key, nrows=len(coords), cols=cols)
                    data = self.store.select(key=key, where=coords, columns=cols)
                    start_row = coords[-1]+1
                else:
                    data = pd.DataFrame()
            if cols == ['index']:
                data = data.index

            data.timeframe = (window_intersect if window_intersect 
                              else self._get_timeframe(key))

            # Load 'look ahead'
            try:
                look_ahead_coords = self.store.select_as_coordinates(
                    key=key, where="index >= period.end", 
                    start=start_row, 
                    stop=start_row+n_look_ahead_rows)
            except IndexError:
                look_ahead_coords = []

            if len(look_ahead_coords) > 0:
                data.look_ahead = self.store.select(
                    key=key, where=look_ahead_coords, columns=cols)
            else:
                data.look_ahead = pd.DataFrame()

            yield data

    def load_metadata(self, key='/'):
        """
        Parameters
        ----------
        key : string, optional
            if None then load metadata for the whole dataset.

        Returns
        -------
        metadata : dict
        """
        if key == '/':
            node = self.store.root
        else:
            node = self.store.get_node(key)

        metadata = node._v_attrs.metadata
        return metadata

    def save_metadata(self, key, metadata):
        """
        Parameters
        ----------
        key : string
        metadata : dict
        """

        if key == '/':
            node = self.store.root
        else:
            node = self.store.get_node(key)

        node._v_attrs.metadata = metadata
        self.store.flush()

    def elements_below_key(self, key='/'):
        """
        Returns
        -------
        list of strings
        """
        if key == '/' or not key:
            node = self.store.root
        else:
            node = self.store.get_node(key)
        return node._v_children.keys()

    def close(self):
        self.store.close()

    def open(self):
        self.store.close()
    
    def _check_columns(self, key, columns):
        if columns is None:
            return
        if not self._table_has_column_names(key, columns):
            raise KeyError('at least one of ' + str(columns) + 
                           ' is not a valid column')

    def _table_has_column_names(self, key, cols):
        """
        Parameters
        ----------
        cols : string or list of strings
        
        Returns
        -------
        boolean
        """
        assert cols is not None
        self._check_key(key)
        if isinstance(cols, str):
            cols = [cols]
        query_cols = set(cols)
        table_cols = set(self._column_names(key) + ['index'])
        return query_cols.issubset(table_cols)

    def _column_names(self, key):
        self._check_key(key)
        storer = self._get_storer(key)
        col_names = storer.non_index_axes[0][1:][0]
        return col_names

    def _check_data_will_fit_in_memory(self, key, nrows, cols=None):
        # Check we won't use too much memory
        mem_requirement = self._estimate_memory_requirement(key, nrows, cols)
        if mem_requirement > MAX_MEM_ALLOWANCE_IN_BYTES:
            raise MemoryError('Requested data would use {:.3f}MBytes:'
                              ' too much memory.'
                              .format(mem_requirement / 1E6))

    def _estimate_memory_requirement(self, key, nrows, cols=None):
        """Returns estimated mem requirement in bytes."""
        BYTES_PER_ELEMENT = 4
        BYTES_PER_TIMESTAMP = 8
        self._check_key(key)
        if cols is None:
            cols = self._column_names(key)
        else:
            self._check_columns(key, cols)
        ncols = len(cols)
        est_mem_usage_for_data = nrows * ncols * BYTES_PER_ELEMENT
        est_mem_usage_for_index = nrows * BYTES_PER_TIMESTAMP
        if cols == ['index']:
            return est_mem_usage_for_index
        else:
            return est_mem_usage_for_data + est_mem_usage_for_index
       
    def _nrows(self, key, timeframe=None):
        """
        Returns
        -------
        nrows : int
        """
        self._check_key(key)
        timeframe_intersect = self.window.intersect(timeframe)
        if timeframe_intersect.empty:
            nrows = 0
        elif timeframe_intersect:
            terms = timeframe_intersect.query_terms('timeframe_intersect')
            coords = self.store.select_as_coordinates(key, terms)
            nrows = len(coords)
        else:
            storer = self._get_storer(key)
            nrows = storer.nrows
        return nrows
        
    def _get_timeframe(self, key):
        """
        Returns
        -------
        nilmtk.TimeFrame of entire table after intersecting with self.window.
        """
        self._check_key(key)
        data_start_date = self.store.select(key, [0]).index[0]
        data_end_date = self.store.select(key, start=-1).index[0]
        timeframe = TimeFrame(data_start_date, data_end_date)
        return self.window.intersect(timeframe)
    
    def _keys(self):
        return self.store.keys()

    def _get_storer(self, key):
        self._check_key(key)
        storer = self.store.get_storer(key)
        assert storer is not None, "cannot get storer for key = " + key
        return storer
    
    def _check_key(self, key):
        if key not in self._keys():
            raise KeyError(key + ' not in store')



def join_key(*args):
    """
    Examples
    --------
    >>> join_key('building1', 'electric', 'meter1')
    '/building1/electric/meter1'

    >>> join_key('/')
    '/'

    >>> join_key('')
    '/'
    """
    key = '/'
    for arg in args:
        arg_stripped = str(arg).strip('/')
        if arg_stripped:
            key += arg_stripped + '/'
    if len(key) > 1:
        key = key[:-1] # remove last trailing slash
    return key
