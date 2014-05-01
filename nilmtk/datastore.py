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
    * Services it provides: delivering a pd.DataFrame of data given a
      specific time span and columns
    * Totally agnostic about what the data 'means'. It could be voltage,
      current, temperature, PIR readings etc.
    * subclasses for NILMTK HDF5, NILMTK CSV, Xively, REDD, iAWE,
      UKPD, etc; MetOffice XLS data, Current Cost meters etc.  
    * One DataStore per HDF5 file or folder or CSV files or Xively
      feed etc 
    * always use JSON for metadata

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
        self.store = pd.HDFStore(filename)
        super(HDFDataStore, self).__init__()

    def load(self, key, cols=None, period=None):
        """
        Parameters
        ----------
        key : string, the location of a table within the DataStore.
        cols : list or 'index', optional
            e.g. [('power', 'active'), ('power', 'reactive'), ('voltage', '')]
            if not provided then will return all columns from the table.
        period : nilmtk.TimeFrame, optional
            defines the time period to load.  If `self.window` is enabled
            then `period` will be intersected with `self.window`.

        Returns
        ------- 
        If `cols=='index'` then returns a pd.DatetimeIndex
        else returns a pd.DataFrame.
        Returns an empty DataFrame if no data is available for the
        specified period (or if the period.intersect(self.window)
        is empty).
        Returned DataFrame has a `timeframe` attribute.

        Raises
        ------
        MemoryError if we try to load too much data.
        """
        self._check_key(key)
        self._check_columns(key, cols)
        window_intersect = self.window.intersect(period)
        if window_intersect.empty:
            df = pd.DataFrame()
            df.timeframe = window_intersect
            return df
        
        # Check we won't use too much memory
        mem_requirement = self.estimate_memory_requirement(key, cols, 
                                                           window_intersect)
        if mem_requirement > MAX_MEM_ALLOWANCE_IN_BYTES:
            raise MemoryError('Requested data would use {:.3f}MBytes:'
                              ' too much memory.'
                              .format(mem_requirement / 1E6))

        # Create list of query terms
        terms = window_intersect.query_terms('window_intersect')
        if cols is not None:
            terms.append("columns==cols")
        if terms == []:
            terms = None

        # Read data
        data = self.store.select(key=key, where=terms)
        if cols == 'index':
            data = data.index
        data.timeframe = window_intersect
        return data
    
    def _check_columns(self, key, columns):
        if columns is None:
            return
        if not self.table_has_column_names(key, columns):
            raise KeyError('at least one of ' + str(columns) + 
                           ' is not a valid column')

    def close(self):
        self.store.close()

    def open(self):
        self.store.close()

    def table_has_column_names(self, key, cols):
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
        table_cols = set(self.column_names(key) + ['index'])
        return query_cols.issubset(table_cols)
    
    def estimate_memory_requirement(self, key, cols=None, timeframe=None):
        """Returns estimated mem requirement in bytes."""
        BYTES_PER_ELEMENT = 4
        BYTES_PER_TIMESTAMP = 8
        self._check_key(key)
        if cols is None:
            cols = self.column_names(key)
        else:
            self._check_columns(key, cols)
        ncols = len(cols)
        nrows = self.nrows(key, timeframe)
        est_mem_usage_for_data = nrows * ncols * BYTES_PER_ELEMENT
        est_mem_usage_for_index = nrows * BYTES_PER_TIMESTAMP
        return est_mem_usage_for_data + est_mem_usage_for_index
    
    def column_names(self, key):
        self._check_key(key)
        storer = self._get_storer(key)
        col_names = storer.non_index_axes[0][1:][0]
        return col_names
    
    def nrows(self, key, timeframe=None):
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
            if terms == []:
                terms = None
            coords = self.store.select_as_coordinates(key, terms)
            nrows = len(coords)
        else:
            storer = self._get_storer(key)
            nrows = storer.nrows
        return nrows
        
    def timeframe(self, key):
        """
        Returns
        -------
        nilmtk.TimeFrame
        """
        self._check_key(key)
        data_start_date = self.store.select(key, [0]).index[0]
        data_end_date = self.store.select(key, start=-1).index[0]
        timeframe = TimeFrame(data_start_date, data_end_date)
        return self.window.intersect(timeframe)
    
    def keys(self):
        return self.store.keys()

    def _get_storer(self, key):
        self._check_key(key)
        storer = self.store.get_storer(key)
        assert storer is not None, "cannot get storer for key = " + key
        return storer
    
    def _check_key(self, key):
        if key not in self.keys():
            raise KeyError(key + ' not in store')

    def load_metadata(self, key=None):
        """
        Parameters
        ----------
        key : string, optional
            if None then load metadata for the whole dataset.

        Returns
        -------
        metadata : dict
        """
        if key is None:
            metadata = self.store.root._v_attrs.dataset            
        else:
            metadata = self._get_storer(key).attrs.metadata
        return metadata

    def save_metadata(self, key, metadata):
        """
        Parameters
        ----------
        key : string
        metadata : dict
        """
        self._get_storer(key).attrs.metadata = metadata
        self.store.flush()
