import yaml
from nilmtk.timeframe import TimeFrame
from io import open

MAX_MEM_ALLOWANCE_IN_BYTES = 2**28


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
    * could have subclasses for NILMTK HDF5, NILMTK CSV, Xively,
      Current Cost meters etc.
    * One DataStore per HDF5 file or folder or CSV files or Xively
      feed etc.

    Attributes
    ----------
    window : nilmtk.TimeFrame
        Defines the timeframe we are interested in.
    """
    def __init__(self):
        """
        Parameters
        ----------
        filename : string
        """
        self.window = TimeFrame()

    def __getitem__(self, key):
        """Loads all of a DataFrame from disk.

        Parameters
        ----------
        key : str

        Returns
        -------
        DataFrame

        Raises
        ------
        KeyError if `key` is not found.
        """
        raise NotImplementedError("NotImplementedError")

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, window):
        window.check_tz()
        self._window = window
        
    def load(self, key, columns=None, sections=None, n_look_ahead_rows=0,
             chunksize=MAX_MEM_ALLOWANCE_IN_BYTES):
        """
        Parameters
        ----------
        key : string, the location of a table within the DataStore.
        columns : list of Measurements, optional
            e.g. [('power', 'active'), ('power', 'reactive'), ('voltage')]
            if not provided then will return all columns from the table.
        sections : TimeFrameGroup; or list of nilmtk.TimeFrame objects;
            or a pd.PeriodIndex, optional.
            Defines the time sections to load.  If `self.window` is enabled
            then each `section` will be intersected with `self.window`.
        n_look_ahead_rows : int, optional, defaults to 0
            If >0 then each returned DataFrame will have a `look_ahead`
            property which will be a DataFrame of length `n_look_ahead_rows`
            of the data immediately in front of the data in the main DataFrame.
        chunksize : int, optional

        Returns
        ------- 
        generator of DataFrame objects
            Each DataFrame is has extra attributes:
                - timeframe : TimeFrame of section intersected with self.window
                - look_ahead : pd.DataFrame:
                    with `n_look_ahead_rows` rows.  The first row will be for
                    `section.end`.  `look_ahead` stores data which appears on 
                    disk immediately after `section.end`; i.e. it ignores
                    the next `section.start`.

            Returns an empty DataFrame if no data is available for the
            specified section (or if the section.intersection(self.window)
            is empty).

        Raises
        ------
        KeyError if `key` is not in store.
        """
        raise NotImplementedError("NotImplementedError")
        
    def append(self, key, value):
        """
        Parameters
        ----------
        key : str
        value : pd.DataFrame

        Notes
        -----
        To quote the Pandas documentation for pandas.io.pytables.HDFStore.append:
        Append does *not* check if data being appended overlaps with existing
        data in the table, so be careful.
        """
        raise NotImplementedError("NotImplementedError")
        
    def put(self, key, value):
        """
        Parameters
        ----------
        key : str
        value : pd.DataFrame
        """
        raise NotImplementedError("NotImplementedError")
        
    def remove(self, key, value):
        """
        Parameters
        ----------
        key : str
        value : pd.DataFrame
        """
        raise NotImplementedError("NotImplementedError")
        
    def load_metadata(self, key='/'):
        """
        Parameters
        ----------
        key : string, optional
            if '/' then load metadata for the whole dataset.

        Returns
        -------
        metadata : dict
        """
        raise NotImplementedError("NotImplementedError")
        
    def save_metadata(self, key, metadata):
        """
        Parameters
        ----------
        key : string
        metadata : dict
        """
        raise NotImplementedError("NotImplementedError")
        
    def elements_below_key(self, key='/'):
        """
        Returns
        -------
        list of strings
        """
    
    def close(self):
        raise NotImplementedError("NotImplementedError")

    def open(self):
        raise NotImplementedError("NotImplementedError")
        
    def get_timeframe(self, key):
        """
        Returns
        -------
        nilmtk.TimeFrame of entire table after intersecting with self.window.
        """
        raise NotImplementedError("NotImplementedError")


def write_yaml_to_file(metadata_filename, metadata):
    metadata_file = open(metadata_filename, 'w')
    yaml.dump(metadata, metadata_file)
    metadata_file.close()


def join_key(*args):
    """
    Examples
    --------
    >>> join_key('building1', 'elec', 'meter1')
    '/building1/elec/meter1'

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
        
def convert_datastore(input_store, output_store):
    """
    Parameters
    ----------
    input_store : nilmtk.DataStore
    output_store : nilmtk.DataStore
    """
    # dataset metadata
    metadata = input_store.load_metadata()
    output_store.save_metadata('/', metadata)
    for building in input_store.elements_below_key():
        building_key = '/'+building
        # building metadata
        metadata = input_store.load_metadata(building_key)
        output_store.save_metadata(building_key, metadata)
        for utility in input_store.elements_below_key(building):
            utility_key = building_key+'/'+utility
            for meter in input_store.elements_below_key(utility_key):
                # ignore cache (should this appear as an element below key?)
                if meter == 'cache':
                    continue
                meter_key = utility_key+'/'+meter
                # store meter data
                for df in input_store.load(meter_key):
                    output_store.append(meter_key, df)


