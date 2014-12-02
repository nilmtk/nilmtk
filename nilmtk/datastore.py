from __future__ import print_function, division
import pandas as pd
from itertools import repeat, tee
from time import time
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import yaml
from os.path import isdir, isfile, join, exists, dirname
from os import listdir, makedirs, remove
from shutil import rmtree
import re
from nilm_metadata.convert_yaml_to_hdf5 import _load_file
from .timeframe import TimeFrame, timeframes_from_periodindex
from .node import Node

# do not edit! added by PythonBreakpoints
from pdb import set_trace as _breakpoint


MAX_MEM_ALLOWANCE_IN_BYTES = 2**29 # 512 MBytes


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
        self.window = TimeFrame()

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, window):
        window.check_tz()
        self._window = window

class HDFDataStore(DataStore):
    def __init__(self, filename, mode='a'):
        """
        Parameters
        ----------
        filename : string
        mode : string, default='a'
            File open mode.  e.g. 'a', 'r' or 'w'
        """
        self.store = pd.HDFStore(filename, mode, complevel=9, complib='blosc')
        super(HDFDataStore, self).__init__()

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
        return self.store[key]

    def load(self, key, cols=None, sections=None, n_look_ahead_rows=0,
             chunksize=MAX_MEM_ALLOWANCE_IN_BYTES):
        """
        Parameters
        ----------
        key : string, the location of a table within the DataStore.
        cols : list of Measurements, optional
            e.g. [('power', 'active'), ('power', 'reactive'), ('voltage')]
            if not provided then will return all columns from the table.
        sections : list of nilmtk.TimeFrame objects or a pd.PeriodIndex, optional
            defines the time sections to load.  If `self.window` is enabled
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
            specified section (or if the section.intersect(self.window)
            is empty).

        Raises
        ------
        KeyError if `key` is not in store.
        """
        # TODO: calculate chunksize default based on physical 
        # memory installed and number of columns

        # Make sure key has a slash at the front but not at the end.
        if key[0] != '/':
            key = '/' + key
        if len(key) > 1 and key[-1] == '/':
            key = key[:-1]

        # Make sure chunksize is an int otherwise `range` complains later.
        chunksize = np.int64(chunksize)

        # Set `sections` variable
        sections = [TimeFrame()] if sections is None else sections
        if isinstance(sections, pd.PeriodIndex):
            sections = timeframes_from_periodindex(sections)

        self.all_sections_smaller_than_chunksize = True

        for section in sections:
            window_intersect = self.window.intersect(section)
            if window_intersect.empty:
                continue
            terms = window_intersect.query_terms('window_intersect')
            try:
                coords = self.store.select_as_coordinates(key=key, where=terms)
            except AttributeError as e:
                if str(e) == ("'NoneType' object has no attribute "
                              "'read_coordinates'"):
                    raise KeyError("key '{}' not found".format(key))
                else:
                    raise
            n_coords = len(coords)
            if n_coords == 0:
                continue
            section_start_i = coords[0]
            section_end_i   = coords[-1]
            del coords
            slice_starts = range(section_start_i, section_end_i, chunksize)
            n_chunks = len(slice_starts)
            for chunk_i, chunk_start_i in enumerate(slice_starts):
                chunk_end_i = chunk_start_i + chunksize
                if chunk_end_i > section_end_i:
                    chunk_end_i = section_end_i
                chunk_end_i += 1

                data = self.store.select(key=key, cols=cols, 
                                         start=chunk_start_i, stop=chunk_end_i)

                if len(data) <= 2:
                    continue

                if chunk_i > 0:
                    self.all_sections_smaller_than_chunksize = False

                # Load look ahead if necessary
                if n_look_ahead_rows > 0:
                    if len(data.index) > 0:
                        look_ahead_start_i = chunk_end_i
                        look_ahead_end_i = look_ahead_start_i + n_look_ahead_rows
                        try:
                            data.look_ahead = self.store.select(
                                key=key, cols=cols, 
                                start=look_ahead_start_i,
                                stop=look_ahead_end_i)
                        except ValueError:
                            data.look_ahead = pd.DataFrame()
                    else:
                        data.look_ahead = pd.DataFrame()

                # Set timeframe
                start = None
                end = None

                # Test if there are any more subchunks
                there_are_more_subchunks = (chunk_i < n_chunks-1)
                if there_are_more_subchunks:
                    if chunk_i == 0:
                        start = window_intersect.start
                elif chunk_i > 0:
                    # This is the last subchunk
                    end = window_intersect.end
                else:
                    # Just a single 'subchunk'
                    start = window_intersect.start
                    end = window_intersect.end

                if start is None:
                    start = data.index[0]
                if end is None:
                    end = data.index[-1]

                data.timeframe = TimeFrame(start, end)
                yield data

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
        self.store.append(key=key, value=value)
        self.store.flush()

    def put(self, key, value):
        """
        Parameters
        ----------
        key : str
        value : pd.DataFrame
        """
        self.store.put(key, value, format='table', 
                       expectedrows=len(value), index=False)
        self.store.create_table_index(key, columns=['index'], 
                                      kind='full', optlevel=9)
        self.store.flush()

    def remove(self, key):
        self.store.remove(key)

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
        if key == '/':
            node = self.store.root
        else:
            node = self.store.get_node(key)

        metadata = deepcopy(node._v_attrs.metadata)
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
        
    def get_timeframe(self, key):
        """
        Returns
        -------
        nilmtk.TimeFrame of entire table after intersecting with self.window.
        """
        data_start_date = self.store.select(key, [0]).index[0]
        data_end_date = self.store.select(key, start=-1).index[0]
        timeframe = TimeFrame(data_start_date, data_end_date)
        return self.window.intersect(timeframe)
    
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

    def _estimate_memory_requirement(self, key, nrows, cols=None, paranoid=False):
        """Returns estimated mem requirement in bytes."""
        BYTES_PER_ELEMENT = 4
        BYTES_PER_TIMESTAMP = 8
        if paranoid:
            self._check_key(key)
        if cols is None:
            cols = self._column_names(key)
        elif paranoid:
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
    
    def _keys(self):
        return self.store.keys()

    def _get_storer(self, key):
        self._check_key(key)
        storer = self.store.get_storer(key)
        assert storer is not None, "cannot get storer for key = " + key
        return storer
    
    def _check_key(self, key):
        """
        Parameters
        ----------
        key : string
        """
        if key not in self._keys():
            raise KeyError(key + ' not in store')


class CSVDataStore(DataStore):
    def __init__(self, filename):
        """
        Parameters
        ----------
        filename : string
        """
        self.filename = filename
        # make root directory
        path = self._key_to_abs_path('/')
        if not exists(path):
            makedirs(path)
        # make metadata directory
        path = self._get_metadata_path()
        if not exists(path):
            makedirs(path)
        super(CSVDataStore, self).__init__()

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
        file_path = self._key_to_abs_path(key)
        if isfile(file_path):
            return pd.read_csv(file_path)
        else:
            raise KeyError('{} not found'.format(key))

    def load(self, key, cols=None, sections=None, n_look_ahead_rows=0,
             chunksize=MAX_MEM_ALLOWANCE_IN_BYTES):
        """
        Parameters
        ----------
        key : string, the location of a table within the DataStore.
        cols : list of Measurements, optional
            e.g. [('power', 'active'), ('power', 'reactive'), ('voltage')]
            if not provided then will return all columns from the table.
        n_look_ahead_rows : not used yet
        chunksize : int, optional

        Returns
        ------- 
        TextFileReader of DataFrame objects
        TODO: do something with args: sections and n_look_ahead_rows
        """
        
        file_path = self._key_to_abs_path(key)
        
        # Set `sections` variable
        sections = [TimeFrame()] if sections is None else sections
        if isinstance(sections, pd.PeriodIndex):
            sections = timeframes_from_periodindex(sections)

        self.all_sections_smaller_than_chunksize = True
        
        # iterate through parameter sections
        for section in sections:
            window_intersect = self.window.intersect(section)
            header_rows = [0,1]
            text_file_reader = pd.read_csv(file_path, 
                                            index_col=0, 
                                            header=header_rows, 
                                            parse_dates=True, 
                                            #usecols=cols,
                                            chunksize=chunksize)
            # iterate through all chunks in file
            for chunk_idx, chunk in enumerate(text_file_reader):
                chunk.timeframe = TimeFrame(chunk.index[0], chunk.index[-1])
                chunk_intersect = window_intersect.intersect(chunk.timeframe)
                # if chunk intersects with section
                if not chunk_intersect.empty:
                    subchunk = chunk[(chunk.index>=chunk_intersect.start) & 
                                    (chunk.index<=chunk_intersect.end)]
                    subchunk.timeframe = TimeFrame(subchunk.index[0], subchunk.index[-1])
                    
                    # Load look ahead if necessary
                    if n_look_ahead_rows > 0:
                        if len(subchunk.index) > 0:
                            rows_to_skip = (chunk_idx+1)*MAX_MEM_ALLOWANCE_IN_BYTES+len(header_rows)+1
                            subchunk.look_ahead = pd.read_csv(file_path, 
                                            index_col=0, 
                                            header=None, 
                                            parse_dates=True,
                                            skiprows=rows_to_skip,
                                            nrows=n_look_ahead_rows)
                        else:
                            subchunk.look_ahead = pd.DataFrame()

                    print(subchunk)
                    print(subchunk.look_ahead)
                    import ipdb
                    ipdb.set_trace()
                    
                    yield subchunk

    def append(self, key, value):
        """
        Parameters
        ----------
        key : str
        value : pd.DataFrame
        """
        print(key)
        file_path = self._key_to_abs_path(key)
        path = dirname(file_path)
        if not exists(path):
            makedirs(path)
        value.to_csv(file_path,
                    mode='a',
                    header=True)
                    
    def put(self, key, value):
        print(key)
        """
        Parameters
        ----------
        key : str
        value : pd.DataFrame
        """
        file_path = self._key_to_abs_path(key)
        path = dirname(file_path)
        if not exists(path):
            makedirs(path)
        value.to_csv(file_path,
                    mode='w',
                    header=True)
                    
    def remove(self, key):
    	file_path = self._key_to_abs_path(key)
    	if isfile(file_path):
    	    remove(file_path)
        else:
            rmtree(file_path)

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
        if key == '/':
            filepath = self._get_metadata_path()
            metadata = _load_file(filepath, 'dataset.yaml')
            meter_devices = _load_file(filepath, 'meter_devices.yaml')
            metadata['meter_devices'] = meter_devices
        else:
            key_object = Key(key)
            if key_object.building and not key_object.meter:
                # load building metadata from file
                filename = 'building'+str(key_object.building)+'.yaml'
                filepath = self._get_metadata_path()
                metadata = _load_file(filepath, filename)
                # set data_location
                for meter_instance in metadata['elec_meters']:
                    # not sure why I need to use meter_instance-1
                    data_location = '/building{:d}/elec/meter{:d}'.format(key_object.building, meter_instance)
                    metadata['elec_meters'][meter_instance]['data_location'] = data_location
            else:
                raise NotImplementedError("NotImplementedError")        
        
        return metadata

    def save_metadata(self, key, metadata):
        """
        Overrites existing metadata at location specified by key
        
        Parameters
        ----------
        key : string
        metadata : dict
        """
        if key == '/':
            # Extract meter_devices
            meter_devices_metadata = metadata['meter_devices']
            dataset_metadata = dict(metadata)
            del dataset_metadata['meter_devices']
            # Write dataset metadata
            metadata_filename = join(self._get_metadata_path(), 'dataset.yaml')
            _write_yaml_to_file(metadata_filename, dataset_metadata)
            # Write meter_devices metadata
            metadata_filename = join(self._get_metadata_path(), 'meter_devices.yaml')
            _write_yaml_to_file(metadata_filename, meter_devices_metadata)
        else:
            # Write building metadata
            key_object = Key(key)
            assert key_object.building and not key_object.meter
            metadata_filename = join(self._get_metadata_path(), 'building{:d}.yaml'.format(key_object.building))
            _write_yaml_to_file(metadata_filename, metadata)

    def elements_below_key(self, key='/'):
        """
        Traverses file hierarchy rather than metadata
        
        Returns
        -------
        list of strings
        """
        
        elements = OrderedDict()
        if key == '/':
            for directory in listdir(self.filename):
                dir_path = join(self.filename, directory)
                if isdir(dir_path) and re.match('building[0-9]*', directory):
                    elements[directory] = join_key(key, directory)
        else:
            relative_path = key[1:]
            dir_path = join(self.filename, relative_path)
            if isdir(dir_path):
                for element in listdir(dir_path):
                    elements[element] = join_key(key, element)

        return elements

    def close(self):
        # not needed for CSV data store
        pass

    def open(self):
        # not needed for CSV data store
        pass
        
    def get_timeframe(self, key):
        """
        Returns
        -------
        nilmtk.TimeFrame of entire table after intersecting with self.window.
        """
        generator = self.load(key)
        start = None
        end = None
        for df in generator:
            if start is None:
                start = df.timeframe.start
            end = df.timeframe.end
        timeframe = TimeFrame(start, end)
        return self.window.intersect(timeframe)
        
    def _get_metadata_path(self):
        return join(self.filename, 'metadata')
        
    def _key_to_abs_path(self, key):
        abs_path = self.filename
        if key and len(key) > 1:
            relative_path = key
            if key[0] == '/':
                relative_path = relative_path[1:]
            abs_path = join(self.filename, relative_path)
            key_object = Key(key)
            if key_object.building and key_object.meter:
                abs_path += '.csv'
        return abs_path
        
def _write_yaml_to_file(metadata_filename, metadata):
    metadata_file = file(metadata_filename, 'w')
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
    
def get_datastore(filename, format, mode='a'):
    if filename is not None:
        if format == 'HDF':
            return HDFDataStore(filename, mode)
        elif format == 'CSV':
            return CSVDataStore(filename)
        else:
            raise ValueError('format not recognised')
    else:
        ValueError('filename is None')

class Key(object):
    """A location of data or metadata within NILMTK.
    
    Attributes
    ----------
    building : int
    meter : int
    utility : str
    """

    def __init__(self, string=None, building=None, meter=None):
        """
        Parameters
        ----------
        string : str, optional
            e.g. 'building1/elec/meter1'
        building : int, optional
        meter : int, optional
        """
        self.utility = None
        if string is None:
            self.building = building
            self.meter = meter
        else:
            split = string.strip('/').split('/')
            assert split[0].startswith('building'), "The first element must be 'building<I>', e.g. 'building1'; not '{}'.".format(split[0])
            try:
                self.building = int(split[0].replace("building", ""))
            except ValueError as e:
                raise ValueError("'building' must be followed by an integer.\n{}"
                                 .format(e))
            if len(split) > 1:
                self.utility = split[1]
            if len(split) == 3:
                assert split[2].startswith('meter')
                self.meter = int(split[-1].replace("meter", ""))
            else:
                self.meter = None
        self._check()

    def _check(self):
        assert isinstance(self.building, int)
        assert self.building >= 1
        if self.meter is not None:
            assert isinstance(self.meter, int)
            assert self.meter >= 1

    def __repr__(self):
        self._check()
        s = "/building{:d}".format(self.building)
        if self.meter is not None:
            s += "/elec/meter{:d}".format(self.meter)
        return s
