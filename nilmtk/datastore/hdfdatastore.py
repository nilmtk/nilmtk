import warnings
from collections.abc import Iterator
from copy import deepcopy
from os.path import isfile

import numpy as np
import pandas as pd

from ..docinherit import doc_inherit
from ..timeframe import TimeFrame
from ..timeframegroup import TimeFrameGroup
from .datastore import MAX_MEM_ALLOWANCE_IN_BYTES, DataStore


class HDFDataStore(DataStore):
    @doc_inherit
    def __init__(self, filename, mode="r"):
        if mode in ["r", "a"] and not isfile(filename):
            raise IOError(f"No such file as {filename}")

        with warnings.catch_warnings():
            # Silence pytables warnings with numpy, out of our control
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=".*numpy.ufunc size changed.*",
            )

            self.store = pd.HDFStore(filename, mode, complevel=9, complib="blosc")
            super(HDFDataStore, self).__init__()

    @doc_inherit
    def __getitem__(self, key):
        return self.store[key]

    @doc_inherit
    def load(
        self,
        key,
        columns=None,
        sections=None,
        n_look_ahead_rows=0,
        chunksize=MAX_MEM_ALLOWANCE_IN_BYTES,
        verbose=False,
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame | None]]:
        # TODO: calculate chunksize default based on physical
        # memory installed and number of columns

        # Make sure key has a slash at the front but not at the end.
        if key[0] != "/":
            key = "/" + key
        if len(key) > 1 and key[-1] == "/":
            key = key[:-1]

        # Make sure chunksize is an int otherwise `range` complains later.
        chunksize = np.int64(chunksize)

        # Set `sections` variable
        sections = [TimeFrame()] if sections is None else sections
        sections = TimeFrameGroup(sections)

        # Replace any Nones with '' in columns:
        if columns is not None:
            columns = [("" if pq is None else pq, "" if ac is None else ac) for pq, ac in columns]

        if verbose:
            print(
                "HDFDataStore.load(key='{}', columns='{}', sections='{}',"
                " n_look_ahead_rows='{}', chunksize='{}')".format(key, columns, sections, n_look_ahead_rows, chunksize)
            )

        self.all_sections_smaller_than_chunksize = True

        for section in sections:
            if verbose:
                print("   ", section)
            window_intersect = self.window.intersection(section)

            if window_intersect.empty:
                data = pd.DataFrame()
                data.timeframe = section
                yield tuple(data, None)
                continue

            terms = window_intersect.query_terms("window_intersect")
            if terms is None:
                section_start_i = 0
                section_end_i = self.store.get_storer(key).nrows
                if section_end_i <= 1:
                    data = pd.DataFrame()
                    data.timeframe = section
                    yield tuple(data, None)
                    continue
            else:
                try:
                    coords = self.store.select_as_coordinates(key=key, where=terms)
                except AttributeError as e:
                    if str(e) == ("'NoneType' object has no attribute 'read_coordinates'"):
                        raise KeyError("key '{}' not found".format(key))
                    else:
                        raise
                n_coords = len(coords)
                if n_coords == 0:
                    data = pd.DataFrame()
                    data.timeframe = window_intersect
                    yield tuple(data, None)
                    continue

                section_start_i = coords[0]
                section_end_i = coords[-1]
                del coords

            slice_starts = range(section_start_i, section_end_i, chunksize)
            n_chunks = int(np.ceil((section_end_i - section_start_i) / chunksize))

            if n_chunks > 1:
                self.all_sections_smaller_than_chunksize = False

            for chunk_i, chunk_start_i in enumerate(slice_starts):
                chunk_end_i = chunk_start_i + chunksize
                there_are_more_subchunks = chunk_i < n_chunks - 1

                if chunk_end_i > section_end_i:
                    chunk_end_i = section_end_i
                chunk_end_i += 1

                data = self.store.select(key=key, columns=columns, start=chunk_start_i, stop=chunk_end_i)

                # if len(data) <= 2:
                #     yield pd.DataFrame()

                # Load look ahead if necessary
                if n_look_ahead_rows > 0:
                    if len(data.index) > 0:
                        look_ahead_start_i = chunk_end_i
                        look_ahead_end_i = look_ahead_start_i + n_look_ahead_rows
                        try:
                            look_ahead = self.store.select(
                                key=key,
                                columns=columns,
                                start=look_ahead_start_i,
                                stop=look_ahead_end_i,
                            )
                        except ValueError:
                            look_ahead = pd.DataFrame()
                    else:
                        look_ahead = pd.DataFrame()

                data.timeframe = _timeframe_for_chunk(there_are_more_subchunks, chunk_i, window_intersect, data.index)
                yield tuple(data, look_ahead)
                del data, look_ahead

    @doc_inherit
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

    @doc_inherit
    def put(self, key, value):
        self.store.put(key, value, format="table", index=False)
        self.store.create_table_index(key, columns=["index"], kind="full", optlevel=9)
        self.store.flush()

    @doc_inherit
    def remove(self, key):
        self.store.remove(key)

    @doc_inherit
    def load_metadata(self, key="/"):
        if key == "/":
            node = self.store.root
        else:
            node = self.store.get_node(key)

        metadata = deepcopy(node._v_attrs.metadata)
        return metadata

    @doc_inherit
    def save_metadata(self, key, metadata):
        if key == "/":
            node = self.store.root
        else:
            node = self.store.get_node(key)

        node._v_attrs.metadata = metadata
        self.store.flush()

    @doc_inherit
    def elements_below_key(self, key="/"):
        if key == "/" or not key:
            node = self.store.root
        else:
            node = self.store.get_node(key)
        return list(node._v_children.keys())

    @doc_inherit
    def close(self):
        self.store.close()

    @doc_inherit
    def open(self, mode="a"):
        self.store.open(mode=mode)

    @doc_inherit
    def get_timeframe(self, key):
        """
        Returns
        -------
        nilmtk.TimeFrame of entire table after intersecting with self.window.
        """
        data_start_date = self.store.select(key, [0]).index[0]
        data_end_date = self.store.select(key, start=-1).index[0]
        timeframe = TimeFrame(data_start_date, data_end_date)
        return self.window.intersection(timeframe)

    def _check_columns(self, key, columns):
        if columns is None:
            return
        if not self._table_has_column_names(key, columns):
            raise KeyError("at least one of " + str(columns) + " is not a valid column")

    def _table_has_column_names(self, key, columns):
        """
        Parameters
        ----------
        columns : string or list of strings

        Returns
        -------
        boolean
        """
        assert columns is not None
        self._check_key(key)
        if isinstance(columns, str):
            columns = [columns]
        query_cols = set(columns)
        table_cols = set(self._column_names(key) + ["index"])
        return query_cols.issubset(table_cols)

    def _column_names(self, key):
        self._check_key(key)
        storer = self._get_storer(key)
        col_names = storer.non_index_axes[0][1:][0]
        return col_names

    def _check_data_will_fit_in_memory(self, key, nrows, columns=None):
        # Check we won't use too much memory
        mem_requirement = self._estimate_memory_requirement(key, nrows, columns)
        if mem_requirement > MAX_MEM_ALLOWANCE_IN_BYTES:
            raise MemoryError("Requested data would use {:.3f}MBytes: too much memory.".format(mem_requirement / 1e6))

    def _estimate_memory_requirement(self, key, nrows, columns=None, paranoid=False):
        """Returns estimated mem requirement in bytes."""
        BYTES_PER_ELEMENT = 4
        BYTES_PER_TIMESTAMP = 8
        if paranoid:
            self._check_key(key)
        if columns is None:
            columns = self._column_names(key)
        elif paranoid:
            self._check_columns(key, columns)
        ncols = len(columns)
        est_mem_usage_for_data = nrows * ncols * BYTES_PER_ELEMENT
        est_mem_usage_for_index = nrows * BYTES_PER_TIMESTAMP
        if columns == ["index"]:
            return est_mem_usage_for_index
        else:
            return est_mem_usage_for_data + est_mem_usage_for_index

    def _nrows(self, key, timeframe=None):
        """
        Returns
        -------
        nrows : int
        """
        timeframe_intersect = self.window.intersection(timeframe)
        if timeframe_intersect.empty:
            nrows = 0
        elif timeframe_intersect:
            terms = timeframe_intersect.query_terms("timeframe_intersect")
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
            raise KeyError(key + " not in store")


def _timeframe_for_chunk(there_are_more_subchunks, chunk_i, window_intersect, index):
    start = None
    end = None

    # Test if there are any more subchunks
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
        start = index[0]
    if end is None:
        end = index[-1]

    return TimeFrame(start, end)
