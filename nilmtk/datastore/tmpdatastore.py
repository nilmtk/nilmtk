import os
import tempfile
from contextlib import suppress

from nilmtk.datastore import HDFDataStore
from nilmtk.docinherit import doc_inherit


class TmpDataStore(HDFDataStore):
    def __init__(self):
        """Create an HDF datastore that is removed when it is closed.

        The descriptor returned by :func:`tempfile.mkstemp` is closed before
        PyTables opens the path. This avoids leaking one descriptor per store
        and permits the file to be reopened on Windows.
        """
        descriptor, tmp_path = tempfile.mkstemp(suffix=".h5", prefix="nilmtk-")
        self.full_path = tmp_path
        try:
            os.close(descriptor)
            super().__init__(filename=self.full_path, mode="a")
        except BaseException:
            store = getattr(self, "store", None)
            if store is not None:
                with suppress(Exception):
                    store.close()
            self._remove_file()
            raise

    @doc_inherit
    def close(self):
        try:
            super().close()
        finally:
            self._remove_file()

    def _remove_file(self):
        with suppress(FileNotFoundError):
            os.remove(self.full_path)
