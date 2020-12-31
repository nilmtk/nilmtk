import os
import tempfile

from nilmtk.datastore import HDFDataStore
from nilmtk.docinherit import doc_inherit


class TmpDataStore(HDFDataStore):
    def __init__(self):
        """ Create a `HDFDataStore` in the OS temporary directory in append mode.
            The created HDF file will remain on the disk until a call to the `close()` method.
        """
        _, tmp_path = tempfile.mkstemp(suffix=".h5", prefix="nilmtk-")
        self.full_path = tmp_path
        super().__init__(filename=self.full_path, mode="a")

    @doc_inherit
    def close(self):
        self.store.close()
        try:
            os.remove(self.full_path)
        except FileNotFoundError:
            pass

