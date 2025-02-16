import shutil
from pathlib import Path, PurePath

import pytest

from nilmtk.datastore import HDFDataStore

H5_PATH = Path("../../data/co_test.h5")


@pytest.fixture
def tmp_hdf(tmp_path: Path) -> Path:
    tmp_file = Path(tmp_path, "tmp.h5")
    shutil.copyfile(H5_PATH, tmp_file)
    return tmp_file


def test_hdf_datastore(tmp_hdf: Path):
    pass
