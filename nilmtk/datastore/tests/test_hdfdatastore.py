import shutil
from pathlib import Path

import pytest

from nilmtk.datastore import HDFDataStore

ROOT = Path(__file__).parents[3]
H5_PATH = Path(ROOT, "data/energy.h5")


@pytest.fixture
def tmp_hdf(tmp_path: Path) -> Path:
    tmp_file = Path(tmp_path, "tmp.h5")
    shutil.copyfile(H5_PATH, tmp_file)
    return tmp_file


def test_hdf_datastore(tmp_hdf: Path):
    KEY = "/building1/elec/meter1"
    store = HDFDataStore(tmp_hdf, "a")
    assert store[KEY].columns.names[0] == "physical_quantity"
    assert store[KEY].columns.names[1] == "type"
    store.close()
