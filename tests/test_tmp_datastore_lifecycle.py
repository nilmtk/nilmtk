import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import nilmtk.datastore.tmpdatastore as tmpdatastore_module
from nilmtk.datastore import HDFDataStore, TmpDataStore

ROOT = Path(__file__).resolve().parents[1]


def test_tmp_datastore_releases_creation_descriptor_and_removes_file(
    monkeypatch, tmp_path
):
    path = tmp_path / "cache.h5"
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
    closed_descriptors = []

    class RecordingOS:
        @staticmethod
        def close(value):
            closed_descriptors.append(value)
            os.close(value)

        @staticmethod
        def remove(value):
            os.remove(value)

    monkeypatch.setattr(
        tmpdatastore_module,
        "tempfile",
        SimpleNamespace(mkstemp=lambda **_kwargs: (descriptor, str(path))),
    )
    monkeypatch.setattr(tmpdatastore_module, "os", RecordingOS)

    store = TmpDataStore()
    try:
        assert closed_descriptors == [descriptor]
        assert path.exists()
    finally:
        store.close()
    store.close()

    assert not path.exists()


def test_tmp_datastore_removes_file_when_hdf_initialization_fails(
    monkeypatch, tmp_path
):
    path = tmp_path / "failed-cache.h5"
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
    monkeypatch.setattr(
        tmpdatastore_module,
        "tempfile",
        SimpleNamespace(mkstemp=lambda **_kwargs: (descriptor, str(path))),
    )
    partial_store = SimpleNamespace(closed=False)

    def close_partial_store():
        partial_store.closed = True

    partial_store.close = close_partial_store

    def fail_initialization(store, *_args, **_kwargs):
        with pytest.raises(OSError):
            os.fstat(descriptor)
        store.store = partial_store
        raise RuntimeError("simulated HDF initialization failure")

    monkeypatch.setattr(HDFDataStore, "__init__", fail_initialization)

    with pytest.raises(RuntimeError, match="simulated HDF initialization failure"):
        TmpDataStore()

    assert partial_store.closed
    assert not path.exists()


def test_process_exit_closes_and_removes_shared_stats_cache(tmp_path):
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "always",
            "-c",
            "import nilmtk; print(nilmtk.STATS_CACHE.full_path)",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "UnclosedFileWarning" not in result.stderr
    assert not Path(result.stdout.strip()).exists()
