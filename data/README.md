# Test data

These small datasets are fixtures for the NILMTK test suite. They contain
synthetic readings and metadata only; they are not benchmark datasets.

Regenerate the four HDF5 fixtures after a storage-format dependency changes:

```bash
uv run python -m nilmtk.tests.generate_data
```

The generator uses a fixed random seed. After regeneration, run:

```bash
uv run pytest -q tests nilmtk/tests
```

Tests must write converted or derived data under pytest temporary directories.
They must not modify the tracked fixtures or `data/random_csv`.
