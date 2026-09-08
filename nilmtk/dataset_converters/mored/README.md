# MORED dataset converter

Install NILMTK using the repository's [installation guide](../../../README.md).
Download the original data following the dataset link below and keep the
published directory and CSV names.

```bash
nilmtk-convert mored /path/to/raw-data /path/to/mored.h5
```

```python
from nilmtk.dataset_converters import convert_mored
from nilmtk import DataSet

convert_mored("/path/to/raw-data", "mored.h5")
with DataSet("mored.h5") as dataset:
    print(dataset.buildings)
```

Download [MORED WPILGT](https://moredataset.github.io/MORED/) into
`Premises_1` through `Premises_10` folders; see [structure.txt](structure.txt).
Any subset of complete premises is supported. Events, WP-only and ILS data are
not converted. Existing output files require `--force` through the CLI. The
Python function follows the usual write-mode convention and replaces the file.

## Data interpretation

- CSV columns are selected by name: `timestamp`, `Vrms` (volts), and `real_power`
  (watts). Their order in the source file does not change their meaning.
- Measurement columns use NILMTK's two-level labels and `float32` values.
- Timestamps are sorted and unique. The first valid reading at a duplicate
  timestamp is retained. Rows without a timestamp or without any finite
  measurement are discarded; a missing voltage does not discard valid power.
- Both `day/month/year hour:minute:second` and the English-month format used by
  `Premises_10/TV.csv` are supported. Invalid nonempty timestamps raise an error.
- There is no resampling, interpolation, clipping, or synthesized aggregate.
- Premises 9 contains only meters 2-6: the authors did not provide mains data.
  `dataset.buildings[9].elec.mains()` returns `None`. Mains-based metrics and
  wiring queries requiring a measured upstream meter cannot be used there.
- Metadata describe only the premises present in the input. The freezer in
  premises 7 is read from `Premises_7/Freezer.csv`.

### Timezone assumption

The published timestamps have no UTC offset. The original converter interpreted
them as UTC; that behavior is preserved and output is represented in
`Africa/Casablanca`, matching the dataset metadata's IANA timezone. The authors'
metadata say `GMT+01:00` but do not establish whether the CSV wall times are UTC
or local. The absolute UTC interpretation is therefore **not independently
confirmed**. If the source export is known to use local time, select it explicitly:

```python
convert_mored("/path/to/data", "mored.h5", source_timezone="Africa/Casablanca")
```

Use the Python API to select `source_timezone`. Ambiguous or nonexistent local
times raise rather than silently shifting data.

### Output format

Only HDF5 is supported (`format="HDF"`). The current NILMTK CSV reader does not
reliably round-trip empty measurement subtypes and mixed-precision zoned
timestamps; CSV is rejected before opening the output.

## Validation and provenance

The tests generate small synthetic datasets for all 53 meters in all ten
premises, reopen the HDF5 result with `DataSet`, and check measurements, meter
paths, metadata, timestamps, subsets, and error cleanup. No raw measurements are
redistributed in the tests. Dataset YAML is adapted from the authors' metadata;
their other appliance descriptions (including washer/dryer classifications)
are retained rather than inferred from filenames alone.

A full WPILGT download was validated on 2026-09-08 using an installed NILMTK
integration wheel: all 44,172,557 readings across 53 meters matched an independent
CSV-to-HDF5 comparison exactly. Loading/resampling, energy and good-section
statistics were exercised for every meter. A two-appliance CO train/predict run
on household 1 also passed; this is an integration check, not an accuracy study.
The timestamp comparison uses the documented UTC source-time assumption.

The complete official `Premises_9/TV.csv` and `Premises_9/Refrigerator.csv` files
were byte-identical (6,707,937 bytes each). Both published channels are preserved;
the converter cannot infer which source label or recording should be corrected.

Dataset reference: Ahajjam et al. (2020),
[MORED: A Moroccan Buildings' Electricity Consumption Dataset](https://doi.org/10.3390/en13246737).
Converter author: Camilo Mariño. Downloaded data remain subject to the dataset
publisher's terms.
