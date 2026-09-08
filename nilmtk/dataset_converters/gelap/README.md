# GeLaP dataset converter

Install NILMTK using the repository's [installation guide](../../../README.md).
Download the original data following the dataset link below and keep the
published directory and CSV names.

```bash
nilmtk-convert gelap /path/to/raw-data /path/to/gelap.h5
```

```python
from nilmtk.dataset_converters import convert_gelap
from nilmtk import DataSet

convert_gelap("/path/to/raw-data", "gelap.h5")
with DataSet("gelap.h5") as dataset:
    print(dataset.buildings)
```

Download [GeLaP](https://mygit.th-deg.de/tcg/gelap). Supply a folder containing
`hh-01.tar.xz` through `hh-20.tar.xz`, or extracted `hh-01` through `hh-20`
folders. Any subset of complete households is supported. Archives are read
sequentially without writing extracted files; extracted folders take precedence.
To require extracted folders, use `convert_gelap(path, output, False)`.
Existing output files require `--force` through the CLI; the Python function
replaces the file. Memory usage is proportional to one meter CSV. On failure
the datastore is closed, but a partial output may remain and must not be used.

## Data interpretation

- Appliance meters 1-10 retain their original label numbers. Their index uses
  `time_reply`, not `time_request`, in Unix milliseconds.
- Meter 11 is the sum of `power1`, `power2`, and `power3` from `smartmeter.csv`.
  A sample must contain all three finite phases. Partial or missing phases never
  become zero demand. Negative power is preserved. Individual phase readings
  are not retained as separate meters.
- Measurements are active power in watts, stored as `float32`, with NILMTK's
  two-level measurement labels.
- UTC timestamps are converted to `Europe/Berlin`, including DST transitions.
  They are sorted and deduplicated, retaining the first valid source reading.
  No resampling, interpolation, or clipping is performed.
- Appliance metadata explicitly connect to meter 11. Only converted households
  are included in the output metadata.
- Appliance mappings were checked against the authors' `info.json`: household
  16 label 1 is a garage door opener; household 20 labels 1 and 2 are coffee
  machine and kettle, respectively. Dryers use `tumble dryer`, and household 15's
  flat iron uses `clothes iron`. Bread slicers and juicers use the generic
  `small cooking appliance` category rather than an incorrect resistive category.
  Labels without a justified canonical equivalent retain `unknown`.

Only HDF5 is supported (`format="HDF"`). NILMTK's current CSV reader does not
reliably round-trip mixed-precision timezone-aware timestamps, so CSV output is
rejected before opening the destination.

## Validation, citation and rights

Tests generate synthetic readings for all 220 meters, reopen the result with
`DataSet`, and check meter topology, timestamps, phase sums, metadata, archives,
subsets and failure cleanup. No raw measurements are redistributed in tests.

The complete twenty-house release was validated on 2026-09-08 with an installed
NILMTK integration wheel. All 2,660,429,595 source rows were read; the expected
output contains 2,660,429,589 rows after excluding one invalid row and five
duplicate timestamps. Every output timestamp and float32 value matched an
independent full-source reference using canonical SHA256 comparisons.

Each complete house was converted through the CLI in four parallel processes.
The resulting HDF5 groups were combined using native HDF5 object copies,
preserving all 220 full indexes. Loading/resampling, energy and good-section
statistics were exercised for every meter; upstream-meter links and a
two-appliance CO train/predict smoke test on household 1 also passed. This tests
integration, not the scientific accuracy of the disaggregation algorithm.
No original measurements are redistributed with this contribution.

Dataset authors: Sebastian Wilhelm, Dietmar Jakob, Jakob Kasbauer and Diane
Ahrens. Cite [GeLaP: German Labeled Dataset for Power Consumption (2021)](https://doi.org/10.1007/978-981-16-2377-6_5).
Converter author: Camilo Mariño.

The authors publish GeLaP under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
That notice is retained in the dataset metadata; conversion does not change the
dataset's terms. Metadata mappings are adapted from the authors' household
labels.
