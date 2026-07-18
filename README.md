# NILMTK core

NILMTK is the data and evaluation layer of the open-source ecosystem for
non-intrusive load monitoring (NILM). It converts and loads energy datasets,
represents buildings and meters, prepares time windows, computes statistics and
metrics, and provides classical reference algorithms.

**Use this repository when your work is about data, meters, preprocessing, or
metrics.** For maintained neural models or reproducible benchmark claims, use
the companion repositories below.

## Choose the right repository

| Job | Canonical repository |
| --- | --- |
| Convert, load, inspect, and score energy data | **[NILMTK core](https://github.com/nilmtk/nilmtk)** — this repository |
| Resolve appliance names, synonyms, meters, and dataset semantics | [NILM Metadata](https://github.com/nilmtk/nilm_metadata) |
| Use or contribute a disaggregation model | [nilmtk-contrib](https://github.com/nilmtk/nilmtk-contrib) |
| Reproduce T1/T2/T3 protocols or publish a leaderboard result | [NILMbench](https://github.com/nilmtk/nilmbench) |

The [NILMTK ecosystem guide](https://nilmtk.github.io/) explains how these
layers fit together, which Docker route to use, and which papers to cite.

## Supported installation

NILMTK core supports Python 3.11 and newer. Use Python 3.11 for an end-to-end
environment shared with nilmtk-contrib and NILMbench.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create
an isolated environment:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install "nilmtk @ git+https://github.com/nilmtk/nilmtk.git"
python -c "import nilmtk; print(nilmtk.__version__)"
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

To use the DEDDIAG converter, install the optional extra:

```bash
uv pip install "nilmtk[deddiag] @ git+https://github.com/nilmtk/nilmtk.git"
```

Do not combine these instructions with old Python 3.6, Anaconda-channel, or
`setup.py develop` tutorials. Those routes describe earlier releases and are
not the supported installation for the current repository.

### Verify the command-line converter

```bash
nilmtk-convert --help
nilmtk-convert list
# Example after downloading REDD:
nilmtk-convert redd /path/to/low_freq /path/to/redd.h5
```

Dataset-specific converter arguments and source links live under
[`nilmtk/dataset_converters`](nilmtk/dataset_converters).

## Docker ownership

NILMTK core is a Python library and does not publish a separate official core
image. This is intentional:

- use the single [nilmtk-contrib Dockerfile](https://github.com/nilmtk/nilmtk-contrib)
  for a general environment containing core, metadata, and model code;
- use [NILMbench](https://github.com/nilmtk/nilmbench) for pinned CPU-smoke and
  CUDA-benchmark runtimes that certify leaderboard results;
- do not create an image for each algorithm.

Keeping container ownership in those two places prevents four repositories from
shipping drifting copies of the same environment.

## Data

NILMTK does not redistribute REDD, UK-DALE, REFIT, or other licensed datasets.
Download data from its official custodian, comply with its license, and convert
it locally. A converted HDF5 dataset can then be opened with:

```python
from nilmtk import DataSet

dataset = DataSet("redd.h5")
print(dataset.metadata)
print(dataset.buildings)
```

NILM Metadata is installed with core and supplies the canonical appliance
taxonomy, synonyms, and meter relationships used while loading datasets.

## What core provides

- converters for public energy datasets;
- lazy access to buildings, meters, appliances, and time frames;
- resampling, alignment, preprocessing, and data-quality statistics;
- standard NILM accuracy and energy metrics;
- the rapid experimentation API used by nilmtk-contrib;
- classical reference disaggregators and baseline utilities.

Detailed API reference is published at
[nilmtk.github.io/nilmtk/master](https://nilmtk.github.io/nilmtk/master/index.html).
The repository manual and notebooks live under [`docs/manual`](docs/manual).

## Development

```bash
git clone https://github.com/nilmtk/nilmtk.git
cd nilmtk
uv sync --extra dev
uv run pytest tests
```

Before opening a pull request, run the narrow test for your change, the current
package gate, and the documentation contract:

```bash
uv run pytest tests
uv run python scripts/check_docs.py
uv build
```

The historical core regression tests live under `nilmtk/tests` and
`nilmtk/stats/tests`. Run the affected files explicitly when changing those
modules; work to bring those fixtures into the default gate is tracked
separately.

Changes to dataset semantics belong in NILM Metadata. New model architectures
belong in nilmtk-contrib. Benchmark task definitions and published result
bundles belong in NILMbench.

## Citation

If you use core dataset conversion, meter abstractions, preprocessing, or
metrics, cite the NILMTK paper:

```bibtex
@inproceedings{batra2014nilmtk,
  title     = {NILMTK: An Open Source Toolkit for Non-intrusive Load Monitoring},
  author    = {Batra, Nipun and Kelly, Jack and Parson, Oliver and Dutta, Haimonti
               and Knottenbelt, William and Rogers, Alex and Singh, Amarjeet
               and Srivastava, Mani},
  booktitle = {Proceedings of the 5th ACM International Conference on Future
               Energy Systems},
  year      = {2014},
  pages     = {265--276},
  doi       = {10.1145/2602044.2602051}
}
```

Also cite the [NILM Metadata paper](https://doi.org/10.1109/COMPSACW.2014.97)
when relying on its schema or taxonomy, the
[nilmtk-contrib paper](https://doi.org/10.1145/3360322.3360844) when using its
model suite, and the [NILMBench2026 paper](https://doi.org/10.1145/3744256.3812587)
when using its protocols, runner, or leaderboard results. Always cite the
original model and dataset papers as well.

## Help and license

Search [existing issues](https://github.com/nilmtk/nilmtk/issues) before opening
a report. Include the exact command, operating system, Python version, dataset
identity, and a minimal reproducer.

NILMTK is released under the [Apache License 2.0](LICENSE).
