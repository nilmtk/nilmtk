# Install NILMTK core

These instructions describe the current repository. NILMTK core supports
Python 3.11 and newer; use Python 3.11 when sharing an environment with
nilmtk-contrib or NILMbench.

## 1. Install uv

Follow the official [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/),
then confirm the command is available:

```bash
uv --version
```

## 2. Create an isolated environment

```bash
uv venv --python 3.11
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Install core

```bash
uv pip install "nilmtk @ git+https://github.com/nilmtk/nilmtk.git"
```

For the optional DEDDIAG converter:

```bash
uv pip install "nilmtk[deddiag] @ git+https://github.com/nilmtk/nilmtk.git"
```

## 4. Verify the installation

```bash
python -c "import nilmtk; print(nilmtk.__version__)"
nilmtk-convert --help
```

## 5. Choose the next layer

- Continue with core if you need converters, meters, preprocessing, or metrics.
- Install [nilmtk-contrib](https://github.com/nilmtk/nilmtk-contrib) if you need
  maintained disaggregation models.
- Use [NILMbench](https://github.com/nilmtk/nilmbench) if you need frozen
  real-data protocols and comparable leaderboard results.
- Read the [ecosystem guide](https://nilmtk.github.io/) for Docker ownership and
  citation guidance.

## Data is separate

Public datasets have their own licenses and download processes. NILMTK does not
bundle them. Obtain each dataset from its official custodian and convert it
locally with the matching module under
[`nilmtk/dataset_converters`](../../../nilmtk/dataset_converters).

## Avoid stale installation guides

Instructions based on Python 3.6, the historical NILMTK Anaconda channel, or
`setup.py develop` refer to earlier releases. Do not mix them with this uv-based
environment.
