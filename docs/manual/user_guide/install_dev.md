# Develop NILMTK core

Use Python 3.11 and uv so local development matches the ecosystem's supported
research environment.

## Clone and install

```bash
git clone https://github.com/nilmtk/nilmtk.git
cd nilmtk
uv sync --extra dev
```

The project metadata installs the reviewed NILM Metadata revision required by
core. You do not need to run `setup.py` or install a separate Conda environment.

## Run checks

Start with the narrowest test that covers your change, then run the current
package gate:

```bash
uv run pytest path/to/test_file.py
uv run pytest tests
uv run python scripts/check_docs.py
uv build
```

The historical regression tests live under `nilmtk/tests` and
`nilmtk/stats/tests`. Run the affected files explicitly when changing core or
statistics behavior; their legacy HDF fixtures are being modernized before the
directories join default discovery.

## Work on core and metadata together

Clone the repositories as siblings and install metadata editably only when your
change genuinely crosses the schema boundary:

```bash
cd ..
git clone https://github.com/nilmtk/nilm_metadata.git
cd nilmtk
uv pip install -e ../nilm_metadata
```

Changes to appliance taxonomy, synonyms, schema, or meter semantics belong in
[NILM Metadata](https://github.com/nilmtk/nilm_metadata). New architectures
belong in [nilmtk-contrib](https://github.com/nilmtk/nilmtk-contrib), and frozen
benchmark protocols or result bundles belong in
[NILMbench](https://github.com/nilmtk/nilmbench).

## Pull request evidence

Include:

- the failure or missing behavior your change addresses;
- focused regression tests, including malformed or boundary inputs;
- every test command and outcome, including known failures;
- dataset identity and a minimal reproducible window when data behavior changes;
- documentation updates when a public contract changes.

Do not commit licensed datasets, credentials, generated environments, or local
HDF5 outputs.
