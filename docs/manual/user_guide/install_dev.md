
# Install NILMTK

NILMTK now supports Python 3.11+.

The following instructions are general enough to work on Linux, macOS or Windows (run the commands on a Powershell session). Remember to adapt it to your environment.

1. Install Git, if not available. On Linux, install using your distribution package manager, e.g.:

```bash
sudo apt install git
```

On Windows, download and installation the official [Git](http://git-scm.com/download/win) client.

2. Download NILMTK:

```bash
cd ~
git clone https://github.com/nilmtk/nilmtk.git
cd nilmtk
```

The next step creates a separate environment for NILMTK (see [the virtualenv](https://virtualenv.pypa.io/en/latest/user_guide.html)).

- Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

- Windows (Powershell)

```bash
python -m venv .venv
& .venv/Scripts/Activate.ps1
```

Next we will install [nilm_metadata](https://github.com/nilmtk/nilm_metadata) (for development, we recommend installing from the repository):

```bash
git clone https://github.com/nilmtk/nilm_metadata/ ~/nilm_metadata
python -m pip install ~/nilm_metadata
```

Install NILMTK:

```bash
python -m pip install -e .[dev]
```

Run the unit tests:

```bash
pytest
```

Then, work away on NILMTK!  When you are done, just do `deactivate` to deactivate the virtual environment (or just clone the terminal/session).
