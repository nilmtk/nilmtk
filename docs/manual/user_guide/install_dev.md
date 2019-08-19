
# Install NILMTK

We recommend using [Anaconda](https://www.anaconda.com/distribution/), which bundles togther most of the required packages. We recommend Anaconda for Python 3.6+ since Python 2.7 will soon be not supported.

If you prefer to avoid Anaconda, you could install most packages using `pip` but you will require a compatible C compiler for your Python distribution. Be sure to use the package versions listed in the file `environment-dev.yml`.

Before anything, install Anaconda. If you already have it installed, be sure to keep it updated.

The following instructions are general enough to work on Linux, macOS or Windows (run the commands on a Powershell session). Remember to adapt it to your environment.

1. Install Git, if not available. On Linux, install using your distribution package manager, e.g.:

```bash
sudo apt-get install git
```

On Windows, download and installation the official [Git](http://git-scm.com/download/win) client.

Alternatively, if you do not have administrator access, you can install Git directly on the Anaconda distribution running `conda install git`.

2. Download NILMTK:

```bash
cd ~
git clone https://github.com/nilmtk/nilmtk.git
cd nilmtk
```

The next step creates a separate environment for NILMTK (see [the Anaconda documentation]((https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))), using NILMTK's `environment-dev.yml` text file to define which packages need to be installed. If you have a previous `nilmtk-env` environment, please remove it first (`conda env remove -n nilmtk-env`).

```bash
conda env create -f environment-dev.yml
conda activate nilmtk-env
```

Next we will install [nilm_metadata](https://github.com/nilmtk/nilm_metadata) (for development, we recommend installing from the repository, even though there is a conda package available):

```bash
cd ~
git clone https://github.com/nilmtk/nilm_metadata/
cd nilm_metadata
python setup.py develop
cd ..
```

Change back to your nilmtk directory and install NILMTK:

```bash
cd ~/nilmtk
python setup.py develop
```

Run the unit tests:

```bash
nosetests
```

Then, work away on NILMTK!  When you are done, just do `conda deactivate` to deactivate the nilmtk-env (or just clone the terminal/session).
