
# Install NILMTK

We recommend using
[Anaconda](https://store.continuum.io/cshop/anaconda/), which bundles
togther most of the required packages. Please download Anaconda for
Python 2.7.x. We do not currently support Python 3.

After installing Anaconda, please do the following.  We have two
sections below:
[one for Linux or OSX](#install-on-ubuntu-like-linux-variants-debian-based-or-osx)
and another [section for Windows](#install-on-windows).

### Install on Ubuntu like Linux variants (debian based) or OSX

NB: The following procedure is for Ubuntu like Linux variants (Debian
based). Please adapt accordingly for your OS. We would welcome
installation instructions for other OSes as well.

#### Experimental but probably easiest installation procedure for Unix or OSX

In this section we will describe a fast and simple, but fairly
untested installation procedure.  Please give this a go and tell us in
[the issue queue](https://github.com/nilmtk/nilmtk/issues) if this
process doesn't work for you.  The old Unix and OSX instructions are
further down this page if you need to try them.

Install git, if necessary:

```bash
sudo apt-get install git
```

Download NILMTK:

```bash
cd ~
git clone https://github.com/nilmtk/nilmtk.git
cd nilmtk
```

The next step uses [conda-env](https://github.com/conda/conda-env) to
install an environment for NILMTK, using NILMTK's `environment.yml`
text file to define which packages need to be installed:

```bash
conda env create
source activate nilmtk-env
```

Next we will install
[nilm_metadata](https://github.com/nilmtk/nilm_metadata) (can't yet
install using pip / conda):

```bash
cd ~
git clone https://github.com/nilmtk/nilm_metadata/
cd nilm_metadata; python setup.py develop; cd ..
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

Then, work away on NILMTK :).  When you are done, just do `source
deactivate` to deactivate the nilmtk-env.


#### Old installation procedure for Unix or OSX

- Update anaconda
```bash
conda update --yes conda
```

- Install HDF5 libaries and python-dev
```bash
sudo apt-get install libhdf5-serial-dev python-dev
```

- Install git client
```bash
sudo apt-get install git
```

- Install pip and other dependencies which might be missing from Anaconda
```bash
conda install --yes pip numpy scipy six scikit-learn pandas numexpr pytables dateutil matplotlib networkx hmmlearn
```
Note that, if you are using `pip` instead of `conda` then remove
`dateutil` and replace `pytables` with `tables`.

Please also note that there is
[a bug in Pandas 0.17](https://github.com/pydata/pandas/issues/11626)
which causes serious issues with data where the datetime index crosses
a daylight saving boundary.  As such, please do not install Pandas
0.17 for use with NILMTK.  Pandas 0.17.1 was released on the 20th Nov
2015 and includes a fix for this bug.  Please make sure you install
Pandas 0.17.1 or higher.

- Install [NILM Metadata](https://github.com/nilmtk/nilm_metadata).
```bash
git clone https://github.com/nilmtk/nilm_metadata/
cd nilm_metadata
python setup.py develop
cd ..
```

- Install psycopg2
First you need to install Postgres:
```bash
sudo apt-get install postgresql postgresql-contrib
sudo apt-get install postgresql-server-dev-all
pip install psycopg2
```

- Misc. pip installs
```bash
pip install nose coveralls coverage
```

- Finally! Install NILMTK
```bash
git clone https://github.com/nilmtk/nilmtk.git
cd nilmtk
python setup.py develop
cd..
```

- If you wish, you may also run NILMTK tests to make sure the installation has succeeded.
```bash
cd nilmtk
nosetests
```

### Install on Windows

- Install [Anaconda](https://store.continuum.io/cshop/anaconda/), which bundles togther most of the required packages. Please download Anaconda for Python 2.7.x. We do not currently support Python 3.

- Install [git](http://git-scm.com/download/win) client. You may need to add `git.exe` to your path in order to run nilmtk tests. 

- Now, we recommend trying
  [the experimental but simple installation instructions above](#experimental-but-probably-easiest-installation-procedure-for-unix-or-osx).
  If they don't work then let us know on [the issue queue](https://github.com/nilmtk/nilmtk/issues) and then try
  the old windows installation procedure below:

#### Old Windows installation procedure

- Install pip and other dependencies which might be missing from Anaconda
```bash
conda install --yes pip numpy scipy six scikit-learn pandas numexpr pytables dateutil matplotlib networkx hmmlearn
```

- Install [NILM Metadata](https://github.com/nilmtk/nilm_metadata) from git bash
```bash
git clone https://github.com/nilmtk/nilm_metadata/
cd nilm_metadata
python setup.py develop
cd ..
```

-  Install postgresql support (currently needed for WikiEnergy converter)
Download release for your python environment:
http://www.stickpeople.com/projects/python/win-psycopg/

- Misc. pip installs
```bash
pip install nose pbs coveralls coverage
```

- Finally! Install nilmtk from git bash
```bash
git clone https://github.com/nilmtk/nilmtk.git
cd nilmtk
python setup.py develop
cd..
```

- If you wish, you may also run NILMTK tests to make sure the installation has succeeded
```bash
cd nilmtk
nosetests
```

# Installation troubleshooting

### C headers installed in a non-standard location

If C headers for a required Python library are installed in a non-standard location (for example, because you don't have root access) then installation might fail with an error like this:

```bash
fatal error: numpy/arrayobject.h: No such file or directory compilation terminated.
error: command 'gcc' failed with exit status 1
```

In this case, first find where `numpy/arrayobject.h` is stored on your system (for example, by using `locate numpy/arrayobject.h`).  (For reference: the standard location is `/usr/include/python2.7/` which  contains lots of `.h` header files and a `numpy` directory which is also full of `.h` files.)  

Now try appending your `C_INCLUDE_PATH` environment variable with the path in which you find `numpy/arrayobject.h` and try building again.

(For reference: [here's the original issue](https://github.com/nilmtk/nilmtk/issues/44) where this problem arose and was solved.)


    
