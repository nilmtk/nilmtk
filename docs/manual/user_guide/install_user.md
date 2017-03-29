
# Install NILMTK

We recommend using
[Anaconda](https://store.continuum.io/cshop/anaconda/), which bundles
togther most of the required packages. Please download Anaconda for
Python 2.7.x. Python 3 support is experimental and hence please only attempt to use
NILMTK on Python 3 if you are experienced with Python.

```bash
# 1. Download environment.yml from NILMTK. For Unix like environment, use wget
# for Windows, download manually
wget https://raw.githubusercontent.com/nilmtk/nilmtk/master/environment.yml

# 2. Creating nilmtk environment. Will download the necessary packages
conda env create
# For Linux/OSX
source activate nilmtk-env
# For Windows
activate nilmtk-env

# 3. Create a Jupyter kernel environment
python -m ipykernel install --user --name nilmtk-env --display-name "Python (nilmtk)"

#4. Configuring the interpreter in your IDE to use nilmtk-env
# See https://github.com/nilmtk/nilmtk/issues/557#issuecomment-290094260 on how to do this in PyCharm.
```

