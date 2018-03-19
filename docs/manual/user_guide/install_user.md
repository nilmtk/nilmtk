
# Install NILMTK

We recommend using
[Anaconda](https://store.continuum.io/cshop/anaconda/), which bundles
togther most of the required packages. Since many packages are deprecating 
Python 2, we are now recommending installing Python 3.6. For the time being, 
Python 2.7 is still supported but that may change in the near future.

```bash
# 0. For Windows only. Download the appropriate version of VS C++ compiler for Python version.
# For Python 3.6 https://www.microsoft.com/en-us/download/details.aspx?id=48159
# For Python 2.7 https://www.microsoft.com/en-us/download/details.aspx?id=44266
# This is needed for building hmmlearn.

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

