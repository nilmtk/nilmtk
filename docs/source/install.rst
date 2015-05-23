
Installing nilmtk
=================

Installing on Ubuntu like Linux variants (debian based) or OSX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NB: The following procedure is for Ubuntu like Linux variants (debian
based). Please adapt accordingly for your OS. We would welcome
installation instructions for other OS as well. We would recommend using
`Anaconda <https://store.continuum.io/cshop/anaconda/>`__, which bundles
togther most of the required packages. Please download Anaconda for
Python 2.7.x. We do not currently support Python 3. After installing
Anaconda, please do the following:

-  Updating anaconda

   .. code:: bash

       conda update --yes conda

-  Installing HDF5 libaries and python-dev

   .. code:: bash

       sudo apt-get install libhdf5-serial-dev python-dev

-  Installing git client

   .. code:: bash

       sudo apt-get install git

-  Installing pip and other dependencies which might be missing from
   Anaconda

   .. code:: bash

       conda install --yes pip numpy scipy six scikit-learn pandas numexpr pytables dateutil matplotlib networkx

-  Installing `NILM
   Metadata <https://github.com/nilmtk/nilm_metadata>`__.

   .. code:: bash

       git clone https://github.com/nilmtk/nilm_metadata/
       cd nilm_metadata
       python setup.py develop
       cd ..

-  Installing `hmmlearn <https://github.com/hmmlearn/hmmlearn>`__

   .. code:: bash

       git clone git://github.com/hmmlearn/hmmlearn.git
       cd hmmlearn
       python setup.py install
       cd ..

-  Misc. pip installs

   .. code:: bash

       pip install psycopg2 nose coveralls coverage

-  Finally! Install nilmtk

   .. code:: bash

       git clone https://github.com/nilmtk/nilmtk.git
       cd nilmtk
       python setup.py develop
       cd..

-  If you wish to contribute to nilmtk, you may also run tests

   .. code:: bash

       cd nilmtk
       nosetests

Installing on Windows
~~~~~~~~~~~~~~~~~~~~~

-  Install `Anaconda <https://store.continuum.io/cshop/anaconda/>`__,
   which bundles togther most of the required packages. Please download
   Anaconda for Python 2.7.x. We do not currently support Python 3.

-  Install `git <http://git-scm.com/download/win>`__ client. You may
   need to add ``git.exe`` to your path in order to run nilmtk tests.

-  Installing pip and other dependencies which might be missing from
   Anaconda

   .. code:: bash

       $ conda install --yes pip numpy scipy six scikit-learn pandas numexpr pytables dateutil matplotlib networkx

-  Installing `NILM
   Metadata <https://github.com/nilmtk/nilm_metadata>`__ from git bash

   .. code:: bash

       $ git clone https://github.com/nilmtk/nilm_metadata/
       $ cd nilm_metadata
       $ python setup.py develop
       $ cd ..

-  Installing `hmmlearn <https://github.com/hmmlearn/hmmlearn>`__

   .. code:: bash

       git clone git://github.com/hmmlearn/hmmlearn.git
       cd hmmlearn
       python setup.py install
       cd ..

-  Installing postgresql support (currently needed for WikiEnergy
   converter) Download release for your python environment:
   http://www.stickpeople.com/projects/python/win-psycopg/

-  Misc. pip installs

   .. code:: bash

       $ pip install nose pbs coveralls coverage

-  Finally! Install nilmtk from git bash

   .. code:: bash

       $ git clone https://github.com/nilmtk/nilmtk.git
       $ cd nilmtk
       $ python setup.py develop
       $ cd..

-  If you wish to contribute to nilmtk, you may also run tests

   .. code:: bash

       $ cd nilmtk
       $ nosetests
