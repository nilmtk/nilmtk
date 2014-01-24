.. currentmodule:: nilmtk
.. _preprocessing:

.. ipython:: python
   :suppress:
   
   %precision 1
   import numpy as np
   import matplotlib.pyplot as plt
      
**********************
Dataset Preprocessing
**********************

As a demonstration, let us first load the iAWE dataset (which has
already been converted to HDF5 format):

.. ipython:: python

   from nilmtk.dataset import DataSet
   from nilmtk.sensors.electricity import Measurement
   dataset = DataSet()
   dataset.load_hdf5('/home/nipun/Dropbox/nilmtk_datasets/iawe/')
   building = dataset.buildings[1]
   electric = building.utility.electric
   electric.appliances.keys()

Finding the range of voltage for air conditioner 1
---------------------------------------------------

.. ipython:: python

   electric.appliances[('air conditioner',1)][('voltage','')].describe()

We observe minimum voltage of 0 and maximum of 5140. Clearly, these are due to some fault in data collection. These readings should be removed  

Removing readings in the dataset where `voltage` >260 or `voltage` <160
------------------------------------------------------------------------

.. ipython:: python

   import nilmtk.preprocessing.electricity.building as prepb
   building = prepb.filter_out_implausible_values(
              building, Measurement('voltage', ''), 160, 260)

Now, observing the voltage variation in the same air conditioner as before.

.. ipython:: python

   building.utility.electric.appliances[('air conditioner',1)][('voltage','')].describe()

Filtering data from 13 July to 4 August
----------------------------------------

.. ipython:: python

   building = prepb.filter_datetime(
    building, '7-13-2013', '8-4-2013')

Downsampling the dataset to 1 minute
-------------------------------------

.. ipython:: python

   building =prepb.downsample(building, rule='1T')

Fill large gaps in appliances with zeros and forward-fill small gaps
----------------------------------------------------------------------

.. ipython:: python

   building = prepb.fill_appliance_gaps(building)

Prepend and append zeros
-------------------------

.. ipython:: python

   building = prepb.prepend_append_zeros(
    building, '7-13-2013', '8-4-2013', '1T', 'Asia/Kolkata')
   building.utility.electric.appliances[('air conditioner',1)][('voltage','')].plot()

Drop missing samples from mains
--------------------------------

.. ipython:: python

   building = prepb.drop_missing_mains(building)

Find intersection of mains and appliance datetime indicies
-----------------------------------------------------------

.. ipython:: python

   building = prepb.make_common_index(building)

