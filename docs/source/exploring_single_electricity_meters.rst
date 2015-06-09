
Exploring single electricity meters
===================================

.. code:: python

    from nilmtk import DataSet
    
    redd = DataSet('/data/REDD/redd.h5')
    elec = redd.buildings[1].elec

