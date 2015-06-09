
Load API
========

Loading API is central to a lot of nilmtk operations and provides a
great deal of flexibility. We'll now look at ways in which we can load
data from the nilmtk DataStore to the memory based on different
conditional queries. To see the full range of possible queries, we'll
use the `iAWE data set <http://iawe.github.io>`__ (whose HDF5 file can
be downloaded `here <https://copy.com/C2sIt1UfDx1mfPlC>`__) alongwith
the REDD data set.

The ``load`` function returns a generator of DataFrames loaded from the
DataStore based on the conditions specified. If no conditions are
specified, then all data from all the columns is loaded.

.. code:: python

    from matplotlib import rcParams
    import matplotlib.pyplot as plt
    %matplotlib inline
    rcParams['figure.figsize'] = (13, 6)
    plt.style.use('ggplot')
    from nilmtk import DataSet
    
    
    iawe = DataSet('/data/iawe/iawe.h5')
    elec = iawe.buildings[1].elec
    elec




.. parsed-literal::

    MeterGroup(meters=
      ElecMeter(instance=1, building=1, dataset='iAWE', site_meter, appliances=[])
      ElecMeter(instance=2, building=1, dataset='iAWE', site_meter, appliances=[])
      ElecMeter(instance=3, building=1, dataset='iAWE', appliances=[Appliance(type='fridge', instance=1)])
      ElecMeter(instance=4, building=1, dataset='iAWE', appliances=[Appliance(type='air conditioner', instance=1)])
      ElecMeter(instance=5, building=1, dataset='iAWE', appliances=[Appliance(type='air conditioner', instance=2)])
      ElecMeter(instance=6, building=1, dataset='iAWE', appliances=[Appliance(type='washing machine', instance=1)])
      ElecMeter(instance=7, building=1, dataset='iAWE', appliances=[Appliance(type='computer', instance=1)])
      ElecMeter(instance=8, building=1, dataset='iAWE', appliances=[Appliance(type='clothes iron', instance=1)])
      ElecMeter(instance=9, building=1, dataset='iAWE', appliances=[Appliance(type='unknown', instance=1)])
      ElecMeter(instance=10, building=1, dataset='iAWE', appliances=[Appliance(type='television', instance=1)])
      ElecMeter(instance=11, building=1, dataset='iAWE', appliances=[Appliance(type='wet appliance', instance=1)])
      ElecMeter(instance=12, building=1, dataset='iAWE', appliances=[Appliance(type='motor', instance=1)])
    )



Let us see what all measurements do we have for fridge.

.. code:: python

    fridge = elec['fridge']

.. code:: python

    fridge.available_columns()




.. parsed-literal::

    [('current', None),
     ('power', 'apparent'),
     ('frequency', None),
     ('voltage', None),
     ('power', 'active'),
     ('power factor', None),
     ('power', 'reactive')]



Loading data
------------

Loading all power columns (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    df = fridge.load().next()


.. parsed-literal::

    {'cols': [('current', None), ('power', 'apparent'), ('frequency', None), ('voltage', None), ('power', 'active'), ('power factor', None), ('power', 'reactive')]}


.. code:: python

    df.head()




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th>physical_quantity</th>
          <th colspan="3" halign="left">power</th>
        </tr>
        <tr>
          <th>type</th>
          <th>apparent</th>
          <th>active</th>
          <th>reactive</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2013-06-07 05:30:00+05:30</th>
          <td>2.486</td>
          <td>0.111</td>
          <td>2.483</td>
        </tr>
        <tr>
          <th>2013-06-07 05:30:01+05:30</th>
          <td>2.555</td>
          <td>0.200</td>
          <td>2.547</td>
        </tr>
        <tr>
          <th>2013-06-07 05:30:02+05:30</th>
          <td>2.485</td>
          <td>0.152</td>
          <td>2.480</td>
        </tr>
        <tr>
          <th>2013-06-07 05:30:03+05:30</th>
          <td>2.449</td>
          <td>0.159</td>
          <td>2.444</td>
        </tr>
        <tr>
          <th>2013-06-07 05:30:04+05:30</th>
          <td>2.519</td>
          <td>0.215</td>
          <td>2.510</td>
        </tr>
      </tbody>
    </table>
    </div>



Loading by specifying column names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    df = fridge.load(cols = [('power', 'active')]).next()
    df.head()


.. parsed-literal::

    {'cols': [('power', 'active')]}




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th>physical_quantity</th>
          <th>power</th>
        </tr>
        <tr>
          <th>type</th>
          <th>active</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2013-06-07 05:30:00+05:30</th>
          <td>0.111</td>
        </tr>
        <tr>
          <th>2013-06-07 05:30:01+05:30</th>
          <td>0.200</td>
        </tr>
        <tr>
          <th>2013-06-07 05:30:02+05:30</th>
          <td>0.152</td>
        </tr>
        <tr>
          <th>2013-06-07 05:30:03+05:30</th>
          <td>0.159</td>
        </tr>
        <tr>
          <th>2013-06-07 05:30:04+05:30</th>
          <td>0.215</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    df = fridge.load(cols = [('voltage', None)]).next()
    df.head()


.. parsed-literal::

    {'cols': [('voltage', None)]}




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
      </thead>
      <tbody>
        <tr>
          <th>2013-06-07 05:30:00+05:30</th>
        </tr>
        <tr>
          <th>2013-06-07 05:30:01+05:30</th>
        </tr>
        <tr>
          <th>2013-06-07 05:30:02+05:30</th>
        </tr>
        <tr>
          <th>2013-06-07 05:30:03+05:30</th>
        </tr>
        <tr>
          <th>2013-06-07 05:30:04+05:30</th>
        </tr>
      </tbody>
    </table>
    </div>



Loading by specifying physical\_type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    df = fridge.load(physical_quantity = 'power').next()
    df.head()


.. parsed-literal::

    {'cols': [('power', 'apparent'), ('power', 'active'), ('power', 'reactive')]}




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th>physical_quantity</th>
          <th colspan="3" halign="left">power</th>
        </tr>
        <tr>
          <th>type</th>
          <th>apparent</th>
          <th>active</th>
          <th>reactive</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2013-06-07 05:30:00+05:30</th>
          <td>2.486</td>
          <td>0.111</td>
          <td>2.483</td>
        </tr>
        <tr>
          <th>2013-06-07 05:30:01+05:30</th>
          <td>2.555</td>
          <td>0.200</td>
          <td>2.547</td>
        </tr>
        <tr>
          <th>2013-06-07 05:30:02+05:30</th>
          <td>2.485</td>
          <td>0.152</td>
          <td>2.480</td>
        </tr>
        <tr>
          <th>2013-06-07 05:30:03+05:30</th>
          <td>2.449</td>
          <td>0.159</td>
          <td>2.444</td>
        </tr>
        <tr>
          <th>2013-06-07 05:30:04+05:30</th>
          <td>2.519</td>
          <td>0.215</td>
          <td>2.510</td>
        </tr>
      </tbody>
    </table>
    </div>



Loading by specifying AC type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    df = fridge.load(ac_type = 'active').next()
    df.head()


.. parsed-literal::

    {'cols': [('power', 'active')]}




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th>physical_quantity</th>
          <th>power</th>
        </tr>
        <tr>
          <th>type</th>
          <th>active</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2013-06-07 05:30:00+05:30</th>
          <td>0.111</td>
        </tr>
        <tr>
          <th>2013-06-07 05:30:01+05:30</th>
          <td>0.200</td>
        </tr>
        <tr>
          <th>2013-06-07 05:30:02+05:30</th>
          <td>0.152</td>
        </tr>
        <tr>
          <th>2013-06-07 05:30:03+05:30</th>
          <td>0.159</td>
        </tr>
        <tr>
          <th>2013-06-07 05:30:04+05:30</th>
          <td>0.215</td>
        </tr>
      </tbody>
    </table>
    </div>



Loading by resmapling to a specified period
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    df = fridge.load(ac_type = 'active', sample_period=60).next()
    df.head()


.. parsed-literal::

    {'cols': [('power', 'active')]}




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th>physical_quantity</th>
          <th>power</th>
        </tr>
        <tr>
          <th>type</th>
          <th>active</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2013-06-07 05:30:00+05:30</th>
          <td>0.157583</td>
        </tr>
        <tr>
          <th>2013-06-07 05:31:00+05:30</th>
          <td>0.160567</td>
        </tr>
        <tr>
          <th>2013-06-07 05:32:00+05:30</th>
          <td>0.158170</td>
        </tr>
        <tr>
          <th>2013-06-07 05:33:00+05:30</th>
          <td>105.332802</td>
        </tr>
        <tr>
          <th>2013-06-07 05:34:00+05:30</th>
          <td>120.265068</td>
        </tr>
      </tbody>
    </table>
    </div>


