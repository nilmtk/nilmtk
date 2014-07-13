
Disaggregation and Metrics
==========================

.. code:: python

    from nilmtk import DataSet, TimeFrame
    from nilmtk.disaggregate import CombinatorialOptimisation
    
    # Open data
    redd = DataSet('/data/REDD/redd.h5')
    
    # Select only the first half of the dataset for training
    redd.store.window = TimeFrame(start=None, end='2011-05-01 00:00:00-04:00')
    
    # Select house
    elec = redd.buildings[1].elec
    
    # Train!
    # (the co object does the appropriate preprocessing)
    # (here we are training on every appliance in the dataset; normally we 
    #  would probably want to filter out appliances which don't consume much energy,
    #  see previously in the user manual for how to do this.)
    co = CombinatorialOptimisation()
    co.train(elec)
    
    print("Model =", co.model)
To allow disaggregation to be done on any arbitrarily large dataset,
disaggregation output is dumped to disk chunk-by-chunk:

.. code:: python

    from nilmtk import HDFDataStore
    
    # Select second half of dataset for testing:
    redd.store.window = TimeFrame(start='2011-05-01 00:00:00-04:00', end=None)
    
    mains = elec.mains()
    output = HDFDataStore('output.h5', 'w')
    
    co.disaggregate(mains, output)
.. code:: python

    output.store.get('/building1/elec/meter9')[:10]



.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th></th>
          <th>power</th>
        </tr>
        <tr>
          <th></th>
          <th>apparent</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2011-05-01 00:00:00-04:00</th>
          <td> 87</td>
        </tr>
        <tr>
          <th>2011-05-01 00:01:00-04:00</th>
          <td>  0</td>
        </tr>
        <tr>
          <th>2011-05-01 00:02:00-04:00</th>
          <td> 87</td>
        </tr>
        <tr>
          <th>2011-05-01 00:03:00-04:00</th>
          <td>  0</td>
        </tr>
        <tr>
          <th>2011-05-01 00:04:00-04:00</th>
          <td>  0</td>
        </tr>
        <tr>
          <th>2011-05-01 00:05:00-04:00</th>
          <td> 87</td>
        </tr>
        <tr>
          <th>2011-05-01 00:06:00-04:00</th>
          <td> 87</td>
        </tr>
        <tr>
          <th>2011-05-01 00:07:00-04:00</th>
          <td>  0</td>
        </tr>
        <tr>
          <th>2011-05-01 00:08:00-04:00</th>
          <td> 87</td>
        </tr>
        <tr>
          <th>2011-05-01 00:09:00-04:00</th>
          <td> 87</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    output.close()
Metrics
-------

First we load the disag output exactly as if it were a normal dataset:

.. code:: python

    disag = DataSet('output.h5')
    disag_elec = disag.buildings[1].elec
    
    from nilmtk.metrics import f1_score
    
    # all metrics take the same arguments:
    f1_score(disag_elec, elec)

.. parsed-literal::

    /usr/local/lib/python2.7/dist-packages/sklearn/metrics/metrics.py:1249: UserWarning: The sum of true positives and false positives are equal to zero for some labels. Precision is ill defined for those labels [1]. The precision and recall are equal to zero for some labels. fbeta_score is ill defined for those labels [1]. 
      average=average)
    /usr/local/lib/python2.7/dist-packages/sklearn/metrics/metrics.py:1249: UserWarning: The sum of true positives and false positives are equal to zero for some labels. Precision is ill defined for those labels [0]. The precision and recall are equal to zero for some labels. fbeta_score is ill defined for those labels [0]. 
      average=average)
    /usr/local/lib/python2.7/dist-packages/sklearn/metrics/metrics.py:1249: UserWarning: The precision and recall are equal to zero for some labels. fbeta_score is ill defined for those labels [0]. 
      average=average)
    /usr/local/lib/python2.7/dist-packages/sklearn/metrics/metrics.py:1249: UserWarning: The sum of true positives and false negatives are equal to zero for some labels. Recall is ill defined for those labels [1]. The precision and recall are equal to zero for some labels. fbeta_score is ill defined for those labels [1]. 
      average=average)
    /usr/local/lib/python2.7/dist-packages/sklearn/metrics/metrics.py:1249: UserWarning: The precision and recall are equal to zero for some labels. fbeta_score is ill defined for those labels [1]. 
      average=average)




.. parsed-literal::

    5           0.662997
    6           0.107531
    7           0.624899
    8           0.712745
    9           0.610921
    11          0.168045
    12          0.057047
    13          0.002481
    14          0.000000
    15          0.137996
    16          0.011372
    17          0.310047
    18          0.232466
    19          0.000000
    (3, 4)      0.096890
    (10, 20)    0.162121
    dtype: float64



those are the F1 scores for each meter instance

