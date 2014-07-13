
Processing pipeline, preprocessing and more stats
=================================================

At the core of NILMTK v0.2 is the concept of an 'out-of-core' processing
pipeline. What does that mean? 'out-of-core' refers to the ability to
handle datasets which are too large to fit into system memory. NILMTK
achieves this by setting up a processing pipeline which handle a chunk
of data at a time. We load small chunks from disk and pull these chunks
through a processing pipeline. Each pipeine is made up of ``Nodes``.
These can either be stats nodes or preprocessing nodes. Under the hood,
a pipeline is implemented as a chain of Python ``generators``. Stats
nodes live in ``nilmtk.stats`` and preprocessing nodes in
``nilmtk.preprocessing``. Most stats are wrapped by helper functions in
``ElecMeter`` and ``MeterGroup`` so you only have to dive in an play
directly with Nodes and the pipeline if you want to assemble your own
stats and preprocessing functions.

Having a pipeline which can handle small chunks not only allows us to
load arbitrarily large datasets. It also allows us to calculate stats on
arbitrary sized chunks of data (e.g. energy per day, or appliance usage
per week etc). To facilitate this, Stats nodes (e.g. for calculating
total energy or for finding missing samples) store their results in a
separate ``Results`` object. For example, the ``TotalEnergy`` stats node
stores its results in a ``TotalEnergyResults`` object:

.. code:: python

    from nilmtk import DataSet
    
    redd = DataSet('/data/REDD/redd.h5')
    elec = redd.buildings[1].elec
    fridge_meter = elec['fridge']
    
    total_fridge_energy = fridge_meter.total_energy(full_results=True)
    type(total_fridge_energy)



.. parsed-literal::

    nilmtk.stats.totalenergyresults.TotalEnergyResults



.. code:: python

    total_fridge_energy



.. parsed-literal::

                                  active                        end
    2011-04-18 09:22:13-04:00  44.750925  2011-05-24 15:56:34-04:00



Why store results in their own objects? Because these objects need to
know how to combine results from multiple chunks.

So, for example, let us get the total energy per day:

.. code:: python

    from nilmtk.timeframe import timeframes_from_periodindex
    import pandas as pd
    
    # First find the total time span for the fridge data:
    tf = fridge_meter.get_timeframe()
    tf



.. parsed-literal::

    TimeFrame(start='2011-04-18 09:22:13-04:00', end='2011-05-24 15:56:34-04:00', empty=False)



.. code:: python

    # Now make a PeriodIndex of daily periods:
    period_index = pd.period_range(start=tf.start, periods=5, freq='D')
    list(period_index) # just converting to a list for pretty printing



.. parsed-literal::

    [Period('2011-04-18', 'D'),
     Period('2011-04-19', 'D'),
     Period('2011-04-20', 'D'),
     Period('2011-04-21', 'D'),
     Period('2011-04-22', 'D')]



Now we can get the energy per day:

.. code:: python

    energy_per_day = fridge_meter.total_energy(sections=period_index, full_results=True)
    energy_per_day



.. parsed-literal::

                                 active                        end
    2011-04-18 09:22:13-04:00  0.678742  2011-04-18 19:59:59-04:00
    2011-04-18 20:00:03-04:00  1.153877  2011-04-19 18:45:09-04:00
    2011-04-19 20:20:05-04:00  1.244343  2011-04-20 19:59:59-04:00
    2011-04-20 20:00:03-04:00  1.003537  2011-04-21 19:59:56-04:00
    2011-04-21 20:00:00-04:00  1.219889  2011-04-22 19:59:58-04:00



And there we have it: the energy use per day. The days start at 8pm
because REDD is UTC-4:

.. code:: python

    redd.metadata['timezone']



.. parsed-literal::

    'US/Eastern'



And we can combine all the energy results from each day:

.. code:: python

    energy_per_day.combined()



.. parsed-literal::

    active    5.300387
    dtype: float64



To make the code as re-usable as possible, each stats module has a
``get_<stat>`` function which takes a vanilla DataFrame.

Load a restricted window of data
--------------------------------

.. code:: python

    from nilmtk import TimeFrame
    fridge_meter.store.window = TimeFrame("2011-04-20  20:00:00-04:00", "2011-04-25  20:00:00-04:00")
    fridge_meter.get_timeframe()
    # all subsequent processing will only consider the defined window



.. parsed-literal::

    TimeFrame(start='2011-04-20 20:00:00-04:00', end='2011-04-25 20:00:00-04:00', empty=False)



To reset the timeframe:

.. code:: python

    fridge_meter.store.window.clear()
    fridge_meter.get_timeframe()



.. parsed-literal::

    TimeFrame(start='2011-04-18 09:22:13-04:00', end='2011-05-24 15:56:34-04:00', empty=False)



The ``Apply`` preprocessing node
--------------------------------

We have an ``Apply`` node which applies an arbitrary Pandas function to
every chunk as it moves through the pipeline:

.. code:: python

    from nilmtk.preprocessing import Apply
    from nilmtk.stats import DropoutRate
.. code:: python

    fridge_meter.store.window = TimeFrame("2011-04-21  20:00:00-04:00", "2011-04-23  20:00:00-04:00")
    good_sections = fridge_meter.good_sections()
    good_sections



.. parsed-literal::

    [TimeFrame(start='2011-04-21 20:00:00-04:00', end='2011-04-22 22:46:53-04:00', empty=False),
     TimeFrame(start='2011-04-22 22:48:31-04:00', end='2011-04-23 19:59:59-04:00', empty=False)]



Fill gaps in appliance data:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # So, we're going to construct a new pipeline.
    # First we need to get a 'source node' from an ElecMeter or a MeterGroup:
    source_node = fridge_meter.get_source_node(sections=good_sections)
    
    # Then, just to see what's going on, we'll work out the dropout rate
    # before we've done any resampling.  We connect the source_node to the DropoutRate node:
    dropout_rate1 = DropoutRate(source_node)
    
    # The third node will be an Apply node.  We'll use Pandas' resample function:
    resample = Apply(func = lambda df: pd.DataFrame.resample(df, rule='3S', fill_method='ffill'), 
                     upstream=dropout_rate1)
    
    # Then we're calculate the dropout rate again.  This should be 0.0 because we've
    # resampled...
    dropout_rate2 = DropoutRate(resample)
    
    # At this point, no data has been loaded from disk yet.  We need to 'pull' data
    # through the pipeline by running 'run' on the last node in the pipeline:
    
    dropout_rate2.run()
.. code:: python

    # The dropout rate before resampling:
    dropout_rate1.results.combined()



.. parsed-literal::

    0.22210446987463711



.. code:: python

    # The dropout rate after resampling:
    dropout_rate2.results.combined()



.. parsed-literal::

    0.0


