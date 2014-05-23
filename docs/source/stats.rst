.. currentmodule:: nilmtk
.. _stats:

.. ipython:: python
   :suppress:
   
   %precision 1
   import numpy as np
   import matplotlib.pyplot as plt
   import calendar
   import seaborn
   
**********************************
Statistics and dataset diagnostics
**********************************

As a demonstration, let us first load the REDD dataset (which has
already been converted to HDF5 format):

.. ipython:: python

   from nilmtk.dataset import DataSet
   dataset = DataSet()
   dataset.load_hdf5('/home/nipun/Dropbox/nilmtk_datasets/redd/low_freq/')
   electric = dataset.buildings[1].utility.electric
   electric.appliances.keys()

Basic stats for a whole building
--------------------------------

`proportion_of_energy_submetered` reports the proportion of energy in
a building that is submetered where 0 = no energy submetered and 1 =
all energy submetered:

.. ipython:: python

   import nilmtk.stats.electricity.building as building_stats
   building_stats.proportion_of_energy_submetered(electric)

   # And get the top k appliances:
   # building_stats.top_k_appliances(electric, k=5)

Diagnosing problems with data
-----------------------------

There are two reasons why data might not be recorded:

1. The appliance and appliance monitor were unplugged from the mains
   (hence the appliance is off when the appliance monitor is off).
2. The appliance monitor is misbehaving (hence we have no reliable
   information about the state of the appliance).

nilmtk has a number of functions to help find periods where samples
for one or more sensors were not recorded.

By default, `plot_missing_samples_using_rectangles` plots rectangles
indicating the presence of a gap in the data, where a 'gap' is defined
by the `max_sample_period` argument.  If two consecutive samples are
more than `max_sample_period` apart then that's a gap!  The default is
`4 x sample_period`. The plot below shows that the two mains channels
are inactive for most of the second half of May 2011:

.. ipython:: python
   
   @savefig plot_missing_samples_using_rectangles.png
   building_stats.plot_missing_samples_using_rectangles(electric)

.. ipython:: python
   :suppress:

   plt.close('all')

The advantages of `plot_missing_samples_using_rectangles` are:

* clearly shows large gaps
* shows all data so can be zoomed in to your heart's content

The disadvantages are:

* The choice of `max_sample_period` is somewhat subjective
* Because it plots lots of rectangles, it can be slow to plot.

To overcome both of these disadvantages, we have a sister function:

.. ipython:: python

   @savefig plot_missing_samples_using_bitmap.png
   building_stats.plot_missing_samples_using_bitmap(electric)

.. ipython:: python
   :suppress:

   plt.close('all')

Here, the darkness of the blue colour indicates the proportion of
samples lost, where dark blue means all samples are lost, light blue
means some samples are lost and white means no samples are lost.  In
comparison to the `plot_missing_samples_using_rectangles` plot, the
`plot_missing_samples_using_bitmap` function shows us that the
circuits in REDD always lose >20% of their samples, but these
dropouts are spread evenly.


Exploring a single appliance
----------------------------

Let's get a more precise understanding of the dropout rate of a
REDD circuit by getting the dropout rate per day:

.. ipython:: python

   import nilmtk.stats.electricity.single as nstats

   oven = electric.appliances[('oven', 1)]
   nstats.dropout_rate_per_period(data=oven, rule='D')

And a histogram of power consumption:

.. ipython:: python

   oven_above_zero = (oven[oven > 1]).icol(0).dropna()
   plt.xlabel('power (watts)')
   plt.ylabel('frequency')
   @savefig oven_power_hist.png
   plt.hist(oven_above_zero.values, bins=100)

.. ipython:: python
   :suppress:

   plt.close('all')

So we now know that the oven spends a lot of its time consuming about
2-50 Watts but it appears to be properly 'on' when it's consuming over
1600 watts.  So let's use 1000 watts as the on power threshold.

And some more stats:

.. ipython:: python

   nstats.get_sample_period(oven)

   # Get the number of hours the oven was on for:
   nstats.hours_on(oven, on_power_threshold=1000)

   # Get the total kWh consumed by the oven:
   nstats.energy(oven)

   # Or the joules consumed:
   nstats.energy(oven, unit='joules')

   # Or the usage (hours on and energy used) per day:
   nstats.usage_per_period(oven, freq='D', on_power_threshold=1000).head(n=10)

And we can plot some histograms to get an understanding of the
behaviour of an appliance. Let's see the usage of the appliance
hour-by-hour over an average day:

.. ipython:: python

   dist = nstats.activity_distribution(oven, bin_size='H', timespan='D', on_power_threshold=1000)
   x = np.arange(dist.size)
   plt.ylabel('frequency')
   plt.xlabel('hour of day')
   plt.xlim([0,24])
   @savefig activity_dist_day.png
   plt.bar(x, dist.values)
   
.. ipython:: python
   :suppress:

   plt.close('all')

Not surprisingly, the oven is used most often around lunch and dinner times.

Or the behaviour day-by-day over an average week:

.. ipython:: python

   dist = nstats.activity_distribution(oven, bin_size='D', timespan='W', on_power_threshold=1000)
   x = np.arange(dist.size)
   plt.ylabel('frequency')
   plt.xlabel('day of week')
   plt.xticks(np.arange(7)+0.5, calendar.day_name[0:7])
   @savefig activity_dist_week.png
   plt.bar(x, dist.values)
    
.. ipython:: python
   :suppress:

   plt.close('all')

We can see that not much cooking was done in the middle of the week.

Let's find out length of time that the oven tends to be active for
across the dataset.

.. ipython:: python
   
   # Get a Series of booleans indicating when the oven is on:
   on_series = nstats.on(oven, on_power_threshold=1000)
   
   # Now get the length of every on-duration
   on_durations = nstats.durations(on_series, 
                                   on_or_off='on',
                                   ignore_n_off_samples=10)
   plt.xlabel('minutes on')
   plt.ylabel('frequency')
   @savefig on_durations.png         
   plt.hist(on_durations/60, bins=10)

.. ipython:: python
   :suppress:

   plt.close('all')
