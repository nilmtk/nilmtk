***********************
Introducing NILMTK v0.2
***********************

NILMTK v0.2 is a complete re-write of NILMTK from the ground up.

Some feature highlights of v0.2:

* The entire pipeline (loading, stats, preprocessing, disag, metrics)
  can cope with arbitrarily large datasets whilst being gently on the
  system RAM usage
* Full metadata support, including for arbitrary wiring hierarchies
* MeterGroup allows users to select and aggregate meters in powerful
  ways with a single line of code.  e.g. to see the total energy used
  by all the lights in a house just do
  'elec.select(category='lighting').total_energy()'.
* Much more elegant handling of "dual supply" and split-phase mains
  (using MeterGroup)
* Experimental support for preconditions checks for some stats
  function.
* OOP design hides complexity from the user
* CO works well
* Most metrics have been converted
* REDD converter works
* about 40 unit tests
* Disag output is now saved chunk-by-chunk to disk, allowing us to
  disag and do metrics on arbitrarily large datasets.  Also lays the
  foundation for sharing raw disag output and for users to be able to
  ignore all the NILMTK pipeline except the metrics
* Automatic selection of the best type of power measurement (reactive,
  active or apparent) etc etc...

However, there are still some things that are implemented in NILMTK
v0.1 but are not yet in v0.2.  For example:

* v0.2 does not yet include all the dataset importers that v0.1 had.
* v0.2 only has one disag algo: CO.  It should be fairly simple to
  port the FHMM algo from v0.1
* There are a few stats functions and metrics which we haven't had a
  chance to port.
