.. currentmodule:: nilmtk
      
***********************
HDF5 Metadata structure
***********************

HDF5 files are structured as ``/building1/electric/meters``

``store.root._v_attrs.dataset``
-------------------------------

* ``meter_devices`` : dict. Keys are device model names
  (e.g. 'EnviR').  Values are also dicts with keys:

  - ``manufacturer`` : string
  - ``model`` : string
  - ``sample_period`` : int or float, seconds
  - ``max_sample_period`` : int or float, seconds
  - ``measurements`` : list of
    nilmtk.Measurements. e.g. ``[Power('apparent')]``
  - ``measurement_limits`` : dict. Each key is an element from
    ``measurements``.  Each value is a dict with two keys: ``lower``
    and ``upper``, both mapping to numbers.
