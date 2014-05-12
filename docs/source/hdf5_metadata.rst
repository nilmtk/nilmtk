.. currentmodule:: nilmtk
      
***********************
HDF5 Metadata structure
***********************

HDF5 files are structured in the form ``/building1/electric/meter1``

``store.root._v_attrs.metadata``
-------------------------------

:meter_devices: (string)
  Keys are device model names (e.g. 'EnviR').
  The purpose is to record information about
  specific  models of meter.  Values are also dicts with keys:

  :model: (string) The model name for this meter device.
  :manufacturer: (string)
  :sample_period: (number) seconds
  :max_sample_period: (number) seconds
  :measurements: (list of ``nilmtk.Measurement`` objects) e.g. ``[Power('apparent')]``
  :measurement_limits: (dict) Each key is an element from
    ``measurements``.  Each value is a dict with two keys: ``lower``
    and ``upper``, each mapping to a number representing the limits in
    units of watts.
    e.g. ``{Power('apparent'): {'lower':0, 'upper': 3000}}``


``store.get_storer('/building1/electric/meter1').attrs.metadata``
-----------------------------------------------------------------

:device_model: (string): ``model`` which keys into ``meter_devices``
:instance: (int starting from 1) the meter instance within the building.
:building: (int starting from 1) the building instance.
:dataset: (string)
:submeter_of: (int): the meter instance of the upstream meter.
:site_meter: (boolean): True if this is a site meter (i.e. furthest
             upstream meter)
:preprocessing:
:dominant_appliance: (<appliance_type), <instance>) which is responsible for 
          most of the power demand on this channel.


``store.get_storer('/building1/electric/meter1').attrs.appliances``
-------------------------------------------------------------------

A list of appliance dicts.  Each dict has:

:type: (string): appliance type. Use NILM Metadata controlled
       vocabulary.
:instance: (int starting from 1): instance of this appliance within
           the building.
:on_power_threshold: (number) watts
:minimum_off_duration: (timedelta)
:minimum_on_duration: (timedelta)

