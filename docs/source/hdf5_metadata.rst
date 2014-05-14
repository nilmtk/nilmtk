.. currentmodule:: nilmtk
      
***********************
HDF5 Metadata structure
***********************

HDF5 files are structured in the form ``/building1/electric/meter1``


Dataset metadata
----------------

``store.root._v_attrs.metadata``

:meter_devices: (dict)
  Keys are device model names (e.g. 'EnviR').
  The purpose is to record information about
  specific  models of meter.  Values are also dicts with keys:

  :model: (string) The model name for this meter device.
  :model_url: (string)
  :manufacturer: (string)
  :manufacturer_url: (string)
  :sample_period: (number) seconds
  :max_sample_period: (number) seconds
  :measurements: (list of ``nilmtk.Measurement`` objects) e.g. ``[Power('apparent')]``
  :measurement_limits: (dict) Each key is an element from
    ``measurements``.  Each value is a dict with two keys: ``lower``
    and ``upper``, each mapping to a number representing the limits in
    units of watts.
    e.g. ``{Power('apparent'): {'lower':0, 'upper': 3000}}``


Building metadata
-----------------

``store.root.building1._v_attrs.metadata``

:instance: (int) The building instance in this dataset, starting from 1
:dataset: (string)
:original_name: (string)


Meter metadata
--------------

``store.root.building1.electric.meter1._v_attrs.metadata``

:device_model: (string): ``model`` which keys into ``meter_devices``
:instance: (int starting from 1) the meter instance within the building.
:building: (int starting from 1) the building instance.
:dataset: (string)
:submeter_of: (int): the meter instance of the upstream meter.  Or 0
              to mean 'one of the site_meters'.
:site_meter: (boolean): True if this is a site meter (i.e. furthest
             upstream meter)
:preprocessing: TBD
:dominant_appliance: (<appliance_type), <instance>) which is responsible for 
          most of the power demand on this channel.
:room: (dict) with ``name`` [and ``instance``].
:floor: (int)
:category: (string) e.g. ``lighting`` or ``sockets``.  Use this if this meter
           feeds a group of appliances and if we do not know the
           identity of each individual appliance.  For example, perhaps
           this is a meter which measures the lighting circuit,
           in which case we use ``'category': 'lighting'``.
           Must use the same controlled vocabulary as for
           appliance types.
:appliances: (list of dicts) See section below on 'Appliance metadata'.


Appliance metadata
------------------

Each appliance dict has:

:type: (string): appliance type. Use NILM Metadata controlled
       vocabulary.
:instance: (int starting from 1): instance of this appliance within
           the building.
:on_power_threshold: (number) watts
:minimum_off_duration: (timedelta)
:minimum_on_duration: (timedelta)
:room: (dict) with ``name`` [and ``instance``]
:count: (int) number of appliance instances.  If absent then assumed
        to be 1.
:multiple: (boolean) True if there are more than one but an unknown
           number of these appliances.  If there are more than one
           appliance and the exact number is known then use ``count``.


Categories
----------

:Traditional: wet, cold, consumer electronics, ICT, cooking, heating
:Misc: misc, sockets
:Size: small, large
:Electrical: 
  - lighting, incandescent, fluorescent, compact, linear, LED
  - resistive
  - power electronics
  - SMPS, no PFC, passive PFC, active PFC
  - single-phase induction motor, capacitor start-run, constant torque
