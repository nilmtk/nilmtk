***************************
File formats used in NILMTK
***************************

We use the metadata schema defined by the NILM Metadata project.  In
particular, see the `NILM Metadata documentation on dataset metadata`_.

.. _`NILM Metadata documentation on dataset metadata`: http://nilm-metadata.readthedocs.org/en/latest/dataset_metadata.html

HDF5
----

The `HDF5 file format`_ was developed and is maintained by The HDF Group.

.. _`HDF5 file format`: http://www.hdfgroup.org/HDF5

The hierarchy used in HDF follows the pattern
``/building<i>/elec/sensor<j>`` where ``i`` and ``j`` are integers
starting from 1.  ``sensor<j>`` belongs to ``ElecMeter instance <j>``
(i.e. we use the same number).  If an ``ElecMeter`` has multiple
sensors then we add a single letter to the end of the sensor name
e.g. ``sensor1a``.
