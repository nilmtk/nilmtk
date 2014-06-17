***************************
File formats used in NILMTK
***************************

We use the metadata schema defined by the NILM Metadata project.  
See the `'dataset metadata' sections of the NILM Metadata documentation`_.

.. _`'dataset metadata' sections of the NILM Metadata documentation`: http://nilm-metadata.readthedocs.org/en/latest/dataset_metadata.html


HDF5
----

The `HDF5 file format`_ is developed by The HDF Group.

.. _`HDF5 file format`: http://www.hdfgroup.org/HDF5

In NILMTK, there is a one-to-one relationship between tables in the
HDF5 file and physical sensors.  In other words, for each physical
sensor there is a single table in HDF5.  We use the term ``sensor`` to
refer to a physical device capable of recording one or more parameters
from a single point.

The hierarchy used in HDF follows the pattern
``/building<i>/elec/sensor<j>`` where ``i`` and ``j`` are integers
starting from 1.  ``sensor<j>`` belongs to ``ElecMeter`` instance ``<j>``
(i.e. we use the same number).

Some ElecMeters measure multiple phases (e.g. for a 3-phase mains
supply) or legs (e.g. north American homes often have a single phase
mains supply split into two 120 volt legs).  In this case, a single
ElecMeter will have multiple sensors.  If an ``ElecMeter`` has
multiple sensors then we add a single letter to the end of the sensor
name e.g. ``sensor1a``.
