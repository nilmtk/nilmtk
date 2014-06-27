***************************
File formats used in NILMTK
***************************

NILMTK uses the metadata schema defined by the `NILM Metadata project
<https://github.com/nilmtk/nilm_metadata>`_.


HDF5
----

The `HDF5 file format <http://www.hdfgroup.org/HDF5>`_ is developed by
The HDF Group.  It is a binary file format with good support in a
number of programming languages.

In NILMTK, there is a one-to-one relationship between tables in the
HDF5 file and physical meters.  In other words, for each physical
meter there is a single table in HDF5.

The hierarchy used in HDF follows the pattern
``/building<i>/elec/meter<j>`` where ``i`` and ``j`` are integers
starting from 1.

Physical quantities are defined in the metadata.  We always use SI
units or SI derived units.  e.g. We use watts not kW.

