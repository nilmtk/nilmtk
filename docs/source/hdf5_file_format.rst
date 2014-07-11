***************************
HDF5 file format for NILMTK
***************************

The `HDF5 file format <http://www.hdfgroup.org/HDF5>`_ is developed by
The HDF Group.  It is an open, binary file format with good support in
a number of programming languages including Python.  In principal, you
could use any language to create your dataset converter but we would
recommend the Python language and we have not tried writing converters
in another other language.  Working with HDF5 files is really easy in
Python using `Pandas <http://pandas.pydata.org/>`_ (which, in turn,
uses the excellent `PyTables <http://www.pytables.org>`_ package).

**Background reading**

If you haven't done so already, then it might be worth reading the
`Pandas tutorial on working with HDF5 files in Pandas
<http://pandas.pydata.org/pandas-docs/stable/io.html#hdf5-pytables>`_.

Writing a dataset converter
===========================

If you would like to contribute a new dataset converter (or just
better understand the existing converters) then the best way to get up
to speed is probably to read this document first.  This document
explains the layout of data in a NILMTK HDF5 file. Then take a look at
the REDD converter for NILMTK (in
``nilmtk/nilmtk/dataset_converters/redd``).

NILMTK dataset converters output an HDF5 file which contains both the
time series data from each meter as well as all relevant metadata.

Time series data
================

Data from each physical meter is stored in its own table (i.e. there
is a one-to-one relationship between tables in the HDF5 file and
physical meters).

Table location (keys)
---------------------

Tables in HDF5 are identified by a hierarchical key.  Each key is a
string. Levels in the hierarchy are separated by a ``/`` character in
the key.  NILMTK uses keys in the form ``/building<i>/elec/meter<j>``
where ``i`` and ``j`` are integers starting from 1.  ``i`` is the
building instance and ``j`` is the meter instance.  For example, the
table storing data from meter instance 1 in building instance 1 would
have the key ``/building1/elec/meter1``.  (We use ``elec`` in the
hierarchy to allow us, in the future, to add support for other sensor
types like water and gas meters).

Contents of each table
----------------------

Illustration
^^^^^^^^^^^^

+---------------------------+------------+----------+----------+
|                           | power      | energy   | voltage  |
+                           +------------+----------+----------+
|                           | active     | reactive |          |
+===========================+============+==========+==========+
| 2014-07-01T05:00:14+00:00 |   0.1      |  0.01    | 231.1    |
+---------------------------+------------+----------+----------+
| 2014-07-01T05:00:15+00:00 | 100.5      | 10.5     | 232.1    |
+---------------------------+------------+----------+----------+

Index column
^^^^^^^^^^^^

The index column is a datetime represented on disk as a nano-second
precision (!) UNIX timestamp stored as an unsigned 64-bit int.  In
Python, we used a timezone-aware ``numpy.datetime64``.

Measurement columns
^^^^^^^^^^^^^^^^^^^

Every column apart from the index column holds a measurement taken by
the meter. These measurements could be power demand, energy,
cumulative energy, voltage or current. Each measurement is
represented as 32-bit floating point number.

We always use SI units or SI derived units and NILMTK assumes that no
unit prefix (e.g. 'kilo-' or 'mega-') has been applied.  In other
words, NILMTK assumes a unit multiplier of 1.  For example, we always
use watts (not kW) for active power.  If the source dataset uses, say,
kW then please multiply these values by 1000 in your converter.

Column labels
^^^^^^^^^^^^^

Column labels are hierarchical with 2 levels (`hierarchical labels are
very well supported by Pandas
<http://pandas.pydata.org/pandas-docs/stable/indexing.html#hierarchical-indexing-multiindex>`_).
The top level describes the ``physical_quantity`` being measured.  The
seconds level describes the ``type`` which, at present, is used for
energy and power columns to describe whether the alternating current
measurement is ``apparent``, ``active`` or ``reactive``.  We use a
controlled vocabulary for both ``physical_quanitity`` and ``type``.
For the full details of this controlled vocabulary, please see the
documentation for ``physical_quantity`` and ``type`` under the
``measurements`` property for the `MeterDevice object in NILM Metadata
<http://nilm-metadata.readthedocs.org/en/latest/dataset_metadata.html#meterdevice>`_.

Compression
^^^^^^^^^^^

We use ``zlib`` to compress our HDF5 files.  ``bzip2`` results in
slightly smaller files (261MB for ``bzip2`` versus 273MB for ``zlip``
for REDD) but doesn't appear to be compatible with `HDFView
<http://www.hdfgroup.org/products/java/release/download.html>`_.

Metadata
========

In order for NILMTK to be able to load the dataset, we need to add
metadata to the HDF5 file.  NILMTK uses the `NILM Metadata schema
<https://github.com/nilmtk/nilm_metadata>`_.

If the dataset is already described in YAML using the NILM Metadata
schema then just call ``nilm_metadata.convert_yaml_to_hdf5()``.

If the dataset is not already described using the NILM Metadata schema
then it will be necessary to do so.  If it is a small dataset then you
could manually write the YAML files and then convert these to HDF5
(this is how our REDD converter works).  If it is a large dataset then
it would be better to programmatically convert the dataset's own
metadata to NILM Metadata and store the metadata directly in the HDF5
file.  For an introduction to NILM Metadata first read the `README
<https://github.com/nilmtk/nilm_metadata/blob/master/README.md>`_ and
then `the tutorial
<http://nilm-metadata.readthedocs.org/en/latest/tutorial.html>`_ and
finally refer to `the dataset_metadata doc page
<http://nilm-metadata.readthedocs.org/en/latest/dataset_metadata.html>`_
for the full description of the metadata schema.
