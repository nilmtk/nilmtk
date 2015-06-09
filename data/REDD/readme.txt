===========================================================================
REDD: The Reference Energy Disaggregation Data Set, Version 1.0
===========================================================================

This document describes the initial release of REDD, a data set for
energy disaggregation.  The data contains power consumption from real
homes, for the whole house as well as for each individual circuit in
the house (labeled by the main type of appliance on that circuit).
The data is intended for use in developing disaggregation methods, which
can predict, from only the whole-home signal, which devices are being
used (though any other uses are of course encouraged as well).

The latest version of the data set will be available at:
http://redd.csail.mit.edu

For an overview of the data collection procedures and a description of
algorithms see:
J. Zico Kolter and Matthew J. Johnson.  REDD: A public data set for
energy disaggregation research.  In proceedings of the SustKDD
workshop on Data Mining Applications in Sustainability, 2011.

All those wishing to reference the data set in academic work should
cite this paper. 

We welcome any comments or questions about the data:
Zico Kolter (kolter@csail.mit.edu)


===========================================================================
File Organization
===========================================================================

The REDD data set contains two main types of home electricity data:
high-frequency current/voltage waveform data of the two power mains
(as well as the voltage signal for a single phase), and
lower-frequency power data including the mains and individual,
labeled circuits in the house.  The data set is organized as follows,
though for convenience the following directories may be available for
separate download:

redd/
  readme.txt             -- this document
  low_freq/              -- ~1Hz power readings, whole home and circuits
  high_freq/             -- aligned and group current/voltage waveforms
  high_freq_raw/         -- raw current/voltage waveforms

 
---------------------------------------------------------------------------
Low Frequency Power Data
---------------------------------------------------------------------------

The low_freq/ directory contains average power readings for both the
two power mains and the individual circuits of the house (eventually,
this will also contain plug loads for houses with individual plug
monitors).  The data is logged at a frequency of about once a second
for a mains and once every three seconds for the circuits.  The
directory is organized as follows:

redd/low_freq/
  house_{1..n}/          -- directories for each house
    labels.dat           -- device category labels for every channel
    channel_{1..k}.dat   -- time/wattage readings for each channel

The main directory consists of several house_i directories, each of
which contain all the power readings for a single house.  Each house
subdirectory consists of a labels.dat and several channels_i.dat
files.  The labels file contains channel numbers and a text label
indicating the general category of device on this channel, for
example:

1 mains_1
2 mains_2
3 refrigerator
4 lighting
...

In cases where the circuit has different device types on it (for
example, circuits that power multiple outlets), we have attempted to
best categorize the main type of appliance on the circuit.

Each channel_i.dat file contains UTC timestamps (as integers) and
power readings (recording the apparent power of the circuit) for
the channel:

...
1306541834      102.964
1306541835      103.125
1306541836      104.001
1306541837      102.994
1306541838      102.361
1306541839      102.589
...


---------------------------------------------------------------------------
High Frequency Waveform Data
---------------------------------------------------------------------------

The high_freq/ directory contains AC waveform data for the power mains
and a single phase of the voltage for the home.  In order to reduce
the data to a manageable size, we have compressed these waveforms using
lossy compression.  Briefly, the procedure is as follows:

Because the voltage signal in most homes is approximately sinusoidal
(unlike the current signals, which can vary substantially from a
sinusoidal wave), we find zero-crossings of the voltage signal to
isolate a single cycle of the AC power.  For the time spanned by this
single cycle, we record both the current and voltage signals, and
report this entire waveform.  However, because the waveforms remain
approximately constant for long periods of time, we only report the
current and voltage waveforms at "change points" in the signal (we
identify change points using a method known as total variation
regularization, but a full description of the approach is outside the
scope of this readme).

As before, the high_freq/ directory contains a subdirectory for each
house, each of which contain current_1.dat, current_2.dat, and
voltage.dat files.

redd/low_freq/
  house_{1..n}/          -- directories for each house
    current_1.dat        -- current waveforms for first power mains
    current_2.dat        -- current waveforms for second power mains
    voltage.dat          -- voltage waveforms

The data files are text files, where each line contains:

1) A decimal UTC timestamp, in the same format as the timestamps for
the low frequency data, but allowing for fractional parts
2) A cycle count.  Although this is represented in the file as a
double, it is in fact an integer that indicates for how many AC cycles
this particular waveform remains.
3) 275 decimal values, indicating the value of the waveform (in amps or
volts), at equally-spaced portions of the cycle.

Thus, an example file might be:

1297340206.597013 135.000000 0.000000 3.623859 7.254136 10.949398 ...
1297340208.844086 722.000000 0.000000 3.638527 7.249567 10.929027 ...
....

Indicating that the waveform in the first line occurred first at
timestamp 1297340206.597013 and lasted for 135 cycles.

---------------------------------------------------------------------------
High Frequency Raw Data
---------------------------------------------------------------------------

Finally, the high_freq_raw/ directory contains raw current and voltage
waveforms (unaligned and without compression), for a small number of
sample points throughout the data.  This is main intended for those
who wish to test different compression/filtering methods beyond what
we do in the high_freq/ data.  Although for practicality we are not
planning to broadly distribute raw data for the entire data set (this
would consist of more than a terabyte of data), if other groups are
able to develop substantially better compression/filtering techniques
then we'd be happy to share the full data or run these proposed
algorithms on the full uncompressed data set.

The high_freq_raw/ directory is organized similar to the high_freq/
directory, except that each current_1, current_2, and voltage files
are instead themselves directories that contain raw binary data:

high_freq_raw/
  house_{1..n}/          -- directories for each house
    current_1/           -- current waveforms for first power mains
      <timestamp>.bz2    -- compressed binary data
      ...
    current_2/           -- current waveforms for second power mains
    voltage/             -- voltage waveforms

The current_1/, current_2/, and voltage/ directories contain a number
of <timestamp>.bz2 files, for example 1303091049.bz2.  These are
compressed binary files containing the raw current/voltage waveforms
from the A/D as a sequence of 32-bit floating point numbers (stored in
little-endian format).  Interspersed through the data are several
'time' ASCII codes (hex: 0x74 0x69 0x6d 0x65).  This indicates that an
8 byte time code is to follow, indicating the the time at the
beginning of the sample.  The time is stored as two binary integers
(again in little endian format): the first indicating the UTC
timestamp in seconds, and the second indicating the fractional portion
of the timestamp in nano-seconds.  The timestamps are taking using the
Windows clock on the collecting machines, so are likely only accurate
to the millisecond level or so.