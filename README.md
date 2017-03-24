# NILMTK: Non-Intrusive Load Monitoring Toolkit

Non-Intrusive Load Monitoring (NILM) is the process of estimating the
energy consumed by individual appliances given just a whole-house
power meter reading.  In other words, it produces an (estimated)
itemised energy bill from just a single, whole-house power meter.

NILM is sometimes called:

* "non-intrusive appliance load monitoring (NALM or NIALM)"
* "[electriciy | energy | smart meter] disaggregation"

NILMTK is a toolkit designed to help *researchers* evaluate the
accuracy of NILM algorithms. **NILMTK is not yet suitable for end
users because NILMTK is not yet capable of out-of-the-box
disaggregation (i.e. disaggregation where you do not yet have
submetered training data), although we hope that it might be some time
in the future**.

Below is an example of sub-metered appliance-level data, which NILM
algorithms aim to produce. N.B. this is not the output of a NILMTK
algorithm!

<img src="https://dl.dropboxusercontent.com/u/75845627/nilmtk/submetered.png" alt="Drawing" style="width: 40% height: 40%;"/>


# Why a toolkit for NILM?

We quote our [NILMTK paper](http://arxiv.org/pdf/1404.3878v1.pdf)
explaining the need for a NILM toolkit:

  > Empirically comparing disaggregation algorithms is currently
  > virtually impossible. This is due to the different data sets used,
  > the lack of reference implementations of these algorithms and the
  > variety of accuracy metrics employed.


# What NILMTK provides

To address this challenge, we present the Non-intrusive Load Monitoring
Toolkit (NILMTK); an open source toolkit designed specifically to enable
the comparison of energy disaggregation algorithms in a reproducible
manner. This work is the first research to compare multiple
disaggregation approaches across multiple publicly available data sets.
NILMTK includes:

-  parsers for a range of existing data sets (8 and counting)
-  a collection of preprocessing algorithms
-  a set of statistics for describing data sets
-  a number of [reference benchmark disaggregation algorithms](https://github.com/nilmtk/nilmtk/wiki/NILM-Algorithms)
-  a common set of accuracy metrics
-  and much more!


# Documentation

[NILMTK Documentation](https://github.com/nilmtk/nilmtk/tree/master/docs/manual)


# Publications

Please see our [list of NILMTK publications](http://nilmtk.github.io/#publications).  If you use NILMTK in academic work then please consider citing our papers.

Please note that NILMTK has evolved *a lot* since these papers were published! Please use the
[online docs](https://github.com/nilmtk/nilmtk/tree/master/docs/manual)
as a guide to the current API.


# Keeping up to date with NILMTK

* [NILMTK-Announce mailing list](https://groups.google.com/forum/#!forum/nilmtk-announce): stay up to speed with NILMTK.  This is a low-traffic mailing list.  We'll just announce new versions, new docs etc.
* [NILMTK on Twitter](https://twitter.com/nilmtk).


# History

* April 2014: v0.1 released
* June 2014: NILMTK presented at [ACM e-Energy](http://conferences.sigcomm.org/eenergy/2014/)
* July 2014: v0.2 released
* Nov 2014: NILMTK wins best demo award at [ACM BuildSys](http://www.buildsys.org/2014/)

For more detail, please see our [changelog](https://github.com/nilmtk/nilmtk/blob/master/docs/manual/development_guide/changelog.md).

