
Non-Intrusive Load Monitoring (NILM)
====================================

Non-Intrusive Load Monitoring (NILM) is the process of estimating the
energy consumed by individual appliances given just a whole-house power
meter reading. In other words, it produces an (estimated) itemised
energy bill from just a single, whole-house power meter.

NILM is sometimes called:

-  "non-intrusive appliance load monitoring (NALM or NIALM)"
-  "[electriciy \| energy \| smart meter] disaggregation"

Below is an illustration (produced using nilmtk) of what NILM, in
general, can do.

.. figure:: https://dl.dropboxusercontent.com/u/75845627/nilmtk/submetered.png
   :alt: Drawing

   Drawing

Why a toolkit for NILM?
=======================

We quote our `nilmtk paper <http://arxiv.org/pdf/1404.3878v1.pdf>`__
explaining the need for a NILM toolkit.

    Empirically comparing disaggregation algorithms is currently
    virtually impossible. This is due to the different data sets used,
    the lack of reference implementations of these algorithms and the
    variety of accuracy metrics employed. To address this challenge, we
    present the Non-intrusive Load Monitoring Toolkit (NILMTK); an open
    source toolkit designed specifically to enable the comparison of
    energy disaggregation algorithms in a reproducible manner. This work
    is the first research to compare multiple disaggregation ap-
    proaches across multiple publicly available data sets. Our toolkit
    includes parsers for a range of existing data sets, a collection of
    preprocessing algorithms, a set of statistics for describing data
    sets, two reference benchmark disaggregation algorithms and a suite
    of accuracy metrics.

