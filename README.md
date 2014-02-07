nilmtk: Non-Intrusive Load Monitoring Toolkit
======

Non-Intrusive Load Monitoring (NILM) is the process of estimating the energy consumed by individual
appliances given just a whole-house power meter
reading.  In other words, it produces an (estimated) itemised
energy bill from just a single, whole-house power meter.

NILM is sometimes called:

* "non-intrusive appliance load monitoring (NALM or NIALM)"
* "[electriciy | energy | smart meter] disaggregation"

Below is an illustration<sup>1</sup> of what NILM, in general, can do.

<img src="https://dl.dropboxusercontent.com/u/75845627/misc/after_disagg.png" alt="Drawing" style="width: 40% height: 40%;"/>

##### NILMTK Documentation

http://nilmtk.github.io/nilmtk/

##### Current state of the project

The project is in its early stages.

At present, NILMTK can:

* import data from the following data sets: REDD, Pecan Street, AMPds, iAWE, SMART* and UKPD
* store imported data in a common in-memory format, which can also be exported to a standard on-disk format
* compute diagnostic statistics for the raw data (uptime, dropout rate, find gaps etc)
* pre-process the data (downsample, fill gaps, mask appliance data with gaps from the mains data, select only contiguous blocks, normalise power etc)
* compute usage statistics (distribution of appliance activity per day, week or month, distribution of on-power, proportion of total energy per appliance, on-durations etc)
* provide a common input and output interface to NILM algorithms
* disaggregate data using two supervised benchmark algorithms: combinatorial optimisation and factorial hidden Markov model (these are not competitive with the current state-of-the-art.  More sophisticated NILM algorithms will be added to NILMTK later this year.)
* compute a range of NILM performance metrics (confusion matrices, error in assigned energy, F1 score, fraction of energy assigned correctly etc).
* work has started on a disaggregation web interface.  [Here's a demo](http://energy.iiitd.edu.in:5002/).

##### Installing

If you just want to use the code without modifying it then:

`python setup.py install`

(you may have to run as `sudo`)

If you want to get involved in development then:

`python setup.py develop`

##### Software Dependencies

1. Pandas
2. matplotlib
3. numpy => 1.8
4. scikit-learn > 0.13

##### Getting started

Loading a supported dataset is simple.  For example, to load [REDD](http://redd.csail.mit.edu/):

```python
from nilmtk.dataset import REDD
redd = REDD()
redd.load('/data/REDD/low_freq/')
```

Please see the [`examples`](https://github.com/nilmtk/nilmtk/tree/master/examples) folder for further information on getting started.  

We have started writing [a user guide](http://nilmtk.github.io/nilmtk/userguide.html) although it is by no means complete yet.  A full user guide will be written in February.

We also have [API documentation](http://nilmtk.github.io/nilmtk/nilmtk.html).  This documentation covers almost all functions in NILMTK but needs some tidying up which will be done over February.

##### Further info

Please see [the nilmtk wiki](https://github.com/nilmtk/nilmtk/wiki) for more details.

##### Development plans for February and March:

* To make the code as easy as possible to use and to maintain, it will be undergoing some refactoring in Febrary 2014.
* Improve the documentation
* Write more unit tests
* easy installation via pip
* [Build a semantic wiki for storing information relevant to NILM (e.g. a database of appliances)](http://jack-kelly.com/wiki_and_online_community_for_electricity_disaggregation) - NILMTK will integrate with the wiki wherever possible, especially regarding appliance metadata.

##### Notes

1. The image is from the following paper and since the main author is contributing to nilmtk, so no permission issues.
The reference is: Nipun Batra, Haimonti Dutta, Amarjeet Singh, “INDiC: Improved Non-Intrusive load monitoring using load Division and     Calibration”, to appear at the 12th International Conference on Machine Learning and Applications (ICMLA’13) will be     held in Miami, Florida, USA, December 4 – December 7, 2013 
    [Preprint](http://nipunbatra.files.wordpress.com/2013/09/icmla.pdf).  [IPython notebook](http://www.iiitd.edu.in/~amarjeet/Research/indic.html)
