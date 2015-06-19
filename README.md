
NILMTK: Non-Intrusive Load Monitoring Toolkit
======

Non-Intrusive Load Monitoring (NILM) is the process of estimating the energy consumed by individual
appliances given just a whole-house power meter
reading.  In other words, it produces an (estimated) itemised
energy bill from just a single, whole-house power meter.

NILM is sometimes called:

* "non-intrusive appliance load monitoring (NALM or NIALM)"
* "[electriciy | energy | smart meter] disaggregation"

Below is an example of sub-metered appliance-level data, which NILM algorithms aim to produce. N.B. this is not the output of a NILMTK algorithm!

<img src="https://dl.dropboxusercontent.com/u/75845627/nilmtk/submetered.png" alt="Drawing" style="width: 40% height: 40%;"/>

##### Documentation

http://nilmtk.github.io/nilmtk/

##### Installation guide

http://nilmtk.github.io/nilmtk/master/install.html

##### Publications

* Batra, N., Kelly, J., Parson, O., Dutta, H., Knottenbelt, W., Rogers, A., Singh, A., Srivastava, M. (2014). NILMTK: An Open Source Toolkit for Non-intrusive Load Monitoring. In Fifth International Conference on Future Energy Systems (ACM e-Energy). Cambridge, UK. arXiv:[1404.3878](http://arxiv.org/abs/1404.3878) DOI:[10.1145/2602044.2602051](http://dx.doi.org/10.1145/2602044.2602051)

Bibtex:

```
@inproceedings{NILMTK,
   title         = {NILMTK: An Open Source Toolkit for
                    Non-intrusive Load Monitoring},
   year          = {2014},
   author        = {Batra, Nipun and Kelly, Jack and Parson, Oliver and
                    Dutta, Haimonti and Knottenbelt, William and
                    Rogers, Alex and Singh, Amarjeet and Srivastava, Mani},
   booktitle     = {Fifth International Conference on Future Energy
                    Systems (ACM e-Energy)},
   address       = {Cambridge, UK},
   archivePrefix = {arXiv},
   arxivId       = {1404.3878},
   doi           = {10.1145/2602044.2602051},
   eprint        = {1404.3878}
}
```

* Kelly, J., Batra, N., Parson, O., Dutta, H., Knottenbelt, W., Rogers,
A., Singh, A., Srivastava, M. (2014). NILMTK v0.2: A Non-intrusive
Load Monitoring Toolkit for Large Scale Data Sets.  In
The first ACM Workshop On Embedded Systems For Energy-Efficient Buildings at BuildSys 2014. Memphis, USA. DOI:[10.1145/2674061.2675024](http://dx.doi.org/10.1145/2674061.2675024) arXiv:[1409.5908](http://arxiv.org/abs/1409.5908)

Bibtex:

```
@Inproceedings{kelly2014NILMTKv02,
  Title      = {NILMTK v0.2: A Non-intrusive Load Monitoring
                Toolkit for Large Scale Data Sets},
  Author     = {Kelly, Jack and Batra, Nipun and Parson, Oliver and
                Dutta, Haimonti and Knottenbelt, William and
                Rogers, Alex and Singh, Amarjeet and Srivastava, Mani},
  Booktitle  = {The first ACM Workshop On Embedded Systems For
                Energy-Efficient Buildings at BuildSys 2014},
  Year       = {2014},
  Doi        = {10.1145/2674061.2675024},
  Eprint     = {1409.5908},
  Eprinttype = {arXiv},
  Address    = {Memphis, USA}
}
```

N.B. NILMTK has evolved *a lot* since these papers were published! Please use the
[online docs](http://nilmtk.github.io/nilmtk/master/index.html)
as a guide to the current API.

##### Keeping up to date with NILMTK

* [NILMTK-Announce mailing list](https://groups.google.com/forum/#!forum/nilmtk-announce): stay up to speed with NILMTK.  It will be a fairly low-traffic mailing list.  We'll just announce new versions, new docs etc.
* [NILMTK on Twitter](https://twitter.com/nilmtk).

##### History

* April 2014: v0.1 released
* June 2014: NILMTK presented at [ACM e-Energy](http://conferences.sigcomm.org/eenergy/2014/)
* July 2014: v0.2 released
* Nov 2014: NILMTK wins best demo award at [ACM BuildSys](http://www.buildsys.org/2014/)

##### Current state of the project

The project is in its early stages.

Please note that NILMTK is currently a research tool.  It is not yet
ready for use by end-users, although we certainly hope that NILMTK
will be capable of doing 'plug and play' disaggregation in the future.

Please see the docs for more info.

##### Submitting a bug report

Please use our [github issue queue](https://github.com/nilmtk/nilmtk/issues) to submit bug reports, rather than emailing them, which will allow any of us to respond to your issue.  If your bug concerns one of NILMTK's dependencies then please contact the relevant project.

Before opening an issue:

1. Search the issue queue in case a duplicate already exists
2. Pull the latest changes from the repository master branch to see if the error goes away

If not, please open a new issue, ensuring:

1. The title summarises the issue
2. The issue is described in prose
3. A snippet of code is included which will allow us to recreate the bug
4. A copy-paste of the stack produced error
5. Include a copy-paste of the output from nilmtk.utils.show_versions()

##### Test coverage

[![Build Status](https://travis-ci.org/nilmtk/nilmtk.svg?branch=master)](https://travis-ci.org/nilmtk/nilmtk) 

[![Coverage Status](https://coveralls.io/repos/nilmtk/nilmtk/badge.png)](https://coveralls.io/r/nilmtk/nilmtk)

[![Code Health](https://landscape.io/github/nilmtk/nilmtk/master/landscape.png)](https://landscape.io/github/nilmtk/nilmtk/master)

