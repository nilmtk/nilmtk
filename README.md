# NILMTK: Non-Intrusive Load Monitoring Toolkit

Non-Intrusive Load Monitoring (NILM) is the process of estimating the
energy consumed by individual appliances given just a whole-house
power meter reading.  In other words, it produces an (estimated)
itemised energy bill from just a single, whole-house power meter.

NILMTK is a toolkit designed to help **researchers** evaluate the accuracy of NILM algorithms. If you are a new Python user, it is recommended to educate yourself on [Pandas](https://pandas.pydata.org/), [Pytables](http://www.pytables.org/) and other tools from the Python ecosystem.

**⚠️It may take time for the NILMTK authors to get back to you regarding queries/issues. However, you are more than welcome to propose changes, support!** Remember to check existing issue tickets, especially the open ones.

# Documentation

[NILMTK Documentation](https://github.com/nilmtk/nilmtk/tree/master/docs/manual)

# Installation

## UV Support
This Python package uses uv for installation. uv is a fast and modern Python package manager that replaces tools like pip and virtualenv, with support for pyproject.toml and ultra-fast dependency resolution. 

To install NILMTK, first install [uv](https://docs.astral.sh/uv/getting-started/installation/) and then run:<br>
```
uv pip install git+https://github.com/nilmtk/nilmtk.git
```

## Docker Support
Docker is an open-source platform for developing, shipping, and running applications in lightweight, portable containers that bundle code, runtime, libraries, and system tools into a single package. It ensures everyone runs the same environment, regardless of host OS, and keeps NILMTK’s dependencies contained without polluting the system Python.


Build and run locally
```
docker build -t nilmtk-uv .
docker run --rm -it nilmtk-uv bash
```
Pull the pre-built image
```
docker pull ghcr.io/enfuego27826/nilmtk:latest
docker run --rm -it ghcr.io/enfuego27826/nilmtk:latest bash
```

It came to our attention that some users follow third-party tutorials to install NILMTK. Always remember to check the dates of such tutorials, many are very outdated and don't reflect NILMTK's current version or the recommended/supported setup.

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

# Publications

If you use NILMTK in academic work then please consider citing our papers. Here are some of the publications (contributors, please update this as required):

1. Nipun Batra, Jack Kelly, Oliver Parson, Haimonti Dutta, William Knottenbelt, Alex Rogers, Amarjeet Singh, Mani Srivastava. NILMTK: An Open Source Toolkit for Non-intrusive Load Monitoring. In: 5th International Conference on Future Energy Systems (ACM e-Energy), Cambridge, UK. 2014. DOI:[10.1145/2602044.2602051](http://dx.doi.org/10.1145/2602044.2602051). arXiv:[1404.3878](http://arxiv.org/abs/1404.3878).
2. Nipun Batra, Jack Kelly, Oliver Parson, Haimonti Dutta, William Knottenbelt, Alex Rogers, Amarjeet Singh, Mani Srivastava. NILMTK: An Open Source Toolkit for Non-intrusive Load Monitoring". In: NILM Workshop, Austin, US. 2014 \[[pdf](http://nilmworkshop14.files.wordpress.com/2014/05/batra_nilmtk.pdf)\]
3. Jack Kelly, Nipun Batra, Oliver Parson, Haimonti Dutta, William Knottenbelt, Alex Rogers, Amarjeet Singh, Mani Srivastava. Demo Abstract: NILMTK v0.2: A Non-intrusive Load Monitoring Toolkit for Large Scale Data Sets. In the first ACM Workshop On Embedded Systems For Energy-Efficient Buildings, 2014. DOI:[10.1145/2674061.2675024](http://dx.doi.org/10.1145/2674061.2675024). arXiv:[1409.5908](http://arxiv.org/abs/1409.5908).
4. Nipun Batra, Rithwik Kukunuri, Ayush Pandey, Raktim Malakar, Rajat Kumar, Odysseas Krystalakos, Mingjun Zhong, Paulo Meira, and Oliver Parson. 2019. Towards reproducible state-of-the-art energy disaggregation. In Proceedings of the 6th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation (BuildSys '19). Association for Computing Machinery, New York, NY, USA, 193–202. DOI:[10.1145/3360322.3360844](https://doi.org/10.1145/3360322.3360844)

Please note that NILMTK has evolved *a lot* since most of these papers were published! Please use the [online docs](https://github.com/nilmtk/nilmtk/tree/master/docs/manual)
as a guide to the current API. 

# Brief history

* August 2019: v0.4 released with the new API. See also [NILMTK-Contrib](https://github.com/nilmtk/nilmtk-contrib).
* June 2019: v0.3.1 released on [Anaconda Cloud](https://anaconda.org/nilmtk/nilmtk/).
* Jav 2018: Initial Python 3 support on the v0.3 branch
* Nov 2014: NILMTK wins best demo award at [ACM BuildSys](http://www.buildsys.org/2014/)
* July 2014: v0.2 released
* June 2014: NILMTK presented at [ACM e-Energy](http://conferences.sigcomm.org/eenergy/2014/)
* April 2014: v0.1 released

For more detail, please see our [changelog](https://github.com/nilmtk/nilmtk/blob/master/docs/manual/development_guide/changelog.md).
