#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright 2013-2020 NILMTK developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import warnings
import subprocess
from setuptools import setup, find_packages

# Retrieve tag from CI if present
TRAVIS_TAG = os.environ.get("TRAVIS_TAG", "")

if TRAVIS_TAG:
    VERSION = TRAVIS_TAG
    ISRELEASED = "dev" not in TRAVIS_TAG
    QUALIFIER = ""
else:
    MAJOR, MINOR, MICRO = 0, 4, 0
    DEV = 1
    ISRELEASED = False
    VERSION = f"{MAJOR}.{MINOR}.{MICRO}"
    QUALIFIER = ""

FULLVERSION = VERSION

if not ISRELEASED and not TRAVIS_TAG:
    try:
        # Try standard git first
        proc = subprocess.Popen(
            ["git", "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE
        )
    except OSError:
        # Fallback for msysgit on Windows
        proc = subprocess.Popen(
            ["git.cmd", "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE
        )
    raw = proc.stdout.read().strip()
    if sys.version_info[0] >= 3:
        raw = raw.decode("ascii")
    rev = raw or None

    if rev:
        FULLVERSION += f".dev{DEV}+git.{rev}"
    else:
        FULLVERSION += f".dev{DEV}"
else:
    FULLVERSION += QUALIFIER

def write_version_py(filename=None):
    template = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = os.path.join(os.path.dirname(__file__),
                                "nilmtk", "version.py")
    with open(filename, "w", encoding="utf-8") as fp:
        fp.write(template % (FULLVERSION, VERSION))

write_version_py()

setup(
    name="nilmtk",
    version=FULLVERSION,
    packages=find_packages(),
    package_data={"": ["*.yaml"]},
    install_requires=[
	"nilm_metadata @ git+https://github.com/nilmtk/nilm_metadata.git",
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "networkx>=3.2.1",
        "scipy",
        "tables",
        "scikit-learn>=0.21.2",
        "hmmlearn>=0.2.1",
        "pyyaml",
        "matplotlib>=3.10.3",
        "jupyterlab",
    ],
    description=(
        "Estimate the energy consumed by individual appliances "
        "from whole-house power meter readings"
    ),
    author="NILMTK developers",
    url="https://github.com/nilmtk/nilmtk",
    download_url=(
        "https://github.com/nilmtk/nilmtk/"
        "tarball/master#egg=nilmtk-dev"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache 2.0",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords=(
        "smartmeters power electricity energy analytics "
        "redd disaggregation nilm nialm"
    ),
)
