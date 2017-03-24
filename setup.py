"""
   Copyright 2013 nilmtk authors

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

from setuptools import setup, find_packages, Extension
from os.path import join
import os
import sys
import warnings

import numpy

"""
CYTHON_DIR = 'nilmtk/disaggregate/feature_detectors'

try:
    # This trick adapted from 
    # http://stackoverflow.com/a/4515279/732596
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

if use_cython:
    sources = [join(CYTHON_DIR, '_feature_detectors.pyx')]
    extensions = [Extension("nilmtk.disaggregate._feature_detectors", 
                            sources=sources,
                            include_dirs=[numpy.get_include()])]
    ext_modules = cythonize(extensions)
else:
    ext_modules = [
        Extension("nilmtk.disaggregate._feature_detectors", 
                  [join(CYTHON_DIR, '_feature_detectors.c')],
                  include_dirs=[numpy.get_include()]),
    ]
    

"""

"""
Following Segment of this file was taken from the pandas project(https://github.com/pydata/pandas) 
"""
# Version Check

MAJOR = 0
MINOR = 2
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
QUALIFIER = ''

FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'
    try:
        import subprocess
        try:
            pipe = subprocess.Popen(["git", "rev-parse", "--short", "HEAD"],
                                    stdout=subprocess.PIPE).stdout
        except OSError:
            # msysgit compatibility
            pipe = subprocess.Popen(
                ["git.cmd", "rev-parse", "--short", "HEAD"],
                stdout=subprocess.PIPE).stdout
        rev = pipe.read().strip()
        # makes distutils blow up on Python 2.7
        if sys.version_info[0] >= 3:
            rev = rev.decode('ascii')

        FULLVERSION += "-%s" % rev
    except:
        warnings.warn("WARNING: Couldn't get git revision")
else:
    FULLVERSION += QUALIFIER


def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), 'nilmtk', 'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()
write_version_py()
# End of Version Check

setup(
    name='nilmtk',
    version=FULLVERSION,
    packages=find_packages(),
    install_requires=[],
    description='Estimate the energy consumed by individual appliances from '
                'whole-house power meter readings',
    author='nilmtk authors',
    author_email='',
    url='https://github.com/nilmtk/nilmtk',
    download_url="https://github.com/nilmtk/nilmtk/tarball/master#egg=nilmtk-dev",
    long_description=open('README.md').read(),
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache 2.0',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    keywords='smartmeters power electricity energy analytics redd '
             'disaggregation nilm nialm'
)
