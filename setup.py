"""
   Copyright 2013-2019 NILMTK developers

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
# import numpy

TRAVIS_TAG = os.environ.get('TRAVIS_TAG', '')

if TRAVIS_TAG:
    #TODO: validate if the tag is a valid version number
    VERSION = TRAVIS_TAG
else:
    MAJOR = 0
    MINOR = 4
    MICRO = 0
    DEV = 1 # For multiple dev pre-releases, please increment this value
    ISRELEASED = False
    VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
    QUALIFIER = ''


FULLVERSION = VERSION
if not TRAVIS_TAG or not ISRELEASED:
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

        # Use a local version tag to include the git revision
        FULLVERSION += ".dev{}+git.{}".format(DEV, rev)
    except:
        FULLVERSION += ".dev{}".format(DEV)
        warnings.warn('WARNING: Could not get the git revision, version will be "{}"'.format(FULLVERSION))
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
    package_data={'': ['*.yaml']},
    install_requires=[
        'future',
        'six',
        'psycopg2-binary',
        'pandas==0.24.2',
        'networkx==2.1',
        'scipy',
        'tables',
        'scikit-learn>=0.21.2',
        'hmmlearn>=0.2.1',
        'pyyaml',
        'matplotlib>=2.2.0',
        'jupyter'
    ],
    description='Estimate the energy consumed by individual appliances from '
                'whole-house power meter readings',
    author='NILMTK developers',
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
