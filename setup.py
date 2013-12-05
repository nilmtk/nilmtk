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

CYTHON_DIR = 'nilmtk'

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
    extensions = [Extension("nilmtk._feature_detectors", 
                            sources=sources)]
    ext_modules = cythonize(extensions)
else:
    ext_modules = [
        Extension("nilmtk._feature_detectors", 
                  [join(CYTHON_DIR, '_feature_detectors.c')]),
    ]

setup(
    name='nilmtk',
    version='0.001',
    packages = find_packages(),
    ext_modules = ext_modules,
    install_requires = ['numpy', 'pandas', 'matplotlib'],
    description='Estimate the energy consumed by individual appliances from '
                'whole-house power meter readings',
    author='nilmtk authors',
    author_email='',
    url='https://github.com/nilmtk/nilmtk',
    download_url = "https://github.com/nilmtk/nilmtk/tarball/master#egg=nilmtk-dev",
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
