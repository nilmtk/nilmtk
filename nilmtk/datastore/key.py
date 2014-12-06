from __future__ import print_function, division
import pandas as pd
from itertools import repeat, tee
from time import time
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import yaml
from os.path import isdir, isfile, join, exists, dirname
from os import listdir, makedirs, remove
from shutil import rmtree
import re
from nilm_metadata.convert_yaml_to_hdf5 import _load_file
from nilmtk.timeframe import TimeFrame, timeframes_from_periodindex
from nilmtk.node import Node
from .datastore import DataStore

# do not edit! added by PythonBreakpoints
from pdb import set_trace as _breakpoint


MAX_MEM_ALLOWANCE_IN_BYTES = 2**29 # 512 MBytes

class Key(object):
    """A location of data or metadata within NILMTK.
    
    Attributes
    ----------
    building : int
    meter : int
    utility : str
    """

    def __init__(self, string=None, building=None, meter=None):
        """
        Parameters
        ----------
        string : str, optional
            e.g. 'building1/elec/meter1'
        building : int, optional
        meter : int, optional
        """
        self.utility = None
        if string is None:
            self.building = building
            self.meter = meter
        else:
            split = string.strip('/').split('/')
            assert split[0].startswith('building'), "The first element must be 'building<I>', e.g. 'building1'; not '{}'.".format(split[0])
            try:
                self.building = int(split[0].replace("building", ""))
            except ValueError as e:
                raise ValueError("'building' must be followed by an integer.\n{}"
                                 .format(e))
            if len(split) > 1:
                self.utility = split[1]
            if len(split) == 3:
                assert split[2].startswith('meter')
                self.meter = int(split[-1].replace("meter", ""))
            else:
                self.meter = None
        self._check()

    def _check(self):
        assert isinstance(self.building, int)
        assert self.building >= 1
        if self.meter is not None:
            assert isinstance(self.meter, int)
            assert self.meter >= 1

    def __repr__(self):
        self._check()
        s = "/building{:d}".format(self.building)
        if self.meter is not None:
            s += "/elec/meter{:d}".format(self.meter)
        return s