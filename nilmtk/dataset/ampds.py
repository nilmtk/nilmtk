from __future__ import print_function, division
import os
import pandas as pd
from nilmtk.dataset import DataSet
from nilmtk.utils import get_immediate_subdirectories
from nilmtk.building import Building


class AMPDS(DataSet):
    """Load data from AMPDS."""

    def __init__(self):
        super(AMPDS, self).__init__()
        self.urls = ['http://ampds.org/']
        self.citations = ['Stephen Makonin, Fred Popowich, Lyn Bartram, '
                        'Bob Gill, and Ivan V. Bajic,'
                        'AMPds: A Public Dataset for Load Disaggregation and'
                        'Eco-Feedback Research, in Electrical Power and Energy'
                        'Conference (EPEC), 2013 IEEE, pp. 1-6, 2013.'
                        ]

    def load_electricity(self, root_directory):
        return None

    def load_water(self, root_directory):
        return None

    def load_gas(self, root_directory):
        return None

    def load(self, root_directory):
        return None




