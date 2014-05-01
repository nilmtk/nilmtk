#!/usr/bin/python
from __future__ import print_function, division
import unittest
from nilmtk.pipeline import Pipeline, EnergyNode, LocateGoodSectionsNode
from nilmtk.pipeline.energynode import _energy_per_power_series
from nilmtk import TimeFrame, EMeter, HDFDataStore, Loader
from nilmtk.consts import JOULES_PER_KWH
from nilmtk.tests.testingtools import data_dir
from os.path import join
import numpy as np
import pandas as pd
from datetime import timedelta

class TestLocateGaps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'energy_complex.h5')
        cls.datastore = HDFDataStore(filename)
        cls.loader = Loader(store=cls.datastore, key='/building1/electric/meter1')

    def test_pipeline(self):
        meter = EMeter()
        meter.load(self.loader)
        nodes = [LocateGoodSectionsNode()]
        pipeline = Pipeline(nodes)
        pipeline.run(meter)


if __name__ == '__main__':
    unittest.main()
