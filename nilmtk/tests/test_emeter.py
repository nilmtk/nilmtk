#!/usr/bin/python
from __future__ import print_function, division
import unittest
from os.path import join
import pandas as pd
from datetime import timedelta
from testingtools import data_dir, WarningTestMixin
from nilmtk.datastore import HDFDataStore
from nilmtk.loader import Loader
from nilmtk import EMeter

class TestEMeter(WarningTestMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'random.h5')
        cls.datastore = HDFDataStore(filename)
        cls.loader = Loader(store=cls.datastore, key='/building1/electric/meter1')

    @classmethod
    def tearDownClass(cls):
        cls.datastore.close()

    def test_load(self):
        meter = EMeter()
        meter.load(self.loader)
        self.assertEqual(meter.metadata['device_name'], 'EnviR')
        self.assertEqual(meter.metadata['device']['sample_period'], 6)

if __name__ == '__main__':
    unittest.main()
