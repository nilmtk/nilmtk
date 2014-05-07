#!/usr/bin/python
from __future__ import print_function, division
import unittest
from os.path import join
import pandas as pd
from datetime import timedelta
from testingtools import data_dir, WarningTestMixin
from nilmtk.datastore import HDFDataStore
from nilmtk import EMeter
from nilmtk.pipeline.tests.test_energy import check_energy_numbers

KEY = '/building1/electric/meter1'

class TestEMeter(WarningTestMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'energy.h5')
        cls.datastore = HDFDataStore(filename)

    @classmethod
    def tearDownClass(cls):
        cls.datastore.close()

    def test_load(self):
        meter = EMeter()
        meter.load(self.datastore, key=KEY)
        self.assertEqual(meter.metadata['device_model'], 'Energy Meter')
        self.assertEqual(meter.metadata['device']['sample_period'], 10)

    def test_total_energy(self):
        meter = EMeter()
        with self.assertRaises(RuntimeError):
            meter.total_energy()
        meter.load(self.datastore, KEY)
        energy = meter.total_energy()
        check_energy_numbers(self, energy)
        

if __name__ == '__main__':
    unittest.main()
