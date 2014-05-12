#!/usr/bin/python
from __future__ import print_function, division
import unittest
from os.path import join
import pandas as pd
from datetime import timedelta
from testingtools import data_dir, WarningTestMixin
from nilmtk.datastore import HDFDataStore
from nilmtk import ElectricityMeter
from nilmtk.pipeline.tests.test_energy import check_energy_numbers

KEY = '/building1/electric/meter1'

class TestElectricityMeter(WarningTestMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'energy.h5')
        cls.datastore = HDFDataStore(filename)

    @classmethod
    def tearDownClass(cls):
        cls.datastore.close()

    def test_load(self):
        meter = ElectricityMeter(1, 1, 'REDD')
        meter.load(self.datastore, key=KEY)
        self.assertEqual(meter.metadata['device_model'], 'Energy Meter')
        self.assertEqual(meter.metadata['device']['sample_period'], 10)

    def test_total_energy(self):
        meter = ElectricityMeter(1, 1, 'REDD')
        with self.assertRaises(RuntimeError):
            meter.total_energy()
        meter.load(self.datastore, KEY)
        energy = meter.total_energy()
        check_energy_numbers(self, energy)
        
    def test_upstream_meter(self):
        meter1 = ElectricityMeter(1,1,'REDD')
        meter2 = ElectricityMeter(2,1,'REDD',metadata={'submeter_of': 1})
        self.assertIs(meter2.upstream_meter, meter1)

if __name__ == '__main__':
    unittest.main()
