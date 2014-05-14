#!/usr/bin/python
from __future__ import print_function, division
import unittest
from os.path import join
import pandas as pd
from datetime import timedelta
from .testingtools import data_dir, WarningTestMixin
from nilmtk.datastore import HDFDataStore
from nilmtk import ElectricityMeter, Mains
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
        meter = ElectricityMeter()
        meter.load(self.datastore, keys=[KEY])
        self.assertEqual(meter.metadata['device_model'], 'Energy Meter')
        self.assertEqual(meter.metadata['device']['sample_period'], 10)

    def test_total_energy(self):
        meter = ElectricityMeter()
        with self.assertRaises(RuntimeError):
            meter.total_energy()
        meter.load(self.datastore, keys=[KEY])
        energy = meter.total_energy()
        check_energy_numbers(self, energy)
        
    def test_upstream_meter(self):
        meter1 = ElectricityMeter({'site_meter': True, 'dataset': 'REDD', 
                                   'building': 1, 'instance': 1})
        with self.assertRaises(ValueError):
            meter1.upstream_meter
        meter2 = ElectricityMeter({'submeter_of': 1, 'dataset': 'REDD', 
                                   'building': 1, 'instance': 2})
        mains = Mains(1, 'REDD', meters=[meter1])
        meter2.mains = mains
        self.assertIs(meter2.upstream_meter, mains)
        meter3 = ElectricityMeter({'submeter_of': 2, 'dataset': 'REDD', 
                                   'building': 1, 'instance': 3})
        self.assertIs(meter3.upstream_meter, meter2)

if __name__ == '__main__':
    unittest.main()
