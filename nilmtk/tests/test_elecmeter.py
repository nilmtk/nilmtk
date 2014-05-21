#!/usr/bin/python
from __future__ import print_function, division
import unittest
from os.path import join
import pandas as pd
from datetime import timedelta
from .testingtools import data_dir, WarningTestMixin
from nilmtk.datastore import HDFDataStore
from nilmtk import ElecMeter
from nilmtk.pipeline.tests.test_energy import check_energy_numbers

METER_INSTANCE = 1

class TestElecMeter(WarningTestMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'energy.h5')
        cls.datastore = HDFDataStore(filename)
        ElecMeter.load_meter_devices(cls.datastore)
        cls.meter_meta = cls.datastore.load_metadata('building1')['elec_meters'][METER_INSTANCE]

    @classmethod
    def tearDownClass(cls):
        cls.datastore.close()

    def test_load(self):
        meter = ElecMeter(store=self.datastore, metadata=self.meter_meta, 
                          meter_instance=METER_INSTANCE)
        self.assertEqual(meter.metadata['device_model'], 'Energy Meter')
        self.assertEqual(meter.device['sample_period'], 10)

    def test_total_energy(self):
        meter = ElecMeter()
        with self.assertRaises(RuntimeError):
            meter.total_energy()
        meter = ElecMeter(store=self.datastore, metadata=self.meter_meta, 
                          meter_instance=METER_INSTANCE)
        energy = meter.total_energy()
        check_energy_numbers(self, energy)
        
    def test_upstream_meter(self):
        meter1 = ElecMeter(metadata={'site_meter': True, 'dataset': 'REDD', 
                                     'building': 1, 'instance': 1})
        with self.assertRaises(ValueError):
            meter1.upstream_meter
        meter2 = ElecMeter(metadata={'submeter_of': 1, 'dataset': 'REDD', 
                                     'building': 1, 'instance': 2})
        self.assertIs(meter2.upstream_meter, meter1)
        meter3 = ElecMeter(metadata={'submeter_of': 2, 'dataset': 'REDD', 
                                     'building': 1, 'instance': 3})
        self.assertIs(meter3.upstream_meter, meter2)

if __name__ == '__main__':
    unittest.main()
