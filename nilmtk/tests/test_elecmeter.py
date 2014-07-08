#!/usr/bin/python
from __future__ import print_function, division
import unittest
from os.path import join
import pandas as pd
from datetime import timedelta
from .testingtools import data_dir, WarningTestMixin
from ..datastore import HDFDataStore
from ..elecmeter import ElecMeter, ElecMeterID
from ..stats.tests.test_totalenergy import check_energy_numbers

METER_ID = ElecMeterID(instance=1, building=1, dataset='REDD')
METER_ID2 = ElecMeterID(instance=2, building=1, dataset='REDD')
METER_ID3 = ElecMeterID(instance=3, building=1, dataset='REDD')

class TestElecMeter(WarningTestMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'energy.h5')
        cls.datastore = HDFDataStore(filename)
        ElecMeter.load_meter_devices(cls.datastore)
        cls.meter_meta = cls.datastore.load_metadata('building1')['elec_meters'][METER_ID.instance]

    @classmethod
    def tearDownClass(cls):
        cls.datastore.close()

    def test_load(self):
        meter = ElecMeter(store=self.datastore, metadata=self.meter_meta, 
                          meter_id=METER_ID)
        self.assertEqual(meter.metadata['device_model'], 'Energy Meter')
        self.assertEqual(meter.device['sample_period'], 10)

    def test_total_energy(self):
        meter = ElecMeter(meter_id=METER_ID)
        with self.assertRaises(RuntimeError):
            meter.total_energy()
        meter = ElecMeter(store=self.datastore, metadata=self.meter_meta, 
                          meter_id=METER_ID)
        energy = meter.total_energy()
        check_energy_numbers(self, energy)
        
    def test_upstream_meter(self):
        meter1 = ElecMeter(metadata={'site_meter': True}, meter_id=METER_ID)
        self.assertIsNone(meter1.upstream_meter())
        meter2 = ElecMeter(metadata={'submeter_of': 1}, meter_id=METER_ID2)
        self.assertIs(meter2.upstream_meter(), meter1)
        meter3 = ElecMeter(metadata={'submeter_of': 2}, meter_id=METER_ID3)
        self.assertIs(meter3.upstream_meter(), meter2)

if __name__ == '__main__':
    unittest.main()
