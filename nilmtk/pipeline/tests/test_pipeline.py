#!/usr/bin/python
from __future__ import print_function, division
import unittest
from nilmtk.pipeline import Pipeline, EnergyNode, ClipNode
from nilmtk import TimeFrame, ElecMeter, HDFDataStore
from nilmtk.tests.testingtools import data_dir
from os.path import join

METER_INSTANCE = 1

class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'random.h5')
        cls.datastore = HDFDataStore(filename)
        ElecMeter.load_meter_devices(cls.datastore)
        cls.meter_meta = cls.datastore.load_metadata('building1')['elec_meters'][METER_INSTANCE]

    def test_pipeline(self):
        meter = ElecMeter(store=self.datastore, metadata=self.meter_meta, 
                          meter_instance=METER_INSTANCE)
        nodes = [ClipNode(), EnergyNode()]
        pipeline = Pipeline(nodes)
        pipeline.run(meter)
        meter.store.mask = [TimeFrame('2012-01-01 00:00:00', '2012-01-01 01:00:00'),
                             TimeFrame('2012-01-01 01:00:00', '2012-01-01 02:00:00'),
                             TimeFrame('2012-01-01 02:00:00', '2012-01-01 03:00:00'),
                             TimeFrame('2012-01-01 03:00:00', '2012-01-01 04:00:00')]
        pipeline.run(meter)


if __name__ == '__main__':
    unittest.main()
