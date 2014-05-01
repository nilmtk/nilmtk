#!/usr/bin/python
from __future__ import print_function, division
import unittest
from nilmtk.pipeline import Pipeline, EnergyNode
from nilmtk import TimeFrame, EMeter, HDFDataStore, Loader
from nilmtk.tests.testingtools import data_dir
from os.path import join

class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'random.h5')
        cls.datastore = HDFDataStore(filename)
        cls.loader = Loader(store=cls.datastore, key='/building1/electric/meter1')

    def test_pipeline(self):
        meter = EMeter()
        meter.load(self.loader)
        nodes = [EnergyNode()]
        pipeline = Pipeline(nodes)
        pipeline.run(meter)
        meter.loader.mask = [TimeFrame('2012-01-01 00:00:00', '2012-01-01 01:00:00'),
                             TimeFrame('2012-01-01 01:00:00', '2012-01-01 02:00:00'),
                             TimeFrame('2012-01-01 02:00:00', '2012-01-01 03:00:00'),
                             TimeFrame('2012-01-01 03:00:00', '2012-01-01 04:00:00')]
        pipeline.run(meter)


if __name__ == '__main__':
    unittest.main()
