#!/usr/bin/python
from __future__ import print_function, division
import unittest
from nilmtk.pipeline import Pipeline, EnergyNode
from nilmtk.pipeline.energynode import _energy_per_power_series
from nilmtk import TimeFrame, EMeter, HDFDataStore, Loader
from nilmtk.consts import JOULES_PER_KWH
from nilmtk.tests.testingtools import data_dir
from os.path import join
import numpy as np
import pandas as pd
from datetime import timedelta

class TestEnergy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'energy.h5')
        cls.datastore = HDFDataStore(filename)
        cls.loader = Loader(store=cls.datastore, key='/building1/electric/meter1')

    def test_energy_per_power_series(self):
        data = np.array([0,  0,  0, 100, 100, 100, 150, 150, 200,   0,   0, 100, 5000,    0])
        secs = np.arange(start=0, stop=len(data)*10, step=10)
        true_kwh = ((data[:-1] * np.diff(secs)) / JOULES_PER_KWH).sum()
        index = [pd.Timestamp('2010-01-01') + timedelta(seconds=sec) for sec in secs]
        df = pd.Series(data=data, index=index)
        self.assertAlmostEqual(true_kwh, _energy_per_power_series(df))

    def test_pipeline(self):
        meter = EMeter()
        meter.load(self.loader)
        nodes = [EnergyNode()]
        pipeline = Pipeline(nodes)
        pipeline.run(meter)
        energy = pipeline.results['energy'].combined
        true_active_kwh =  0.0163888888889
        self.assertAlmostEqual(energy['active'], true_active_kwh)
        self.assertAlmostEqual(energy['reactive'], true_active_kwh*0.9)
        self.assertAlmostEqual(energy['apparent'], true_active_kwh*1.1)


if __name__ == '__main__':
    unittest.main()
