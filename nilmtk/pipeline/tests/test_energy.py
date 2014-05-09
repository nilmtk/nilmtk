#!/usr/bin/python
from __future__ import print_function, division
import unittest
from nilmtk.pipeline import Pipeline, EnergyNode, ClipNode
from nilmtk.pipeline.energynode import _energy_for_power_series
from nilmtk import TimeFrame, ElectricityMeter, HDFDataStore
from nilmtk.consts import JOULES_PER_KWH
from nilmtk.tests.testingtools import data_dir
from os.path import join
import numpy as np
import pandas as pd
from datetime import timedelta

KEY = '/building1/electric/meter1'

def check_energy_numbers(testcase, energy):
    energy = energy.combined
    true_active_kwh =  0.0163888888889
    testcase.assertAlmostEqual(energy['active'], true_active_kwh)
    testcase.assertAlmostEqual(energy['reactive'], true_active_kwh*0.9)
    testcase.assertAlmostEqual(energy['apparent'], true_active_kwh*1.1)


class TestEnergy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'energy.h5')
        cls.datastore = HDFDataStore(filename)

    def test_energy_per_power_series(self):
        data = np.array([0,  0,  0, 100, 100, 100, 150, 150, 200,   0,   0, 100, 5000,    0])
        secs = np.arange(start=0, stop=len(data)*10, step=10)
        true_kwh = ((data[:-1] * np.diff(secs)) / JOULES_PER_KWH).sum()
        index = [pd.Timestamp('2010-01-01') + timedelta(seconds=sec) for sec in secs]
        df = pd.Series(data=data, index=index)
        kwh = _energy_for_power_series(df, max_sample_period=15)
        self.assertAlmostEqual(true_kwh, kwh)

    def test_pipeline(self):
        meter = ElectricityMeter()
        meter.load(self.datastore, KEY)
        nodes = [ClipNode(), EnergyNode()]
        pipeline = Pipeline(nodes)
        pipeline.run(meter)
        energy = pipeline.results['energy']
        check_energy_numbers(self, energy)


if __name__ == '__main__':
    unittest.main()
