#!/usr/bin/python
from __future__ import print_function, division
import unittest
from nilmtk.pipeline import Pipeline, EnergyNode, LocateGoodSectionsNode
from nilmtk.pipeline.locategoodsectionsnode import reframe_index
from nilmtk.pipeline.energynode import _energy_per_power_series
from nilmtk import TimeFrame, EMeter, HDFDataStore, Loader
from nilmtk.consts import JOULES_PER_KWH
from nilmtk.tests.testingtools import data_dir
from os.path import join
import numpy as np
import pandas as pd
from datetime import timedelta

class TestLocateGaps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'energy_complex.h5')
        cls.datastore = HDFDataStore(filename)
        cls.loader = Loader(store=cls.datastore, key='/building1/electric/meter1')

    def test_pipeline(self):
        meter = EMeter()
        meter.load(self.loader)
        nodes = [LocateGoodSectionsNode()]
        pipeline = Pipeline(nodes)
        pipeline.run(meter)

    def test_reframe_index(self):
        # First test: don't specify a window
        index = pd.date_range('2011-01-01', freq='S', end='2011-02-01')
        reframed = reframe_index(index)
        np.testing.assert_array_equal(index, reframed)
        
        # Second test: grow the window at the start
        reframed = reframe_index(index, window_start=pd.Timestamp('2010-01-01'))
        self.assertEqual(reframed[0], pd.Timestamp('2010-01-01'))
        self.assertEqual(len(reframed), len(index)+1)
        np.testing.assert_array_equal(index, reframed[1:])

        # Third test: grow the window at the end
        reframed = reframe_index(index, window_end=pd.Timestamp('2013-01-01'))
        self.assertEqual(reframed[-1], pd.Timestamp('2013-01-01'))
        self.assertEqual(len(reframed), len(index)+1)
        np.testing.assert_array_equal(index, reframed[:-1])

        # Grow the window at both ends
        reframed = reframe_index(index, 
                                 window_start=pd.Timestamp('2009-05-04'),
                                 window_end=pd.Timestamp('2013-02-04'))
        self.assertEqual(reframed[0], pd.Timestamp('2009-05-04'))
        self.assertEqual(reframed[-1], pd.Timestamp('2013-02-04'))
        self.assertEqual(len(reframed), len(index)+2)
        np.testing.assert_array_equal(index, reframed[1:-1])

        # Shrink window at start and grow it at end
        start = pd.Timestamp('2011-01-03')
        end = pd.Timestamp('2013-01-05')
        reframed = reframe_index(index, window_start=start, window_end=end)
        self.assertEqual(reframed[0], start)
        self.assertEqual(reframed[-1], end)
        self.assertEqual(len(reframed), len(pd.date_range(start, freq='S', 
                                                          end='2011-02-01'))+1)

        # Shrink the window at both ends (and test TZ)
        TZ = 'Europe/London'
        start = pd.Timestamp('2011-01-03', tz=TZ)
        end = pd.Timestamp('2011-01-05', tz=TZ)
        index = index.tz_localize(TZ)
        reframed = reframe_index(index, window_start=start, window_end=end)
        self.assertEqual(reframed[0], start)
        self.assertEqual(reframed[-1], end)
        self.assertEqual(len(reframed), len(pd.date_range(start, freq='S', end=end)))
        start_i = np.where(index == start)[0][0]
        end_i = np.where(index == end)[0][0]
        np.testing.assert_array_equal(index[start_i:end_i+1], reframed)
        self.assertEqual(reframed.tzinfo.zone, TZ)



if __name__ == '__main__':
    unittest.main()
