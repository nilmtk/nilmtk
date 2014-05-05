#!/usr/bin/python
from __future__ import print_function, division
import unittest
from os.path import join
import pandas as pd
from datetime import timedelta
from testingtools import data_dir
from nilmtk.datastore import HDFDataStore
from nilmtk import TimeFrame
from nilmtk.measurement import Power, Voltage, Energy

class TestHDFDataStore(unittest.TestCase):
    START_DATE = pd.Timestamp('2012-01-01 00:00:00', tz=None)
    NROWS = 1E4
    END_DATE = START_DATE + timedelta(seconds=NROWS-1)
    TIMEFRAME = TimeFrame(START_DATE, END_DATE)

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'random.h5')
        cls.datastore = HDFDataStore(filename)
        cls.keys = ['/building1/electric/meter{:d}'.format(i) 
                    for i in range(1,6)]

    @classmethod
    def tearDownClass(cls):
        cls.datastore.close()

    def test_keys(self):
        self.assertEqual(self.datastore.keys(), self.keys)

    def test_column_names(self):
        for key in self.keys:
            self.assertEqual(self.datastore.column_names(key), 
                             [Power('active'), Energy('reactive'), Voltage()])

    def test_timeframe(self):
        self.datastore.window.clear()
        for key in self.keys:
            self.assertEqual(self.datastore.timeframe(key), self.TIMEFRAME)

        self._apply_mask()
        for key in self.keys:
            self.datastore.window.enabled = False
            self.assertEqual(self.datastore.timeframe(key), self.TIMEFRAME)
            self.datastore.window.enabled = True
            self.assertEqual(self.datastore.timeframe(key), self.datastore.window)

    def test_n_rows(self):
        self._apply_mask()
        for key in self.keys:
            self.datastore.window.enabled = True
            self.assertEqual(self.datastore.nrows(key), 10*60)
            self.datastore.window.enabled = False
            self.assertEqual(self.datastore.nrows(key), self.NROWS)

    def test_estimate_memory_requirement(self):
        self._apply_mask()
        for key in self.keys:
            self.datastore.window.enabled = True
            mem = self.datastore.estimate_memory_requirement(key, self.datastore.nrows(key))
            self.assertEqual(mem, 12000)
            self.datastore.window.enabled = False
            mem = self.datastore.estimate_memory_requirement(key, self.datastore.nrows(key))
            self.assertEqual(mem, 200000)

    def test_load(self):
        timeframe = TimeFrame('2012-01-01 00:00:00', '2012-01-01 00:00:05')
        self.datastore.window.clear()
        gen = self.datastore.load(key=self.keys[0], 
                                  cols=[Power('active')],
                                  periods=[timeframe])
        df = next(gen)
        self.assertEqual(df.index[0], timeframe.start)
        self.assertEqual(df.index[-1], timeframe.end - timedelta(seconds=1))
        self.assertEqual(df.look_ahead.index[0], timeframe.end)
        self.assertEqual(len(df.look_ahead), 10)


    #--------- helper functions ---------------------#

    def _apply_mask(self):
        self.datastore.window = TimeFrame('2012-01-01 00:10:00',
                                        '2012-01-01 00:20:00')

    
if __name__ == '__main__':
    unittest.main()
