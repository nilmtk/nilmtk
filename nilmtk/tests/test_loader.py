#!/usr/bin/python
from __future__ import print_function, division
import unittest
from os.path import join
import pandas as pd
from datetime import timedelta
from testingtools import data_dir
from nilmtk.datastore import HDFDataStore
from nilmtk.loader import Loader
from nilmtk import TimeFrame

class TestLoader(unittest.TestCase):
    START_DATE = pd.Timestamp('2012-01-01 00:00:00', tz=None)
    NROWS = 1E4
    END_DATE = START_DATE + timedelta(seconds=NROWS-1)
    TIMEFRAME = TimeFrame(START_DATE, END_DATE)

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'random.h5')
        cls.datastore = HDFDataStore(filename)
        cls.loader = Loader(store=cls.datastore, key='/building1/electric/meter1')

    @classmethod
    def tearDownClass(cls):
        cls.datastore.close()

    def test_load_chunks(self):
        self.datastore.window.clear()
        chunks = self.loader.load()
        i = 0
        for chunk in chunks:
            i += 1
            self.assertEqual(chunk.index[0], self.TIMEFRAME.start)
            self.assertEqual(chunk.index[-1], self.TIMEFRAME.end)
            self.assertEqual(len(chunk), self.NROWS)
        self.assertEqual(i, 1)

        timeframes = [TimeFrame('2012-01-01 00:00:00', '2012-01-01 00:00:05'),
                      TimeFrame('2012-01-01 00:10:00', '2012-01-01 00:10:05')]
        self.loader.mask = timeframes
        chunks = self.loader.load()
        i = 0
        for chunk in chunks:
            self.assertEqual(chunk.index[0], timeframes[i].start)
            self.assertEqual(chunk.index[-1], timeframes[i].end-timedelta(seconds=1))
            self.assertEqual(len(chunk), 5)
            i += 1
        self.assertEqual(i, 2)

        # Check when we have a narrow mask
        self.datastore.window = TimeFrame('2012-01-01 00:10:02', '2012-01-01 00:10:10')
        self.loader.mask = timeframes
        chunks = self.loader.load()
        i = 0
        for chunk in chunks:
            if i == 0:
                self.assertTrue(chunk.empty)
            else:
                self.assertEqual(chunk.index[0], pd.Timestamp('2012-01-01 00:10:02'))
                self.assertEqual(chunk.index[-1], pd.Timestamp('2012-01-01 00:10:04'))
                self.assertEqual(len(chunk), 3)
            i += 1
        self.assertEqual(i, 2)


if __name__ == '__main__':
    unittest.main()
