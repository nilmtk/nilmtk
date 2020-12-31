import numpy as np
import pandas as pd
import unittest

from .testingtools import data_dir
from datetime import timedelta
from nilmtk.datastore import HDFDataStore, CSVDataStore, TmpDataStore
from nilmtk import TimeFrame
from os.path import join, isfile


# class name can't begin with test
class SuperTestDataStore(object):
    START_DATE = pd.Timestamp('2012-01-01 00:00:00', tz=None)
    NROWS = 1E4
    END_DATE = START_DATE + timedelta(seconds=NROWS-1)
    TIMEFRAME = TimeFrame(START_DATE, END_DATE)

    def test_timeframe(self):
        self.datastore.window.clear()
        for key in self.keys:
            self.assertEqual(self.datastore.get_timeframe(key), self.TIMEFRAME)

        self._apply_mask()
        for key in self.keys:
            self.datastore.window.enabled = False
            self.assertEqual(self.datastore.get_timeframe(key), self.TIMEFRAME)
            self.datastore.window.enabled = True
            self.assertEqual(self.datastore.get_timeframe(key), self.datastore.window)

    def test_load(self):
        timeframe = TimeFrame('2012-01-01 00:00:00', '2012-01-01 00:00:05')
        self.datastore.window.clear()
        gen = self.datastore.load(key=self.keys[0],
                                  columns=[('power', 'active')],
                                  sections=[timeframe],
                                  n_look_ahead_rows=10)
        df = next(gen)
        self.assertEqual(df.index[0], timeframe.start)
        self.assertEqual(df.index[-1], timeframe.end - timedelta(seconds=1))
        self.assertEqual(df.look_ahead.index[0], timeframe.end) # This test, for some odd reason, fails for CSVDataStore in Python3, intermittently.  Very odd.
        self.assertEqual(len(df.look_ahead), 10)

    def test_load_chunks(self):
        self.datastore.window.clear()
        chunks = self.datastore.load(key=self.keys[0])
        i = 0
        for chunk in chunks:
            i += 1
            self.assertEqual(chunk.index[0], self.TIMEFRAME.start)
            self.assertEqual(chunk.index[-1], self.TIMEFRAME.end)
            self.assertEqual(len(chunk), self.NROWS)
        self.assertEqual(i, 1)

        timeframes = [TimeFrame('2012-01-01 00:00:00', '2012-01-01 00:00:05'),
                      TimeFrame('2012-01-01 00:10:00', '2012-01-01 00:10:05')]
        chunks = self.datastore.load(key=self.keys[0], sections=timeframes)
        i = 0
        for chunk in chunks:
            self.assertEqual(chunk.index[0], timeframes[i].start)
            self.assertEqual(chunk.index[-1], timeframes[i].end-timedelta(seconds=1))
            self.assertEqual(len(chunk), 5)
            i += 1
        self.assertEqual(i, 2)

        # Check when we have a narrow mask
        self.datastore.window = TimeFrame('2012-01-01 00:10:02', '2012-01-01 00:10:10')
        chunks = self.datastore.load(key=self.keys[0], sections=timeframes)
        i = 0
        for chunk in chunks:
            if chunk.empty:
                continue
            self.assertEqual(chunk.index[0], pd.Timestamp('2012-01-01 00:10:02'))
            self.assertEqual(chunk.index[-1], pd.Timestamp('2012-01-01 00:10:04'))
            self.assertEqual(len(chunk), 3)
            i += 1
        self.assertEqual(i, 1)

    def test_load_chunks_small_chunksize(self):
        self.datastore.window.clear()
        timeframes = [TimeFrame('2012-01-01 00:00:00', '2012-01-01 00:01:00'),
                      TimeFrame('2012-01-01 00:10:00', '2012-01-01 00:11:00')]
        chunks = self.datastore.load(key=self.keys[0], sections=timeframes,
                                     chunksize=20)
        one_sec = timedelta(seconds=1)

        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.index[0], chunk.timeframe.start)
            self.assertTrue((chunk.timeframe.end - one_sec) <= 
                            chunk.index[-1] <= 
                            chunk.timeframe.end)        

    #--------- helper functions ---------------------#

    def _apply_mask(self):
        self.datastore.window = TimeFrame('2012-01-01 00:10:00',
                                        '2012-01-01 00:20:00')

class TestHDFDataStore(unittest.TestCase, SuperTestDataStore):

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'random.h5')
        cls.datastore = HDFDataStore(filename)
        cls.keys = ['/building1/elec/meter{:d}'.format(i) for i in range(1, 6)]
                                        
    @classmethod
    def tearDownClass(cls):
        cls.datastore.close()

    def test_column_names(self):
        for key in self.keys:
            self.assertEqual(self.datastore._column_names(key), 
                             [('power', 'active'), ('energy', 'reactive'),
                              ('voltage', '')])

    def test_n_rows(self):
        self._apply_mask()
        for key in self.keys:
            self.datastore.window.enabled = True
            self.assertEqual(self.datastore._nrows(key), 10*60)
            self.datastore.window.enabled = False
            self.assertEqual(self.datastore._nrows(key), self.NROWS)

    def test_estimate_memory_requirement(self):
        self._apply_mask()
        for key in self.keys:
            self.datastore.window.enabled = True
            mem = self.datastore._estimate_memory_requirement(key, self.datastore._nrows(key))
            self.assertEqual(mem, 12000)
            self.datastore.window.enabled = False
            mem = self.datastore._estimate_memory_requirement(key, self.datastore._nrows(key))
            self.assertEqual(mem, 200000)

class TestCSVDataStore(unittest.TestCase, SuperTestDataStore):

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'random_csv')
        cls.datastore = CSVDataStore(filename)
        cls.keys = ['/building1/elec/meter{:d}'.format(i) for i in range(1, 6)]

    @classmethod
    def tearDownClass(cls):
        cls.datastore.close()


class TestTmpDataStore(unittest.TestCase, SuperTestDataStore):
    @classmethod
    def setUpClass(cls):
        cls.datastore = TmpDataStore()
        cls.keys = ['/building1/elec/meter{:d}'.format(i) for i in range(1, 6)]
        for key in cls.keys:
            col_names = [ ( "power", "active" ) ]
            n_rows = int(SuperTestDataStore.NROWS)
            idx = pd.date_range(
                    start=SuperTestDataStore.START_DATE,
                    end=SuperTestDataStore.END_DATE,
                    periods=n_rows
            )
            data = pd.DataFrame(100 * np.random.rand(n_rows, 1), index=idx)
            data.columns = pd.MultiIndex.from_tuples(col_names)
            cls.datastore.put(key, data)

    @classmethod
    def tearDownClass(cls):
        cls.datastore.close()


if __name__ == '__main__':
    unittest.main()

