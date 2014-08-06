#!/usr/bin/python
from __future__ import print_function, division
import unittest
from os.path import join
import pandas as pd
from datetime import timedelta
from testingtools import data_dir
from nilmtk.datastore import HDFDataStore
from nilmtk import TimeFrame
from nilmtk import DataSet, TimeFrame
from nilmtk.disaggregate import CombinatorialOptimisation
from nilmtk import HDFDataStore
from sh import rm


# do not edit! added by PythonBreakpoints
from pdb import set_trace as _breakpoint


class TestCO(unittest.TestCase):
    START_DATE = pd.Timestamp('2012-01-01 00:00:00', tz=None)
    NROWS = 1E4
    END_DATE = START_DATE + timedelta(seconds=NROWS - 1)
    TIMEFRAME = TimeFrame(START_DATE, END_DATE)

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'co_test.h5')
        cls.dataset = DataSet(filename)

    @classmethod
    def tearDownClass(cls):
        cls.dataset.store.close()

    def test_co_correctness(self):
        elec = self.dataset.buildings[1].elec
        co = CombinatorialOptimisation()
        co.train(elec)
        mains = elec.mains()
        output = HDFDataStore('output.h5', 'w')
        co.disaggregate(mains, output, resample_seconds=1)

        for meter in range(2,4):
            df1 = output.store.get('/building1/elec/meter{}'.format(meter))
            df2 = self.dataset.store.store.get('/building1/elec/meter{}'.format(meter))

            self.assertEqual((df1==df2).sum().values[0], len(df1.index))
            self.assertEqual(len(df1.index), len(df2.index))
        output.close()
        rm("output.h5")


        

if __name__ == '__main__':
    unittest.main()
