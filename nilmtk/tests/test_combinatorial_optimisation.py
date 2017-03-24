#!/usr/bin/python
from __future__ import print_function, division
import unittest
from os.path import join
from os import remove
import pandas as pd
from .testingtools import data_dir
from nilmtk.datastore import HDFDataStore
from nilmtk import DataSet
from nilmtk.disaggregate import CombinatorialOptimisation


class TestCO(unittest.TestCase):

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
        pred = co.disaggregate_chunk(mains.load(sample_period=1).next())
        gt = {}
        for meter in elec.submeters().meters:
            gt[meter] = meter.load(sample_period=1).next().squeeze()
        gt = pd.DataFrame(gt)
        pred = pred[gt.columns]
        self.assertTrue(gt.equals(pred))


if __name__ == '__main__':
    unittest.main()
