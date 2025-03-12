import unittest
from os.path import join

import pandas as pd

from nilmtk import DataSet
from nilmtk.legacy.disaggregate import CombinatorialOptimisation

from .testingtools import data_dir


class TestCO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), "co_test.h5")
        cls.dataset = DataSet(filename)

    @classmethod
    def tearDownClass(cls):
        cls.dataset.store.close()

    def test_co_correctness(self):
        elec = self.dataset.buildings[1].elec
        co = CombinatorialOptimisation()
        co.train(elec)
        mains = elec.mains()

        pred = co.disaggregate_chunk(next(mains.load(sample_period=1)))
        gt = {}
        for meter in elec.submeters().meters:
            gt[meter] = next(meter.load(sample_period=1)).squeeze()
        gt = pd.DataFrame(gt)
        pred = pred[gt.columns]
        self.assertTrue(gt.equals(pred))


if __name__ == "__main__":
    unittest.main()
