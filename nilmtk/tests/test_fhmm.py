import unittest
from os.path import join
from os import remove
from .testingtools import data_dir
from nilmtk.datastore import HDFDataStore
from nilmtk import DataSet
from nilmtk.legacy.disaggregate import FHMM


class TestFHMM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'co_test.h5')
        cls.dataset = DataSet(filename)

    @classmethod
    def tearDownClass(cls):
        cls.dataset.store.close()

    def test_fhmm_correctness(self):
        elec = self.dataset.buildings[1].elec
        fhmm = FHMM()
        fhmm.train(elec)
        mains = elec.mains()
        output = HDFDataStore('output.h5', 'w')
        fhmm.disaggregate(mains, output, sample_period=1)

        for meter in range(2, 4):
            df1 = output.store.get('/building1/elec/meter{}'.format(meter))
            df2 = self.dataset.store.store.get(
                '/building1/elec/meter{}'.format(meter))

            self.assertEqual((df1 == df2).sum().values[0], len(df1.index))
            self.assertEqual(len(df1.index), len(df2.index))
        output.close()
        remove("output.h5")


if __name__ == '__main__':
    unittest.main()
