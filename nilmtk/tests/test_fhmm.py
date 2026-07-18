import unittest
from os.path import join
from tempfile import TemporaryDirectory

from nilmtk import DataSet
from nilmtk.datastore import HDFDataStore
from nilmtk.legacy.disaggregate import FHMM

from .testingtools import data_dir


class TestFHMM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), 'co_test.h5')
        cls.dataset = DataSet(filename)

    @classmethod
    def tearDownClass(cls):
        cls.dataset.close()

    def test_fhmm_correctness(self):
        elec = self.dataset.buildings[1].elec
        fhmm = FHMM()
        fhmm.train(elec)
        mains = elec.mains()
        with TemporaryDirectory() as directory:
            output = HDFDataStore(join(directory, 'output.h5'), 'w')
            try:
                fhmm.disaggregate(mains, output, sample_period=1)

                for meter in range(2, 4):
                    key = f'/building1/elec/meter{meter}'
                    df1 = output.store.get(key)
                    df2 = self.dataset.store.store.get(key)

                    self.assertEqual(
                        (df1 == df2).sum().values[0], len(df1.index)
                    )
                    self.assertEqual(len(df1.index), len(df2.index))
            finally:
                output.close()

    def test_training_is_reproducible(self):
        elec = self.dataset.buildings[1].elec
        first = FHMM(random_state=7)
        second = FHMM(random_state=7)

        first.train(elec)
        second.train(elec)

        for meter in first.individual:
            self.assertTrue(
                (first.individual[meter].means_ ==
                 second.individual[meter].means_).all()
            )
            self.assertTrue(
                (first.individual[meter].transmat_ ==
                 second.individual[meter].transmat_).all()
            )


if __name__ == '__main__':
    unittest.main()
