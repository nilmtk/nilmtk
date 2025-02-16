import unittest
from os.path import join

from nilmtk import (
    Appliance,
    DataSet,
    ElecMeter,
    HDFDataStore,
    MeterGroup,
    TimeFrame,
    global_meter_group,
)
from nilmtk.building import BuildingID
from nilmtk.elecmeter import ElecMeterID
from nilmtk.legacy.disaggregate import CombinatorialOptimisation
from nilmtk.metrics import f1_score
from nilmtk.tests.testingtools import data_dir
from nilmtk.utils import nodes_adjacent_to_root, tree_root


class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filename = join(data_dir(), "co_test.h5")
        cls.dataset = DataSet(filename)

    @classmethod
    def tearDownClass(cls):
        cls.dataset.store.close()

    def test_f1(self):
        pass
        # The code below doesn't work yet because it complains that
        # AttributeError: Attribute 'metadata' does not exist in node: '/'
        """
        co = CombinatorialOptimisation()
        co.train(self.dataset.buildings[1].elec)
        disag_filename = join(data_dir(), 'co-disag.h5')
        output = HDFDataStore(disag_filename, 'w')
        co.disaggregate(self.dataset.buildings[1].elec.mains(), output)
        output.close()
        disag = DataSet(disag_filename)
        disag_elec = disag.buildings[1].elec
        f1 = f1_score(disag_elec, self.dataset.buildings[1].elec)
        """


if __name__ == "__main__":
    unittest.main()
